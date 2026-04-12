"""
Business tools and tool registry.

This module intentionally keeps tool execution separate from inference and RAG.
Changing the tool list later should mostly happen here.
"""
from __future__ import annotations

import csv
import inspect
import json
import random
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import CONFIG_DIR, DOMAIN_CFG, KB_DIR, logger, resolve_project_path
from .domain_utils import canonicalize_intent, canonicalize_slots
from .rag_assets import load_delivery_rag_entries, load_lens_rag_entries, next_time_slot_after, normalize_lookup


REFERENCE_RE = re.compile(r"\b\d{2}\s?\d{2}\b")
TIME_SLOT_RE = re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        logger.warning("Missing CSV KB file: %s", path)
        return []
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        logger.warning("Missing JSONL KB file: %s", path)
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize_phone(value: str | None) -> str:
    return "".join(ch for ch in str(value or "") if ch.isdigit())


def _draft_id(prefix: str = "DRF") -> str:
    return prefix + "-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))


def _compact_dict(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value not in (None, "", [], {})}


def _material_treatment_tokens(slots: dict[str, Any]) -> list[str]:
    values = []
    for key in ["product", "material", "treatment", "color", "reference", "diameter", "index"]:
        if slots.get(key):
            values.append(f"{key}={slots[key]}")
    for eye in ["od", "og"]:
        for suffix in ["sphere", "cyl", "axis"]:
            key = f"{eye}_{suffix}"
            if slots.get(key):
                values.append(f"{key}={slots[key]}")
    if slots.get("addition"):
        values.append(f"addition={slots['addition']}")
    return values


def _norm_lookup(value: str | None) -> str:
    return normalize_lookup(value)


def _extract_time_slots(text: str) -> list[str]:
    return sorted({match.group(0) for match in TIME_SLOT_RE.finditer(str(text or ""))})


def _delivery_context_from_schedule(schedule: dict[str, Any]) -> dict[str, Any]:
    return {
        "delivery_target": "optician_agency",
        "delivery_eta_policy": "approximate_agency_window",
        "delivery_schedule": {
            "agence": schedule.get("agence"),
            "secteur": schedule.get("secteur"),
            "next_slot": schedule.get("next_slot"),
            "tous_creneaux": schedule.get("tous_creneaux", []),
            "requested_slot": schedule.get("requested_slot"),
        },
    }


class KnowledgeBase:
    def __init__(self, root: Path | None = None) -> None:
        kb_root = root or KB_DIR
        self.lens_catalog = _read_csv(kb_root / "lens_catalog.csv")
        self.stores = _read_csv(kb_root / "stores.csv")
        self.policies = _read_jsonl(kb_root / "policies.jsonl")
        self.orders = _read_csv(kb_root / "orders_mock.csv")
        self.command_examples = _read_jsonl(CONFIG_DIR / "commandes_intents.jsonl")
        self.few_shot_examples = _read_jsonl(CONFIG_DIR / "few_shots.jsonl")
        self.command_orders: list[dict[str, Any]] = []
        self.reference_catalog: dict[str, dict[str, Any]] = {}
        self.delivery_index: list[dict[str, Any]] = []
        self.lens_rag_index: list[dict[str, Any]] = []

        for row in self.lens_catalog:
            row["price_min_dt"] = float(row.get("price_min_dt", 0) or 0)
            row["price_max_dt"] = float(row.get("price_max_dt", 0) or 0)

        for row in self.command_examples:
            intent = canonicalize_intent(row.get("intent"))
            slots = canonicalize_slots(row.get("slots", {}))
            if intent == "order_tracking" and slots.get("order_id"):
                self.command_orders.append(
                    _compact_dict(
                        {
                            "order_id": slots.get("order_id"),
                            "num_client": slots.get("num_client"),
                            "order_status": slots.get("status"),
                            "order_type": slots.get("order_type"),
                            "priority": slots.get("priority"),
                            "customer_name": slots.get("customer_name"),
                            "date": slots.get("date"),
                            "date_from": slots.get("date_from"),
                            "date_to": slots.get("date_to"),
                            "source": "commandes_intents",
                        }
                    )
                )

            if intent == "create_order" and slots.get("reference"):
                reference = str(slots["reference"]).replace(" ", "")
                entry = self.reference_catalog.setdefault(
                    reference,
                    {
                        "reference": reference,
                        "seen_count": 0,
                        "products": set(),
                        "materials": set(),
                        "treatments": set(),
                        "colors": set(),
                        "diameters": set(),
                        "examples": [],
                    },
                )
                entry["seen_count"] += 1
                if slots.get("product"):
                    entry["products"].add(str(slots["product"]))
                if slots.get("material"):
                    entry["materials"].add(str(slots["material"]))
                if slots.get("treatment"):
                    entry["treatments"].add(str(slots["treatment"]))
                if slots.get("color"):
                    entry["colors"].add(str(slots["color"]))
                if slots.get("diameter"):
                    entry["diameters"].add(str(slots["diameter"]))
                if len(entry["examples"]) < 3:
                    entry["examples"].append(
                        {
                            "user": row.get("user") or row.get("client") or row.get("opticien"),
                            "assistant": row.get("assistant") or row.get("agent"),
                        }
                    )

        for row in self.few_shot_examples:
            user_text = str(row.get("user") or row.get("client") or row.get("opticien") or "")
            assistant_text = str(row.get("assistant") or row.get("agent") or "")
            combined = f"{user_text} {assistant_text}"
            for match in REFERENCE_RE.findall(combined):
                reference = match.replace(" ", "")
                entry = self.reference_catalog.setdefault(
                    reference,
                    {
                        "reference": reference,
                        "seen_count": 0,
                        "products": set(),
                        "materials": set(),
                        "treatments": set(),
                        "colors": set(),
                        "diameters": set(),
                        "examples": [],
                    },
                )
                entry["seen_count"] += 1
                lowered = combined.lower()
                if "orma" in lowered:
                    entry["materials"].add("orma")
                if "marron" in lowered:
                    entry["colors"].add("marron")
                if "diamètre 70" in lowered or "diametre 70" in lowered:
                    entry["diameters"].add("70")
                if len(entry["examples"]) < 3:
                    entry["examples"].append({"user": user_text, "assistant": assistant_text})

        self._load_external_rag_indexes()

        logger.info(
            (
                "Business context loaded: %d products, %d stores, %d policies, %d orders, %d command examples, "
                "%d delivery RAG entries, %d lens RAG entries"
            ),
            len(self.lens_catalog),
            len(self.stores),
            len(self.policies),
            len(self.orders),
            len(self.command_examples),
            len(self.delivery_index),
            len(self.lens_rag_index),
        )

    def _register_delivery_entry(self, row: dict[str, Any], source_path: Path) -> None:
        metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
        agence = str(metadata.get("agence", "")).strip()
        secteur = str(metadata.get("secteur", "")).strip()
        slots = metadata.get("tous_creneaux", [])
        if isinstance(slots, str):
            slots = [slots]
        if not isinstance(slots, list):
            slots = []
        slots = [str(item).strip() for item in slots if str(item).strip()]
        if not slots:
            slots = _extract_time_slots(str(row.get("text", "")))

        delivery_entry = {
            "id": row.get("id"),
            "agence": agence,
            "secteur": secteur,
            "nb_livraisons_jour": metadata.get("nb_livraisons_jour"),
            "premier_creneau": metadata.get("premier_creneau") or (slots[0] if slots else None),
            "tous_creneaux": slots,
            "source": str(source_path),
        }
        if not delivery_entry["agence"] and not delivery_entry["secteur"]:
            return
        self.delivery_index.append(_compact_dict(delivery_entry))

    def _register_lens_entry(self, row: dict[str, Any], source_path: Path) -> None:
        metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
        lens_entry = {
            "id": row.get("id"),
            "code": str(metadata.get("code", "")).strip().upper(),
            "name": str(metadata.get("nom", "")).strip(),
            "brand": str(metadata.get("marque", "")).strip(),
            "geometry": str(metadata.get("geometrie", "")).strip(),
            "material": str(metadata.get("matiere", "")).strip(),
            "photochromic": str(metadata.get("photochromique", "")).strip(),
            "diameter": str(metadata.get("diametre", "")).strip(),
            "text": str(row.get("text", "")).strip(),
            "source": str(source_path),
        }
        if not lens_entry["code"] and not lens_entry["name"]:
            return
        self.lens_rag_index.append(_compact_dict(lens_entry))

    def _load_external_rag_indexes(self) -> None:
        self.delivery_index = load_delivery_rag_entries()
        self.lens_rag_index = load_lens_rag_entries()


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., dict[str, Any]]


class ToolRegistry:
    def __init__(self, kb: KnowledgeBase | None = None) -> None:
        self.kb = kb or KnowledgeBase()
        descriptions = DOMAIN_CFG.get("tool_descriptions", {})
        self._tools: dict[str, ToolDefinition] = {
            "get_price": ToolDefinition(
                name="get_price",
                description=descriptions.get("get_price", "Lookup pricing from the catalog."),
                parameters={
                    "type": "object",
                    "properties": {
                        "product": {"type": "string"},
                        "index": {"type": "string"},
                        "coating": {"type": "string"},
                        "treatment": {"type": "string"},
                        "city": {"type": "string"},
                    },
                    "required": ["product", "index"],
                },
                handler=self.get_price,
            ),
            "get_store_info": ToolDefinition(
                name="get_store_info",
                description=descriptions.get("get_store_info", "Lookup store information by city."),
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
                handler=self.get_store_info,
            ),
            "track_order": ToolDefinition(
                name="track_order",
                description=descriptions.get("track_order", "Track an order."),
                parameters={
                    "type": "object",
                    "properties": {
                        "num_client": {"type": "string"},
                        "order_id": {"type": "string"},
                        "priority": {"type": "string"},
                        "agence": {"type": "string"},
                        "city": {"type": "string"},
                        "secteur": {"type": "string"},
                        "requested_slot": {"type": "string"},
                        "delivery_slot": {"type": "string"},
                    },
                    "required": ["num_client", "order_id"],
                },
                handler=self.track_order,
            ),
            "create_order": ToolDefinition(
                name="create_order",
                description=descriptions.get("create_order", "Create a new order draft."),
                parameters={
                    "type": "object",
                    "properties": {
                        "num_client": {"type": "string"},
                        "product": {"type": "string"},
                        "material": {"type": "string"},
                        "treatment": {"type": "string"},
                        "color": {"type": "string"},
                        "reference": {"type": "string"},
                        "diameter": {"type": "string"},
                        "quantity": {"type": "string"},
                        "priority": {"type": "string"},
                        "index": {"type": "string"},
                    },
                    "required": ["num_client", "product"],
                },
                handler=self.create_order,
            ),
            "check_availability": ToolDefinition(
                name="check_availability",
                description=descriptions.get("check_availability", "Check availability hints for a lens request."),
                parameters={
                    "type": "object",
                    "properties": {
                        "product": {"type": "string"},
                        "material": {"type": "string"},
                        "treatment": {"type": "string"},
                        "reference": {"type": "string"},
                        "diameter": {"type": "string"},
                        "index": {"type": "string"},
                        "city": {"type": "string"},
                    },
                },
                handler=self.check_availability,
            ),
            "confirm_reference": ToolDefinition(
                name="confirm_reference",
                description=descriptions.get("confirm_reference", "Confirm a product reference from known order examples."),
                parameters={
                    "type": "object",
                    "properties": {"reference": {"type": "string"}},
                    "required": ["reference"],
                },
                handler=self.confirm_reference,
            ),
            "get_delivery_schedule": ToolDefinition(
                name="get_delivery_schedule",
                description=descriptions.get(
                    "get_delivery_schedule",
                    "Get delivery schedule windows by agency/sector.",
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "agence": {"type": "string"},
                        "city": {"type": "string"},
                        "secteur": {"type": "string"},
                        "requested_slot": {"type": "string"},
                        "delivery_slot": {"type": "string"},
                    },
                },
                handler=self.get_delivery_schedule,
            ),
            "lookup_lens_catalog": ToolDefinition(
                name="lookup_lens_catalog",
                description=descriptions.get(
                    "lookup_lens_catalog",
                    "Lookup lens details from RAG catalog by code/name/material/diameter.",
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "name": {"type": "string"},
                        "material": {"type": "string"},
                        "diameter": {"type": "string"},
                        "photochromic": {"type": "string"},
                    },
                },
                handler=self.lookup_lens_catalog,
            ),
            "book_appointment": ToolDefinition(
                name="book_appointment",
                description=descriptions.get("book_appointment", "Book an appointment."),
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "date": {"type": "string"},
                        "time_slot": {"type": "string"},
                        "phone": {"type": "string"},
                    },
                    "required": ["city", "date", "time_slot", "phone"],
                },
                handler=self.book_appointment,
            ),
        }

    def list_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for tool in self._tools.values():
            tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            )
        return tools

    def execute(self, name: str | None, args: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if not name:
            return None
        tool = self._tools.get(name)
        if tool is None:
            return {"status": "error", "message": f"Unknown tool: {name}"}
        try:
            provided_args = dict(args or {})
            signature = inspect.signature(tool.handler)
            accepts_var_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            if accepts_var_kwargs:
                filtered_args = provided_args
            else:
                allowed_params = {
                    param_name
                    for param_name, parameter in signature.parameters.items()
                    if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                }
                filtered_args = {
                    key: value
                    for key, value in provided_args.items()
                    if key in allowed_params
                }
            return tool.handler(**filtered_args)
        except TypeError as exc:
            logger.error("Tool argument mismatch for %s: %s", name, exc)
            return {"status": "error", "message": str(exc)}
        except Exception as exc:
            logger.error("Tool %s failed: %s", name, exc)
            return {"status": "error", "message": str(exc)}

    def get_price(
        self,
        product: str,
        index: str,
        coating: str | None = None,
        treatment: str | None = None,
        city: str | None = None,
    ) -> dict[str, Any]:
        desired_treatment = coating or treatment
        for row in self.kb.lens_catalog:
            if row.get("product") != product or row.get("index") != index:
                continue
            if desired_treatment and row.get("coating") != desired_treatment:
                continue
            return {
                "status": "ok",
                "sku": row.get("sku", ""),
                "product": product,
                "index": index,
                "treatment": row.get("coating", ""),
                "city": city,
                "price_min_dt": row.get("price_min_dt", 0),
                "price_max_dt": row.get("price_max_dt", 0),
                "stock_policy": row.get("stock_policy", ""),
            }
        return {"status": "not_found", "product": product, "index": index, "treatment": desired_treatment}

    def get_store_info(self, city: str) -> dict[str, Any]:
        city_lower = city.lower()
        for row in self.kb.stores:
            if city_lower not in str(row.get("city", "")).lower():
                continue
            return {
                "status": "ok",
                "store_name": row.get("store_name", ""),
                "city": row.get("city", ""),
                "hours_weekday": row.get("hours_weekday", ""),
                "hours_sat": row.get("hours_sat", ""),
                "hours_sun": row.get("hours_sun", "ferme"),
            }
        return {"status": "not_found", "city": city}

    def track_order(
        self,
        num_client: str,
        order_id: str,
        priority: str | None = None,
        agence: str | None = None,
        city: str | None = None,
        secteur: str | None = None,
        requested_slot: str | None = None,
        delivery_slot: str | None = None,
    ) -> dict[str, Any]:
        slots = canonicalize_slots({"num_client": num_client, "order_id": order_id, "priority": priority})
        order_id = str(slots.get("order_id", ""))
        num_client = str(slots.get("num_client", ""))
        priority = slots.get("priority")

        def attach_delivery_context(result: dict[str, Any], *, city_hint: str | None = None) -> dict[str, Any]:
            schedule = self.get_delivery_schedule(
                agence=agence,
                city=city or city_hint,
                secteur=secteur,
                requested_slot=requested_slot or delivery_slot,
                delivery_slot=delivery_slot,
            )
            if schedule.get("status") == "ok":
                result.update(_delivery_context_from_schedule(schedule))
            return result

        for row in self.kb.command_orders:
            if row.get("order_id") != order_id:
                continue
            if str(row.get("num_client", "")) and str(row.get("num_client")) != num_client:
                return {
                    "status": "verification_failed",
                    "order_id": order_id,
                    "num_client": num_client,
                    "matched_num_client": row.get("num_client"),
                    "source": row.get("source", "commandes_intents"),
                }
            response = {
                "status": "ok",
                "order_id": order_id,
                "num_client": num_client,
                "priority": priority or row.get("priority"),
                "order_status": row.get("order_status", "unknown"),
                "order_type": row.get("order_type"),
                "customer_name": row.get("customer_name"),
                "date": row.get("date"),
                "date_from": row.get("date_from"),
                "date_to": row.get("date_to"),
                "source": row.get("source", "commandes_intents"),
            }
            return attach_delivery_context(response)

        for row in self.kb.orders:
            if row.get("order_id") != order_id:
                continue
            response = {
                "status": "ok",
                "order_id": order_id,
                "num_client": num_client,
                "priority": priority,
                "order_status": row.get("status", "unknown"),
                "city": row.get("city", ""),
                "eta_days": row.get("eta_days", ""),
                "created_at": row.get("created_at", ""),
                "verification": "order_id_matched_only",
                "source": "orders_mock",
            }
            return attach_delivery_context(response, city_hint=str(row.get("city", "")))

        return {"status": "not_found", "order_id": order_id, "num_client": num_client, "priority": priority}

    def create_order(self, **kwargs: Any) -> dict[str, Any]:
        slots = canonicalize_slots(kwargs)
        missing_fields = [field for field in ["num_client", "product"] if not slots.get(field)]
        product_text = str(slots.get("product", "")).lower()

        recommended_missing: list[str] = []
        if not slots.get("material") and not slots.get("index"):
            recommended_missing.append("material_or_index")
        if not slots.get("diameter"):
            recommended_missing.append("diameter")
        if any(keyword in product_text for keyword in ["progressive", "bifocal", "top 28"]) and not slots.get("addition"):
            recommended_missing.append("addition")
        if any(slots.get(field) for field in ["od_sphere", "od_cyl", "od_axis"]) and not any(
            slots.get(field) for field in ["og_sphere", "og_cyl", "og_axis"]
        ):
            recommended_missing.append("og_values_or_confirmation")
        if any(slots.get(field) for field in ["og_sphere", "og_cyl", "og_axis"]) and not any(
            slots.get(field) for field in ["od_sphere", "od_cyl", "od_axis"]
        ):
            recommended_missing.append("od_values_or_confirmation")

        recap_items = _material_treatment_tokens(slots)
        recap = ", ".join(recap_items) if recap_items else "core request captured"

        result = {
            "status": "collecting" if missing_fields else "draft",
            "draft_id": _draft_id(),
            "submitted": False,
            "needs_confirmation": True,
            "missing_fields": missing_fields,
            "recommended_missing": recommended_missing,
            "recap": recap,
        }
        result.update(_compact_dict(slots))
        return result

    def check_availability(
        self,
        product: str | None = None,
        material: str | None = None,
        treatment: str | None = None,
        reference: str | None = None,
        diameter: str | None = None,
        index: str | None = None,
        city: str | None = None,
    ) -> dict[str, Any]:
        slots = canonicalize_slots(
            {
                "product": product,
                "material": material,
                "treatment": treatment,
                "reference": reference,
                "diameter": diameter,
                "index": index,
                "city": city,
            }
        )

        reference_info = self.confirm_reference(str(slots["reference"])) if slots.get("reference") else None
        catalog_match = None
        if slots.get("product") and slots.get("index"):
            for row in self.kb.lens_catalog:
                if row.get("product") == slots.get("product") and row.get("index") == slots.get("index"):
                    catalog_match = row
                    break

        rag_lookup = self.lookup_lens_catalog(
            code=str(slots.get("reference", "")) if slots.get("reference") else None,
            name=str(slots.get("product", "")) if slots.get("product") else None,
            material=str(slots.get("material", "")) if slots.get("material") else None,
            diameter=str(slots.get("diameter", "")) if slots.get("diameter") else None,
            photochromic=str(slots.get("treatment", "")) if slots.get("treatment") else None,
        )

        delivery_schedule = (
            self.get_delivery_schedule(city=str(slots.get("city")))
            if slots.get("city")
            else {"status": "not_found"}
        )

        result = {
            "status": "ok",
            "availability": "to_confirm",
            "reference_found": bool(reference_info and reference_info.get("status") == "ok"),
            "catalog_match": bool(catalog_match) or rag_lookup.get("status") == "ok",
            "official_rag_match": rag_lookup.get("status") == "ok",
            "structured_kb_match": bool(catalog_match),
            "fulfillment_hint": catalog_match.get("stock_policy") if catalog_match else "requires_stock_confirmation",
            "message": "Availability needs stock/backoffice confirmation before final commitment.",
        }
        result.update(_compact_dict(slots))
        if reference_info and reference_info.get("status") == "ok":
            result["reference_details"] = {
                key: value
                for key, value in reference_info.items()
                if key not in {"status", "message", "examples"}
            }
        if catalog_match:
            result["catalog_city"] = catalog_match.get("city")
            result["catalog_treatment"] = catalog_match.get("coating")
        if rag_lookup.get("status") == "ok":
            result["lens_matches"] = rag_lookup.get("matches", [])[:3]
        if delivery_schedule.get("status") == "ok":
            result["delivery_schedule"] = {
                "agence": delivery_schedule.get("agence"),
                "secteur": delivery_schedule.get("secteur"),
                "next_slot": delivery_schedule.get("next_slot"),
                "tous_creneaux": delivery_schedule.get("tous_creneaux", []),
            }
        return result

    def confirm_reference(self, reference: str) -> dict[str, Any]:
        reference = str(reference).replace(" ", "").strip()
        rag_lookup = self.lookup_lens_catalog(code=reference)
        if rag_lookup.get("status") == "ok":
            best = (rag_lookup.get("matches") or [{}])[0]
            return {
                "status": "ok",
                "reference": reference,
                "seen_count": len(rag_lookup.get("matches") or []),
                "products": [best.get("name")] if best.get("name") else [],
                "materials": [best.get("material")] if best.get("material") else [],
                "treatments": [],
                "colors": [],
                "diameters": [best.get("diameter")] if best.get("diameter") else [],
                "examples": [],
                "source": "official_rag_catalogue_verres",
                "best_match": best,
            }

        entry = self.kb.reference_catalog.get(reference)
        if entry is not None:
            return {
                "status": "ok",
                "reference": reference,
                "seen_count": entry.get("seen_count", 0),
                "products": sorted(entry.get("products", set())),
                "materials": sorted(entry.get("materials", set())),
                "treatments": sorted(entry.get("treatments", set())),
                "colors": sorted(entry.get("colors", set())),
                "diameters": sorted(entry.get("diameters", set())),
                "examples": entry.get("examples", []),
                "source": "commandes_intents",
            }

        return {"status": "not_found", "reference": reference}

    def get_delivery_schedule(
        self,
        agence: str | None = None,
        city: str | None = None,
        secteur: str | None = None,
        requested_slot: str | None = None,
        delivery_slot: str | None = None,
    ) -> dict[str, Any]:
        if not self.kb.delivery_index:
            return {
                "status": "not_found",
                "reason": "delivery_index_empty",
            }

        agence_lookup = _norm_lookup(agence or city)
        secteur_lookup = _norm_lookup(secteur)
        requested_slot_clean = str(requested_slot or delivery_slot or "").strip()

        scored: list[tuple[int, dict[str, Any]]] = []
        for entry in self.kb.delivery_index:
            score = 0
            agence_value = _norm_lookup(entry.get("agence"))
            secteur_value = _norm_lookup(entry.get("secteur"))

            if agence_lookup:
                if agence_lookup == agence_value:
                    score += 4
                elif agence_lookup in agence_value or agence_value in agence_lookup:
                    score += 2

            if secteur_lookup:
                if secteur_lookup == secteur_value:
                    score += 4
                elif secteur_lookup in secteur_value or secteur_value in secteur_lookup:
                    score += 2

            if not agence_lookup and not secteur_lookup:
                score = 1

            if entry.get("secteur"):
                score += 1
            if entry.get("tous_creneaux"):
                score += 1

            if score > 0:
                scored.append((score, entry))

        if not scored:
            available_agences = sorted({str(item.get("agence", "")).strip() for item in self.kb.delivery_index if item.get("agence")})
            return {
                "status": "not_found",
                "agence": agence or city,
                "secteur": secteur,
                "available_agences": available_agences[:20],
            }

        scored.sort(key=lambda item: item[0], reverse=True)
        best = scored[0][1]
        all_slots = [str(item) for item in best.get("tous_creneaux", []) if str(item).strip()]
        next_slot = best.get("premier_creneau") or (all_slots[0] if all_slots else None)
        if requested_slot_clean and all_slots:
            next_slot = next_time_slot_after(all_slots, requested_slot_clean, inclusive=True) or next_slot

        return {
            "status": "ok",
            "agence": best.get("agence"),
            "secteur": best.get("secteur"),
            "nb_livraisons_jour": best.get("nb_livraisons_jour"),
            "premier_creneau": best.get("premier_creneau"),
            "tous_creneaux": all_slots,
            "requested_slot": requested_slot_clean or None,
            "next_slot": next_slot,
            "delivery_target": "optician_agency",
            "delivery_eta_policy": "approximate_agency_window",
            "source": best.get("source"),
        }

    def lookup_lens_catalog(
        self,
        code: str | None = None,
        name: str | None = None,
        material: str | None = None,
        diameter: str | None = None,
        photochromic: str | None = None,
    ) -> dict[str, Any]:
        if not self.kb.lens_rag_index:
            return {
                "status": "not_found",
                "reason": "lens_index_empty",
            }

        code_lookup = _norm_lookup(code)
        name_lookup = _norm_lookup(name)
        material_lookup = _norm_lookup(material)
        diameter_lookup = _norm_lookup(diameter)
        photo_lookup = _norm_lookup(photochromic)

        if not any([code_lookup, name_lookup, material_lookup, diameter_lookup, photo_lookup]):
            return {
                "status": "not_found",
                "reason": "no_lookup_filters",
            }

        scored: list[tuple[int, dict[str, Any]]] = []
        for entry in self.kb.lens_rag_index:
            score = 0
            entry_code = _norm_lookup(entry.get("code"))
            entry_name = _norm_lookup(entry.get("name"))
            entry_material = _norm_lookup(entry.get("material"))
            entry_diameter = _norm_lookup(entry.get("diameter"))
            entry_photo = _norm_lookup(entry.get("photochromic"))

            if code_lookup:
                if code_lookup == entry_code:
                    score += 8
                elif code_lookup in entry_code:
                    score += 4

            if name_lookup:
                if name_lookup in entry_name:
                    score += 4
                elif any(token in entry_name for token in name_lookup.split() if len(token) > 2):
                    score += 2

            if material_lookup and material_lookup in entry_material:
                score += 3
            if diameter_lookup and diameter_lookup in entry_diameter:
                score += 2
            if photo_lookup and photo_lookup in entry_photo:
                score += 2

            if score > 0:
                scored.append((score, entry))

        if not scored:
            return {
                "status": "not_found",
                "code": code,
                "name": name,
                "material": material,
                "diameter": diameter,
            }

        scored.sort(key=lambda item: item[0], reverse=True)
        matches = []
        for _, entry in scored[:5]:
            matches.append(
                {
                    "code": entry.get("code"),
                    "name": entry.get("name"),
                    "brand": entry.get("brand"),
                    "geometry": entry.get("geometry"),
                    "material": entry.get("material"),
                    "photochromic": entry.get("photochromic"),
                    "diameter": entry.get("diameter"),
                    "text": entry.get("text"),
                    "source": entry.get("source"),
                }
            )

        return {
            "status": "ok",
            "matches": matches,
        }

    def book_appointment(self, city: str, date: str, time_slot: str, phone: str) -> dict[str, Any]:
        appointment_id = "APT-" + "".join(random.choices(string.digits, k=6))
        normalized_phone = _normalize_phone(phone)
        return {
            "status": "draft",
            "appointment_id": appointment_id,
            "city": city,
            "date": date,
            "time_slot": time_slot,
            "phone": normalized_phone,
            "needs_confirmation": True,
            "submitted": False,
        }


_REGISTRY: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ToolRegistry()
    return _REGISTRY
