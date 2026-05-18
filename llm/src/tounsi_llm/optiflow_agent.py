"""OptiFlow agent loop: bridges LLM intent/slots to canonical HTTP tools.

The current production model is fine-tuned for intent + slot extraction (not
free-form JSON tool-calling). This module therefore routes the LLM's intent
into an OptiFlow tool from `optiflow_manifest`, executes it via the HTTP
executor, and returns a typed observation.

Phase 2 (true tool-calling): the `decide_tool` step can be swapped to call a
generic instruction-tuned LLM that emits `{"tool_name": ..., "arguments": {...}}`
without changing the rest of the pipeline.
"""
from __future__ import annotations

from typing import Any

from .config import logger
from .optiflow_backend import execute_optiflow_tool
from .optiflow_manifest import OPTIFLOW_TOOLS, get_optiflow_tool, list_optiflow_tools


# Mapping from LLM intent (as produced by the trained classifier) to an
# OptiFlow canonical tool name. Intents not listed fall back to local KB.
INTENT_TO_OPTIFLOW_TOOL: dict[str, str] = {
    "track_order": "track_order",
    "order_status": "track_order",
    "order_tracking": "track_order",
    "suivi_commande": "track_order",
    "reference_confirmation": "track_order",
    "list_my_orders": "get_my_orders",
    "get_my_orders": "get_my_orders",
    "list_orders": "get_my_orders",
    "mes_commandes": "get_my_orders",
    "search_orders": "search_orders",
    "find_order": "search_orders",
    "recherche_commande": "search_orders",
    "order_detail": "get_order_detail",
    "get_order_detail": "get_order_detail",
    "detail_commande": "get_order_detail",
    "client_orders": "get_orders_by_client",
    "orders_by_client": "get_orders_by_client",
    "commandes_client": "get_orders_by_client",
    "verify_client": "verify_client",
    "client_lookup": "verify_client",
    "get_num_client": "verify_client",
    "opticien_profile": "get_opticien_profile",
    "my_profile": "get_opticien_profile",
    "profil_opticien": "get_opticien_profile",
}


# Mapping from canonical LLM slot name to canonical OptiFlow tool argument.
# We accept several slot aliases produced by the existing extractor.
_SLOT_ALIASES: dict[str, list[str]] = {
    "code_client": ["code_client", "codeClient", "num_client", "numClient", "client_code"],
    "reference": ["reference", "order_id", "num_cmd", "numCmd", "order_reference"],
    "id": ["id", "order_internal_id", "commande_id"],
    "page": ["page"],
    "page_size": ["page_size", "pageSize"],
    "statut": ["statut", "status"],
    "num_cmd": ["num_cmd", "numCmd", "order_id"],
    "num_client": ["num_client", "numClient", "code_client"],
    "numero_bon": ["numero_bon", "numeroBon"],
    "date_from": ["date_from", "dateFrom"],
    "date_to": ["date_to", "dateTo"],
    "type_commande": ["type_commande", "typeCommande"],
    "nom_porteur": ["nom_porteur", "nomPorteur"],
}


def _pick(slots: dict[str, Any], canonical: str) -> Any:
    for alias in _SLOT_ALIASES.get(canonical, [canonical]):
        value = slots.get(alias)
        if value not in (None, ""):
            return value
    return None


def _build_args(tool_name: str, slots: dict[str, Any], user_context: dict[str, Any]) -> dict[str, Any]:
    """Project the LLM slots onto the tool's declared parameter set."""
    tool = get_optiflow_tool(tool_name)
    if tool is None:
        return {}
    properties = tool.get("parameters", {}).get("properties", {}) or {}
    args: dict[str, Any] = {}
    for prop_name in properties.keys():
        value = _pick(slots, prop_name)
        if value is None and prop_name == "code_client":
            value = user_context.get("USER_CODE_CLIENT") or user_context.get("code_client")
        if value not in (None, ""):
            args[prop_name] = value
    return args


def _missing_required(tool_name: str, args: dict[str, Any]) -> list[str]:
    tool = get_optiflow_tool(tool_name)
    if tool is None:
        return []
    required = tool.get("parameters", {}).get("required", []) or []
    return [name for name in required if args.get(name) in (None, "")]


def _order_payload(result: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    data = result.get("data")
    if not isinstance(data, dict):
        return {}
    if isinstance(data.get("order"), dict):
        return data["order"]
    return data


def _merge_order_payload(base: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
    if not detail:
        return base
    merged = dict(base or {})
    for key, value in detail.items():
        if value not in (None, "", [], {}):
            merged[key] = value
    return merged


def _detail_id_candidates(order: dict[str, Any], args: dict[str, Any]) -> list[str]:
    candidates = [
        order.get("id"),
        order.get("commandeId"),
        order.get("_id"),
        order.get("uuid"),
        order.get("numCmd"),
        order.get("reference"),
        args.get("reference"),
    ]
    out: list[str] = []
    for value in candidates:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _first_search_item(result: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    items = data.get("items") if isinstance(data.get("items"), list) else []
    if not items:
        items = data.get("data") if isinstance(data.get("data"), list) else []
    if items and isinstance(items[0], dict):
        return items[0]
    return {}


def _enrich_tracking_with_detail(
    observation: dict[str, Any],
    *,
    args: dict[str, Any],
    access_token: str,
    backend_url: str,
    user_role: str = "",
) -> dict[str, Any]:
    """Best-effort enrichment of the public track_order response.

    track_order returns only public fields (no internal id, no prescription).
    To produce the WOW card, we look up the same reference via:
      - search_orders (ADMIN/OPERATEUR) → items[0] (rich payload + id)
      - get_my_orders (OPTICIEN)        → items[0] (rich payload + id)
    Then we optionally chain get_order_detail(id) for the canonical detail.
    All calls are best-effort; failures never break the existing track_order card.
    """
    if observation.get("status") != "ok" or not access_token:
        return observation
    order = _order_payload(observation)
    reference = str(order.get("reference") or args.get("reference") or "").strip()
    role = (user_role or "").upper()

    rich_item: dict[str, Any] = {}
    rich_source = ""
    if reference:
        # Try the role-appropriate resolver first, then fall back to the other.
        candidates_tools: list[str] = []
        if role in {"ADMIN", "OPERATEUR", "AGENT"}:
            candidates_tools = ["search_orders", "get_my_orders"]
        else:
            candidates_tools = ["get_my_orders", "search_orders"]
        for tool_name in candidates_tools:
            envelope = execute_optiflow_tool(
                name=tool_name,
                args={"num_cmd": reference},
                access_token=access_token,
                backend_url=backend_url,
                timeout_s=4,
            )
            if envelope.get("status") == "ok":
                item = _first_search_item(envelope)
                if item:
                    rich_item = item
                    rich_source = tool_name
                    break

    candidates: list[str] = []
    if rich_item.get("id"):
        candidates.append(str(rich_item["id"]))
    candidates.extend(_detail_id_candidates(order, args))

    detail_payload: dict[str, Any] = {}
    detail_envelope: dict[str, Any] = {}
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        envelope = execute_optiflow_tool(
            name="get_order_detail",
            args={"id": candidate},
            access_token=access_token,
            backend_url=backend_url,
            timeout_s=4,
        )
        if envelope.get("status") != "ok":
            continue
        detail_payload = _order_payload(envelope)
        if detail_payload:
            detail_envelope = envelope
            break

    if not rich_item and not detail_payload:
        return observation

    enriched = dict(observation)
    merged = dict(order)
    if rich_item:
        merged = _merge_order_payload(merged, rich_item)
        enriched["rich_result_source"] = rich_source
    if detail_payload:
        merged = _merge_order_payload(merged, detail_payload)
        enriched["detail_result"] = detail_envelope
        enriched["detail_source"] = "get_order_detail"
    enriched["data"] = merged
    return enriched


def decide_tool(
    *,
    intent: str,
    slots: dict[str, Any],
    user_context: dict[str, Any],
) -> dict[str, Any] | None:
    """Resolve (intent, slots) into an OptiFlow tool call descriptor.

    Returns None when the intent does not map to a backend tool.
    The descriptor is `{tool_name, arguments, missing}`.
    """
    tool_name = INTENT_TO_OPTIFLOW_TOOL.get(intent)
    if not tool_name:
        return None

    role = (user_context.get("USER_ROLE") or "").strip().upper()
    available = {tool["name"] for tool in list_optiflow_tools(role=role)}
    if tool_name not in available:
        logger.info(
            "Tool %s not allowed for role=%s, intent=%s -> skipping backend call",
            tool_name, role, intent,
        )
        return None

    arguments = _build_args(tool_name, slots, user_context)
    missing = _missing_required(tool_name, arguments)
    return {"tool_name": tool_name, "arguments": arguments, "missing": missing}


def run_optiflow_agent_step(
    *,
    intent: str,
    slots: dict[str, Any],
    user_context: dict[str, Any],
) -> dict[str, Any] | None:
    """Single agent step: decide → execute → return observation envelope.

    Returns None when no OptiFlow tool applies (caller should fall back to KB).
    Returns `{tool_call, tool_result, missing}` otherwise. `tool_result` may be
    None when required slots are missing (the agent should ask the user).
    """
    backend_url = (user_context.get("BACKEND_URL") or "").strip()
    access_token = (user_context.get("ACCESS_TOKEN") or "").strip()

    decision = decide_tool(intent=intent, slots=slots, user_context=user_context)
    if not decision:
        return None

    tool_name = decision["tool_name"]
    tool_meta = get_optiflow_tool(tool_name) or {}
    requires_token = (tool_meta.get("http", {}).get("auth", "bearer") == "bearer")

    # If we cannot reach the backend, do not pretend; let the caller fall back.
    if not backend_url:
        logger.info("No backend_url in user_context; skipping OptiFlow tool %s", tool_name)
        return None
    if requires_token and not access_token:
        return {
            "tool_call": {"name": tool_name, "arguments": decision["arguments"]},
            "tool_result": {
                "status": "unauthorized",
                "tool": tool_name,
                "error": "Authentification requise pour cet outil.",
            },
            "missing": decision["missing"],
        }

    if decision["missing"]:
        return {
            "tool_call": {"name": tool_name, "arguments": decision["arguments"]},
            "tool_result": None,
            "missing": decision["missing"],
        }

    observation = execute_optiflow_tool(
        name=tool_name,
        args=decision["arguments"],
        access_token=access_token or None,
        backend_url=backend_url,
    )
    if tool_name == "track_order" and access_token:
        observation = _enrich_tracking_with_detail(
            observation,
            args=decision["arguments"],
            access_token=access_token,
            backend_url=backend_url,
            user_role=str(user_context.get("USER_ROLE") or ""),
        )
    return {
        "tool_call": {"name": tool_name, "arguments": decision["arguments"]},
        "tool_result": observation,
        "missing": [],
    }


__all__ = [
    "INTENT_TO_OPTIFLOW_TOOL",
    "OPTIFLOW_TOOLS",
    "decide_tool",
    "run_optiflow_agent_step",
]
