"""
Business tools and tool registry.

This module intentionally keeps tool execution separate from inference and RAG.
Changing the tool list later should mostly happen here.
"""
from __future__ import annotations

import csv
import json
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import DOMAIN_CFG, KB_DIR, logger


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


class KnowledgeBase:
    def __init__(self, root: Path | None = None) -> None:
        kb_root = root or KB_DIR
        self.lens_catalog = _read_csv(kb_root / "lens_catalog.csv")
        self.stores = _read_csv(kb_root / "stores.csv")
        self.policies = _read_jsonl(kb_root / "policies.jsonl")
        self.orders = _read_csv(kb_root / "orders_mock.csv")

        for row in self.lens_catalog:
            row["price_min_dt"] = float(row.get("price_min_dt", 0) or 0)
            row["price_max_dt"] = float(row.get("price_max_dt", 0) or 0)

        logger.info(
            "Tool KB loaded: %d products, %d stores, %d policies, %d orders",
            len(self.lens_catalog),
            len(self.stores),
            len(self.policies),
            len(self.orders),
        )


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
                        "order_id": {"type": "string"},
                        "phone": {"type": "string"},
                    },
                    "required": ["order_id"],
                },
                handler=self.track_order,
            ),
            "create_order": ToolDefinition(
                name="create_order",
                description=descriptions.get("create_order", "Create a new order."),
                parameters={
                    "type": "object",
                    "properties": {
                        "product": {"type": "string"},
                        "index": {"type": "string"},
                        "city": {"type": "string"},
                        "phone": {"type": "string"},
                        "coating": {"type": "string"},
                    },
                    "required": ["product", "index", "city", "phone"],
                },
                handler=self.create_order,
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
            return tool.handler(**(args or {}))
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
        city: str | None = None,
    ) -> dict[str, Any]:
        for row in self.kb.lens_catalog:
            if row.get("product") != product or row.get("index") != index:
                continue
            if coating and row.get("coating") != coating:
                continue
            return {
                "status": "ok",
                "sku": row.get("sku", ""),
                "product": product,
                "index": index,
                "coating": row.get("coating", ""),
                "city": city,
                "price_min_dt": row.get("price_min_dt", 0),
                "price_max_dt": row.get("price_max_dt", 0),
            }
        return {"status": "not_found", "product": product, "index": index}

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

    def track_order(self, order_id: str, phone: str | None = None) -> dict[str, Any]:
        for row in self.kb.orders:
            if row.get("order_id") != order_id:
                continue
            if phone:
                expected = str(row.get("phone", "")).replace(" ", "")
                actual = phone.replace(" ", "")
                if expected and expected != actual:
                    return {"status": "verification_failed", "order_id": order_id}
            return {"status": "ok", **row}
        return {"status": "not_found", "order_id": order_id}

    def create_order(
        self,
        product: str,
        index: str,
        city: str,
        phone: str,
        coating: str | None = None,
    ) -> dict[str, Any]:
        order_id = "ORD-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
        return {
            "status": "ok",
            "order_id": order_id,
            "product": product,
            "index": index,
            "coating": coating or "none",
            "city": city,
            "phone": phone,
            "delivery_days": 3,
        }

    def book_appointment(self, city: str, date: str, time_slot: str, phone: str) -> dict[str, Any]:
        appointment_id = "APT-" + "".join(random.choices(string.digits, k=6))
        return {
            "status": "ok",
            "appointment_id": appointment_id,
            "city": city,
            "date": date,
            "time_slot": time_slot,
            "phone": phone,
        }


_REGISTRY: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ToolRegistry()
    return _REGISTRY

