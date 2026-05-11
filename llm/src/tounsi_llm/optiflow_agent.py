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
    "suivi_commande": "track_order",
    "list_my_orders": "get_my_orders",
    "get_my_orders": "get_my_orders",
    "list_orders": "get_my_orders",
    "search_orders": "search_orders",
    "find_order": "search_orders",
    "order_detail": "get_order_detail",
    "get_order_detail": "get_order_detail",
    "client_orders": "get_orders_by_client",
    "orders_by_client": "get_orders_by_client",
    "verify_client": "verify_client",
    "client_lookup": "verify_client",
    "opticien_profile": "get_opticien_profile",
    "my_profile": "get_opticien_profile",
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
