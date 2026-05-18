"""OptiFlow backend integration helpers for LLM tools.

This module is intentionally separate from the core inference pipeline. It lets a
VM-hosted LLM call the real OptiFlow backend for specific tools while preserving
local KB/mock behavior when no backend URL is configured.
"""
from __future__ import annotations

import os
from typing import Any
from urllib.parse import urljoin

import requests

from .config import logger


def _normalize_base_url(value: str | None) -> str:
    text = str(value or "").strip()
    return text.rstrip("/") if text else ""


def _resolve_track_order_url(backend_url: str | None = None) -> str:
    explicit_url = _normalize_base_url(os.getenv("OPTIFLOW_TRACK_ORDER_URL", ""))
    if explicit_url:
        return explicit_url

    base_url = _normalize_base_url(backend_url or os.getenv("OPTIFLOW_BACKEND_BASE_URL", ""))
    if not base_url:
        return ""

    if base_url.endswith("/api"):
        return urljoin(f"{base_url}/", "chatbot/tools/track-order")
    return urljoin(f"{base_url}/", "api/chatbot/tools/track-order")


def _first_non_empty(*values: Any) -> str | None:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return None


def _normalize_backend_track_order(payload: dict[str, Any]) -> dict[str, Any]:
    backend_status = str(payload.get("status") or "").strip().lower()
    query = payload.get("query") if isinstance(payload.get("query"), dict) else {}
    order = payload.get("order") if isinstance(payload.get("order"), dict) else None

    if backend_status == "found" and order:
        opticien = order.get("opticien") if isinstance(order.get("opticien"), dict) else {}
        return {
            "status": "ok",
            "order_id": _first_non_empty(order.get("numCmd"), query.get("orderId")),
            "num_client": _first_non_empty(opticien.get("codeClient"), query.get("numClient")),
            "order_status": _first_non_empty(order.get("statutLabel"), order.get("statut"), "unknown"),
            "order_type": order.get("typeCommande"),
            "customer_name": order.get("nomPorteur"),
            "date": order.get("dateCommande"),
            "next_action": order.get("nextAction"),
            "source": "optiflow_backend",
            "backend_status": backend_status,
            "backend_trace_id": payload.get("traceId"),
            "backend_result": payload,
        }

    if backend_status == "not_found":
        return {
            "status": "not_found",
            "order_id": _first_non_empty(query.get("orderId")),
            "num_client": _first_non_empty(query.get("numClient")),
            "source": "optiflow_backend",
            "backend_status": backend_status,
            "backend_trace_id": payload.get("traceId"),
            "backend_result": payload,
        }

    if backend_status == "ambiguous":
        return {
            "status": "not_found",
            "order_id": _first_non_empty(query.get("orderId")),
            "num_client": _first_non_empty(query.get("numClient")),
            "missing_fields": ["order_id"],
            "source": "optiflow_backend",
            "backend_status": backend_status,
            "backend_trace_id": payload.get("traceId"),
            "backend_result": payload,
        }

    return {
        "status": "error",
        "order_id": _first_non_empty(query.get("orderId")),
        "num_client": _first_non_empty(query.get("numClient")),
        "source": "optiflow_backend",
        "backend_status": backend_status or "backend_error",
        "backend_trace_id": payload.get("traceId"),
        "backend_result": payload,
    }


def track_order_from_optiflow_backend(
    *,
    num_client: str,
    order_id: str,
    priority: str | None = None,
    agence: str | None = None,
    city: str | None = None,
    secteur: str | None = None,
    requested_slot: str | None = None,
    delivery_slot: str | None = None,
    access_token: str | None = None,
    backend_url: str | None = None,
) -> dict[str, Any] | None:
    """Call OptiFlow backend tool endpoint if configured.

    Returns None when disabled so the caller can keep the existing KB fallback.
    """

    url = _resolve_track_order_url(backend_url=backend_url)
    if not url:
        return None

    service_key = os.getenv("OPTIFLOW_TRACK_ORDER_KEY", "").strip()
    timeout_s = float(os.getenv("OPTIFLOW_TRACK_ORDER_TIMEOUT_SECONDS", "8"))

    body = {
        "numClient": num_client,
        "orderId": order_id,
        "priority": priority,
        "agence": agence,
        "city": city,
        "secteur": secteur,
        "requestedSlot": requested_slot,
        "deliverySlot": delivery_slot,
    }
    body = {key: value for key, value in body.items() if value not in (None, "")}

    headers = {"Content-Type": "application/json"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    if service_key:
        headers["X-OptiFlow-Tool-Key"] = service_key

    try:
        response = requests.post(url, json=body, headers=headers, timeout=timeout_s)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return {
                "status": "error",
                "order_id": order_id,
                "num_client": num_client,
                "source": "optiflow_backend",
                "message": "Invalid backend JSON payload",
            }
        return _normalize_backend_track_order(payload)
    except requests.Timeout:
        logger.warning("OptiFlow track_order timeout url=%s order_id=%s", url, order_id)
        return {
            "status": "error",
            "order_id": order_id,
            "num_client": num_client,
            "source": "optiflow_backend",
            "message": "Erreur technique. Réessayez dans un instant.",
        }
    except requests.RequestException as exc:
        logger.warning("OptiFlow track_order request failed url=%s order_id=%s error=%s", url, order_id, exc)
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        error_messages = {
            401: "Votre session a expiré. Reconnectez-vous.",
            403: "Vous n'avez pas les droits pour cette action.",
            404: "Commande introuvable ou accès non autorisé.",
            500: "Erreur technique. Réessayez dans un instant.",
        }
        return {
            "status": "error",
            "order_id": order_id,
            "num_client": num_client,
            "source": "optiflow_backend",
            "http_status": status_code,
            "message": error_messages.get(status_code, str(exc)),
        }


# ---------------------------------------------------------------------------
# Generic HTTP executor for the OptiFlow tool manifest (phase 1: read-only)
# ---------------------------------------------------------------------------

_GENERIC_HTTP_ERRORS = {
    400: "Requête invalide.",
    401: "Votre session a expiré. Reconnectez-vous.",
    403: "Vous n'avez pas les droits pour cette action.",
    404: "Ressource introuvable.",
    422: "Paramètres invalides ou incomplets.",
    500: "Erreur technique côté backend. Réessayez dans un instant.",
    502: "Backend indisponible. Réessayez dans un instant.",
    503: "Service temporairement indisponible.",
}


def _render_path(path: str, args: dict[str, Any], param_map: dict[str, str]) -> tuple[str, set[str]]:
    """Substitute {placeholders} in path; return (rendered_path, used_keys).

    Placeholder names follow the LLM (snake_case) arg names. The optional
    `param_map` is only used for query/body remapping, not for path tokens.
    """
    used: set[str] = set()
    rendered = path
    for key, value in args.items():
        token = "{" + key + "}"
        if token in rendered and value not in (None, ""):
            rendered = rendered.replace(token, str(value))
            used.add(key)
    return rendered, used


def _select(
    args: dict[str, Any],
    keys: list[str],
    skip: set[str],
    param_map: dict[str, str],
) -> dict[str, Any]:
    """Pick the requested keys from args and rename them via param_map."""
    out: dict[str, Any] = {}
    for key in keys or []:
        if key in skip:
            continue
        value = args.get(key)
        if value in (None, ""):
            continue
        backend_key = param_map.get(key, key)
        out[backend_key] = value
    return out


def execute_optiflow_tool(
    *,
    name: str,
    args: dict[str, Any] | None,
    access_token: str | None,
    backend_url: str | None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Execute one canonical OptiFlow tool against the Nest backend.

    Returns a normalized envelope: {status, tool, data?, error?, http_status?}.
    `status` is one of: ok | error | unauthorized | forbidden | not_found | invalid.
    """
    # Local import to avoid a circular dependency with the manifest module.
    from .optiflow_manifest import get_optiflow_tool

    tool = get_optiflow_tool(name)
    if tool is None:
        return {"status": "invalid", "tool": name, "error": f"Unknown tool: {name}"}

    base = _normalize_base_url(backend_url or os.getenv("OPTIFLOW_BACKEND_BASE_URL", ""))
    if not base:
        return {
            "status": "error",
            "tool": name,
            "error": "backend_url not configured",
        }

    http = tool.get("http") or {}
    method = str(http.get("method", "GET")).upper()
    path_template = str(http.get("path", "/"))
    auth_mode = str(http.get("auth", "bearer")).lower()
    param_map = http.get("param_map") or {}
    args = dict(args or {})

    rendered_path, used_in_path = _render_path(path_template, args, param_map)
    # If the template still contains an unresolved placeholder, abort early.
    if "{" in rendered_path:
        return {
            "status": "invalid",
            "tool": name,
            "error": f"Missing path arguments for {name}: {rendered_path}",
        }

    url = base + rendered_path if rendered_path.startswith("/") else base + "/" + rendered_path

    query_params = (
        _select(args, http.get("query") or [], used_in_path, param_map)
        if method == "GET" else {}
    )
    body_payload = (
        _select(args, http.get("body") or [], used_in_path, param_map)
        if method != "GET" else None
    )

    headers: dict[str, str] = {"Accept": "application/json"}
    if auth_mode == "bearer":
        token = (access_token or "").strip()
        if not token:
            return {
                "status": "unauthorized",
                "tool": name,
                "error": "Access token requis pour cet outil.",
            }
        headers["Authorization"] = token if token.lower().startswith("bearer ") else f"Bearer {token}"
    if body_payload is not None:
        headers["Content-Type"] = "application/json"

    timeout = float(timeout_s if timeout_s is not None else os.getenv("OPTIFLOW_TOOL_TIMEOUT_SECONDS", "10"))

    try:
        response = requests.request(
            method,
            url,
            params=query_params or None,
            json=body_payload if body_payload else None,
            headers=headers,
            timeout=timeout,
        )
    except requests.Timeout:
        logger.warning("OptiFlow tool=%s timeout url=%s", name, url)
        return {"status": "error", "tool": name, "error": "Backend timeout."}
    except requests.RequestException as exc:
        logger.warning("OptiFlow tool=%s request failed url=%s error=%s", name, url, exc)
        return {"status": "error", "tool": name, "error": str(exc)}

    status_code = response.status_code
    try:
        data = response.json()
    except ValueError:
        data = {"raw": response.text[:500]}

    if 200 <= status_code < 300:
        return {"status": "ok", "tool": name, "http_status": status_code, "data": data}

    status_label = {
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        422: "invalid",
    }.get(status_code, "error")

    return {
        "status": status_label,
        "tool": name,
        "http_status": status_code,
        "error": _GENERIC_HTTP_ERRORS.get(status_code, f"Erreur backend (HTTP {status_code})."),
        "data": data,
    }
