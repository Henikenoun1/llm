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
