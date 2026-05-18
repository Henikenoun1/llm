"""Canonical tool manifest exposed to the LLM for the OptiFlow integration.

Phase 1 covers read-only "suivi" (order tracking, search, profile).

Each tool entry has:
  - name: canonical tool name used by the LLM (snake_case, stable)
  - description: short, action-oriented description (LLM-facing)
  - parameters: JSON-Schema-like spec ({type, properties, required})
  - http: routing metadata for the executor
      method:    GET | POST
      path:      Nest API path (relative to backend_url, may contain {placeholders})
      query:     list of LLM arg names sent as URL query params (GET)
      body:      list of LLM arg names sent in the JSON body (POST)
      param_map: optional { llm_arg_name: backend_param_name } for snake->camel mapping
      auth:      "bearer" (default) or "public"
      roles:     optional list of allowed USER_ROLE values
"""
from __future__ import annotations

from typing import Any


OPTIFLOW_TOOLS: list[dict[str, Any]] = [
    {
        "name": "track_order",
        "description": (
            "Suivi public d'une commande par code client + référence (numéro de commande). "
            "Aucune authentification requise."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code_client": {"type": "string", "description": "Code client (ex: C00042)."},
                "reference": {"type": "string", "description": "Numéro de commande (numCmd)."},
            },
            "required": ["code_client", "reference"],
        },
        "http": {
            "method": "GET",
            "path": "/api/commandes/track",
            "query": ["code_client", "reference"],
            "param_map": {"code_client": "codeClient"},
            "auth": "public",
        },
    },
    {
        "name": "get_my_orders",
        "description": (
            "Liste paginée des commandes de l'opticien connecté. "
            "À utiliser quand le user demande 'mes commandes' sans filtre précis."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "page": {"type": "integer", "default": 1},
                "page_size": {"type": "integer", "default": 10},
                "statut": {"type": "string", "description": "Filtre statut optionnel."},
                "num_cmd": {"type": "string", "description": "Numéro commande partiel."},
                "date_from": {"type": "string", "description": "ISO YYYY-MM-DD."},
                "date_to": {"type": "string", "description": "ISO YYYY-MM-DD."},
                "type_commande": {"type": "string", "enum": ["PRECAL", "STOCK"]},
                "nom_porteur": {"type": "string"},
            },
            "required": [],
        },
        "http": {
            "method": "GET",
            "path": "/api/commandes/me",
            "query": [
                "page", "page_size", "statut", "num_cmd",
                "date_from", "date_to", "type_commande", "nom_porteur",
            ],
            "param_map": {
                "page_size": "pageSize",
                "num_cmd": "numCmd",
                "date_from": "dateFrom",
                "date_to": "dateTo",
                "type_commande": "typeCommande",
                "nom_porteur": "nomPorteur",
            },
            "auth": "bearer",
            "roles": ["OPTICIEN", "ADMIN", "OPERATEUR"],
        },
    },
    {
        "name": "get_order_detail",
        "description": (
            "Détail complet d'une commande (lignes, statut, montant, dates). "
            "Utiliser quand l'utilisateur connaît l'ID interne de la commande."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "ID interne de la commande."},
            },
            "required": ["id"],
        },
        "http": {
            "method": "GET",
            "path": "/api/commandes/{id}",
            "auth": "bearer",
        },
    },
    {
        "name": "search_orders",
        "description": (
            "Recherche avancée des commandes (numCmd, codeClient, dates, nom porteur). "
            "À privilégier dès qu'il y a un critère de filtrage explicite."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "num_cmd": {"type": "string"},
                "code_client": {"type": "string"},
                "num_client": {"type": "string"},
                "numero_bon": {"type": "string"},
                "date_from": {"type": "string"},
                "date_to": {"type": "string"},
                "nom_porteur": {"type": "string"},
            },
            "required": [],
        },
        "http": {
            "method": "GET",
            "path": "/api/commandes/search/advanced",
            "query": [
                "num_cmd", "code_client", "num_client", "numero_bon",
                "date_from", "date_to", "nom_porteur",
            ],
            "param_map": {
                "num_cmd": "numCmd",
                "code_client": "codeClient",
                "num_client": "numClient",
                "numero_bon": "numeroBon",
                "date_from": "dateFrom",
                "date_to": "dateTo",
                "nom_porteur": "nomPorteur",
            },
            "auth": "bearer",
            "roles": ["OPTICIEN", "ADMIN", "OPERATEUR"],
        },
    },
    {
        "name": "get_orders_by_client",
        "description": (
            "Récupère les commandes d'un client donné par son codeClient. "
            "Utile quand l'opticien veut voir l'historique d'un porteur."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code_client": {"type": "string"},
                "page": {"type": "integer", "default": 1},
                "page_size": {"type": "integer", "default": 10},
            },
            "required": ["code_client"],
        },
        "http": {
            "method": "GET",
            "path": "/api/commandes/by-client-code/{code_client}",
            "param_map": {
                "code_client": "codeClient",
                "page_size": "pageSize",
            },
            "query": ["page", "page_size"],
            "auth": "bearer",
            "roles": ["OPTICIEN", "ADMIN", "OPERATEUR"],
        },
    },
    {
        "name": "get_opticien_profile",
        "description": (
            "Profil de l'opticien connecté (agence, contacts, codeClient, "
            "et dernières commandes de la session)."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
        "http": {
            "method": "GET",
            "path": "/api/opticien",
            "auth": "bearer",
            "roles": ["OPTICIEN"],
        },
    },
    {
        "name": "verify_client",
        "description": (
            "Vérifie qu'un codeClient existe (utile avant un suivi public ou la création d'une commande)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code_client": {"type": "string"},
            },
            "required": ["code_client"],
        },
        "http": {
            "method": "GET",
            "path": "/api/commandes/clients/verify/{code_client}",
            "param_map": {"code_client": "codeClient"},
            "auth": "bearer",
            "roles": ["OPTICIEN", "ADMIN", "OPERATEUR"],
        },
    },
]


def list_optiflow_tools(role: str | None = None) -> list[dict[str, Any]]:
    """Return the LLM-facing manifest, optionally filtered by USER_ROLE."""
    role_norm = (role or "").strip().upper()
    visible: list[dict[str, Any]] = []
    for tool in OPTIFLOW_TOOLS:
        allowed = tool.get("http", {}).get("roles")
        if allowed and role_norm and role_norm not in {r.upper() for r in allowed}:
            continue
        visible.append({
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
        })
    return visible


def get_optiflow_tool(name: str) -> dict[str, Any] | None:
    for tool in OPTIFLOW_TOOLS:
        if tool["name"] == name:
            return tool
    return None
