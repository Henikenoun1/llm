"""Lightweight schema test for the extended ChatRequest model."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    session_id: str
    model_variant: str = "prod"
    runtime_mode: str | None = None
    user_context: dict[str, Any] = Field(default_factory=dict)
    system: str | None = None
    access_token: str | None = None
    backend_url: str | None = None
    available_tools: list[dict[str, Any]] = Field(default_factory=list)


payload = {
    "message": "wach 7el commande mte3i",
    "session_id": "test-session-001",
    "model_variant": "prod",
    "runtime_mode": "speak",
    "system": "ReAct system prompt here",
    "access_token": "Bearer eyJhbGciOi...",
    "backend_url": "http://localhost:3001",
    "user_context": {
        "user_id": "u-1",
        "prenom": "Ahmed",
        "role": "OPTICIEN",
        "code_client": "C00042",
    },
    "available_tools": [
        {"name": "track_order", "description": "..."},
        {"name": "get_my_orders", "description": "..."},
    ],
}

req = ChatRequest(**payload)
assert req.system and req.system.startswith("ReAct"), "system not stored"
assert req.access_token and req.access_token.startswith("Bearer "), "access_token not stored"
assert req.backend_url == "http://localhost:3001"
assert req.user_context["role"] == "OPTICIEN"
assert len(req.available_tools) == 2
print("OK ChatRequest accepts OptiFlow extensions:")
print({
    "system": req.system[:20] + "...",
    "access_token_prefix": req.access_token[:10] + "...",
    "backend_url": req.backend_url,
    "user_context_keys": sorted(req.user_context.keys()),
    "tools": [t["name"] for t in req.available_tools],
})
