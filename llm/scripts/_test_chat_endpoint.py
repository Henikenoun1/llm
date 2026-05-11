"""Integration test: simulate a /chat call and verify our merge is propagated.

We monkey-patch the heavy startup dependencies so we don't have to load
LLM models. The point is to validate that:
  1. ChatRequest accepts the OptiFlow extensions (system, access_token, backend_url)
  2. The merge logic forwards them via user_context to production_infer
  3. The OptiFlow agent picks the matching tool when the intent matches
"""
from __future__ import annotations

import json
import sys
import urllib.request
from unittest.mock import MagicMock


def login() -> str:
    body = json.dumps({"email": "admin@optiflow.com", "password": "Admin123!Secure"}).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:3001/api/auth/sign-in",
        data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))["accessToken"]


def main() -> int:
    sys.path.insert(0, "src")

    # Patch heavy modules BEFORE importing server.
    import tounsi_llm.inference as inference_mod
    captured: dict = {}

    def fake_production_infer(**kwargs):
        captured.update(kwargs)
        # Run the real OptiFlow agent step to validate the wiring.
        from tounsi_llm.optiflow_agent import run_optiflow_agent_step
        normalized = inference_mod._normalize_user_context(kwargs.get("user_context"))
        step = run_optiflow_agent_step(
            intent="track_order",
            slots={"reference": "BSTK0020", "code_client": "BINC002"},
            user_context=normalized,
        )
        return {
            "response": "Test response (mocked)",
            "intent": "track_order",
            "slots": {"reference": "BSTK0020", "code_client": "BINC002"},
            "missing_slots": [],
            "tool_call": (step or {}).get("tool_call"),
            "tool_result": (step or {}).get("tool_result"),
            "rag_results": [],
            "memory_hits": [],
            "session_state": {},
            "needs_human_review": False,
            "latency_ms": 1.0,
            "model_variant": kwargs.get("model_variant", "prod"),
            "runtime_mode": kwargs.get("runtime_mode", "speak"),
            "response_source": "mock",
            "response_script_target": "arabic",
            "response_script_detected": "arabic",
            "correction_applied": False,
        }

    inference_mod.production_infer = fake_production_infer

    # Now import the server module.
    import tounsi_llm.server as srv

    # Bypass startup (no models, no RAG): inject light stubs.
    srv._retriever = MagicMock()
    srv._retriever.search.return_value = []
    srv._memory_store = MagicMock()
    srv._memory_store.get_session_history.return_value = []
    srv._memory_store.append_exchange = MagicMock()
    srv._tool_registry = MagicMock()
    srv._correction_store = MagicMock()
    srv._correction_store.find_best.return_value = None

    # Disable API key auth path
    srv.CFG.require_api_key = False

    from fastapi.testclient import TestClient
    client = TestClient(srv.app)

    token = login()
    payload = {
        "message": "wach 7el commande mte3i BSTK0020 client BINC002",
        "session_id": "test-session-step7",
        "model_variant": "prod",
        "runtime_mode": "speak",
        "system": "ReAct system prompt for OptiFlow",
        "access_token": token,
        "backend_url": "http://localhost:3001",
        "user_context": {
            "user_id": "admin-1",
            "prenom": "Admin",
            "role": "ADMIN",
            "code_client": "BINC002",
        },
        "available_tools": [{"name": "track_order"}],
    }

    response = client.post("/chat", json=payload)
    print("HTTP", response.status_code)
    body = response.json()

    print("[merge] access_token in user_context?",
          "access_token" in (captured.get("user_context") or {}))
    print("[merge] backend_url in user_context?",
          "backend_url" in (captured.get("user_context") or {}))
    print("[merge] system_prompt in user_context?",
          "system_prompt" in (captured.get("user_context") or {}))
    print("[merge] available_tools forwarded?",
          "available_tools" in (captured.get("user_context") or {}))

    print("[response] intent:", body.get("intent"))
    print("[response] tool_call:", body.get("tool_call"))
    tr = body.get("tool_result") or {}
    print("[response] tool_result.status:", tr.get("status"))
    print("[response] tool_result.http_status:", tr.get("http_status"))
    data = tr.get("data") or {}
    if isinstance(data, dict):
        print("[response] order statutLabel:", data.get("statutLabel"))

    return 0 if response.status_code == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
