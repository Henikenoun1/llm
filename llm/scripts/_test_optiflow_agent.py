"""Test the OptiFlow agent step (decide → execute) end-to-end."""
from __future__ import annotations

import json
import sys
import urllib.request

BACKEND = "http://localhost:3001"

def login(email: str, password: str) -> str:
    body = json.dumps({"email": email, "password": password}).encode("utf-8")
    req = urllib.request.Request(
        f"{BACKEND}/api/auth/sign-in",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))["accessToken"]


def main() -> int:
    sys.path.insert(0, "src")
    from tounsi_llm.optiflow_agent import run_optiflow_agent_step, decide_tool

    token = login("admin@optiflow.com", "Admin123!Secure")
    base_ctx = {
        "USER_ID": "admin-1",
        "USER_PRENOM": "Admin",
        "USER_ROLE": "ADMIN",
        "USER_CODE_CLIENT": "BINC002",
        "BACKEND_URL": BACKEND,
        "ACCESS_TOKEN": token,
    }

    print("--- decide_tool only ---")
    for intent in ["track_order", "list_my_orders", "search_orders", "get_order_detail", "verify_client", "out_of_scope_intent"]:
        d = decide_tool(intent=intent, slots={"reference": "BSTK0020", "code_client": "BINC002", "id": "binome20-commande-stock-20"}, user_context=base_ctx)
        print(f"  intent={intent} -> {d}")

    print()
    print("--- run_optiflow_agent_step ---")
    cases = [
        ("track from text", {"intent": "track_order", "slots": {"reference": "BSTK0020", "code_client": "BINC002"}}),
        ("track missing slots", {"intent": "track_order", "slots": {}}),
        ("get_order_detail", {"intent": "get_order_detail", "slots": {"id": "binome20-commande-stock-20"}}),
        ("search by num_cmd partial", {"intent": "search_orders", "slots": {"num_cmd": "BSTK"}}),
        ("verify_client", {"intent": "verify_client", "slots": {"code_client": "BINC002"}}),
        ("orders_by_client (uses USER_CODE_CLIENT)", {"intent": "client_orders", "slots": {}}),
        ("non-mappable intent", {"intent": "small_talk", "slots": {}}),
    ]
    for label, payload in cases:
        result = run_optiflow_agent_step(user_context=base_ctx, **payload)
        if result is None:
            print(f"[{label}] -> SKIP (no backend tool)")
            continue
        tc = result.get("tool_call") or {}
        tr = result.get("tool_result") or {}
        compact = {
            "tool": tc.get("name"),
            "args_keys": list((tc.get("arguments") or {}).keys()),
            "missing": result.get("missing"),
            "result_status": tr.get("status") if isinstance(tr, dict) else None,
            "result_http": tr.get("http_status") if isinstance(tr, dict) else None,
        }
        if isinstance(tr, dict):
            data = tr.get("data")
            if isinstance(data, dict):
                if "items" in data and isinstance(data["items"], list):
                    compact["items"] = len(data["items"])
                elif data.get("numCmd"):
                    compact["numCmd"] = data["numCmd"]
                elif data.get("statutLabel"):
                    compact["statutLabel"] = data["statutLabel"]
        print(f"[{label}] {compact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
