"""End-to-end test of execute_optiflow_tool against the running OptiFlow backend (v2)."""
from __future__ import annotations

import json
import sys
import urllib.request

BACKEND = "http://localhost:3001"

ADMIN = {"email": "admin@optiflow.com", "password": "Admin123!Secure"}
OPTICIEN_CANDIDATES = [
    {"email": "opticien02@optiflow.local", "password": "Optiflow2025!"},
    {"email": "opticien02@optiflow.local", "password": "Opticien123!"},
    {"email": "opticien02@optiflow.local", "password": "Password123!"},
    {"email": "opticien02@optiflow.local", "password": "ChangeMe123!"},
]

KNOWN_ORDER_ID = "binome20-commande-stock-20"
KNOWN_NUM_CMD = "BSTK0020"
KNOWN_CODE_CLIENT = "BINC002"


def login(email: str, password: str) -> str | None:
    body = json.dumps({"email": email, "password": password}).encode("utf-8")
    req = urllib.request.Request(
        f"{BACKEND}/api/auth/sign-in",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"[login FAIL] {email}: {exc}")
        return None
    token = payload.get("accessToken")
    if token:
        print(f"[login OK] {email} token={token[:20]}...")
    return token


def main() -> int:
    sys.path.insert(0, "src")
    from tounsi_llm.optiflow_backend import execute_optiflow_tool

    admin_token = login(**ADMIN)
    if not admin_token:
        return 1

    opticien_token = None
    for candidate in OPTICIEN_CANDIDATES:
        opticien_token = login(**candidate)
        if opticien_token:
            print(f"[opticien creds] {candidate['email']} / {candidate['password']}")
            break

    cases = [
        ("track_order_real", {"name": "track_order", "args": {"code_client": KNOWN_CODE_CLIENT, "reference": KNOWN_NUM_CMD}, "access_token": None}),
        ("track_order_unknown", {"name": "track_order", "args": {"code_client": "ZZZ999", "reference": "DOES-NOT-EXIST"}, "access_token": None}),
        ("get_order_detail_real", {"name": "get_order_detail", "args": {"id": KNOWN_ORDER_ID}, "access_token": admin_token}),
        ("search_orders_admin", {"name": "search_orders", "args": {"num_cmd": "BSTK"}, "access_token": admin_token}),
        ("verify_client_real", {"name": "verify_client", "args": {"code_client": KNOWN_CODE_CLIENT}, "access_token": admin_token}),
        ("get_orders_by_client_real", {"name": "get_orders_by_client", "args": {"code_client": KNOWN_CODE_CLIENT, "page": 1, "page_size": 5}, "access_token": admin_token}),
        ("unknown_tool", {"name": "do_something_dangerous", "args": {}, "access_token": admin_token}),
        ("get_my_orders_no_token", {"name": "get_my_orders", "args": {}, "access_token": None}),
    ]

    if opticien_token:
        cases.extend([
            ("get_my_orders_opticien", {"name": "get_my_orders", "args": {"page": 1, "page_size": 3}, "access_token": opticien_token}),
            ("get_opticien_profile", {"name": "get_opticien_profile", "args": {}, "access_token": opticien_token}),
        ])

    for label, kwargs in cases:
        result = execute_optiflow_tool(backend_url=BACKEND, **kwargs)
        compact = {"status": result.get("status"), "http": result.get("http_status")}
        if result.get("error"):
            compact["error"] = result["error"][:80]
        data = result.get("data")
        if isinstance(data, dict):
            if "items" in data and isinstance(data["items"], list):
                compact["items_count"] = len(data["items"])
            elif data.get("status") and data.get("status") != "error":
                compact["payload_status"] = data["status"]
                if isinstance(data.get("order"), dict):
                    compact["order_numCmd"] = data["order"].get("numCmd")
            elif data.get("numCmd"):
                compact["numCmd"] = data["numCmd"]
                compact["statut"] = data.get("statut")
            else:
                compact["data_keys"] = list(data.keys())[:6]
        print(f"[{label}] {compact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
