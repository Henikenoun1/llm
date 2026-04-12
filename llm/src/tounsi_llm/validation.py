"""
Project validation and coherence checks.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .config import (
    ADAPTERS_DIR,
    CFG,
    CONFIG_DIR,
    DOMAIN_CFG,
    EVAL_DIR,
    FEW_SHOTS_CFG,
    KB_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    resolve_project_path,
)
from .data_prep import _build_rag_grounded_conversations, _load_jsonl, _load_rag_grounded_self_sup_texts
from .data_sources import dataset_specs, detect_script
from .domain_utils import canonicalize_intent, canonicalize_slots, normalize_messages
from .storage import get_database_backend
from .tools import get_tool_registry


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _required_columns(path: Path, expected: list[str]) -> list[str]:
    if not path.exists():
        return expected
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",") if path.read_text(encoding="utf-8").splitlines() else []
    return [column for column in expected if column not in header]


def _json_state_stats(path: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "valid_json": False,
        "sessions": 0,
    }
    if not path.exists():
        return stats

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        stats["valid_json"] = isinstance(payload, dict)
        if isinstance(payload, dict):
            stats["sessions"] = len(payload)
    except Exception as exc:
        stats["error"] = str(exc)
    return stats


def _jsonl_stats(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "parent_exists": path.parent.exists(),
        "rows": _count_jsonl_rows(path) if path.exists() else 0,
    }


def _mask_database_url(url: str | None) -> str | None:
    if not url:
        return None
    if "://" not in url or "@" not in url:
        return url
    scheme, rest = url.split("://", 1)
    creds, tail = rest.split("@", 1)
    if ":" in creds:
        username = creds.split(":", 1)[0]
        return f"{scheme}://{username}:***@{tail}"
    return f"{scheme}://***@{tail}"


def _intent_counts(rows: list[dict[str, Any]], field: str = "intent") -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter[canonicalize_intent(row.get(field))] += 1
    return dict(sorted(counter.items()))


def _script_distribution_from_texts(texts: list[str]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for text in texts:
        script = detect_script(str(text))
        counts[script] = counts.get(script, 0) + 1
    total = sum(counts.values())
    percentages = {
        key: round(value / max(total, 1) * 100, 2)
        for key, value in sorted(counts.items())
    }
    return {
        "total": total,
        "counts": dict(sorted(counts.items())),
        "percentages": percentages,
    }


def _self_sup_script_coverage(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "distribution": {"total": 0, "counts": {}, "percentages": {}}}
    rows = _load_jsonl(path)
    texts = [str(row.get("text", "")) for row in rows if row.get("text")]
    return {
        "exists": True,
        "distribution": _script_distribution_from_texts(texts),
    }


def _sft_script_coverage(path: Path) -> dict[str, Any]:
    if not path.exists():
        empty = {"total": 0, "counts": {}, "percentages": {}}
        return {"exists": False, "user": empty, "assistant": empty}

    rows = _load_jsonl(path)
    user_texts: list[str] = []
    assistant_texts: list[str] = []
    for row in rows:
        messages = normalize_messages(row.get("messages", []) if isinstance(row.get("messages"), list) else [])
        for message in messages:
            if message["role"] == "user":
                user_texts.append(message["content"])
            elif message["role"] == "assistant":
                assistant_texts.append(message["content"])

    return {
        "exists": True,
        "user": _script_distribution_from_texts(user_texts),
        "assistant": _script_distribution_from_texts(assistant_texts),
    }


def _training_profile_summary() -> dict[str, Any]:
    return {
        "target_runtime_profile": CFG.target_runtime_profile,
        "hardware_expectation": "NVIDIA T4 16GB (Ubuntu VM)",
        "base_model": CFG.base_model,
        "batch_size": CFG.batch_size,
        "gradient_accumulation_steps": CFG.grad_accum_steps,
        "lora": {
            "rank": CFG.lora_rank,
            "alpha": CFG.lora_alpha,
            "dropout": CFG.lora_dropout,
        },
        "self_sup": {
            "epochs": CFG.epochs_self_sup,
            "learning_rate": CFG.self_sup_learning_rate,
            "max_steps": CFG.self_sup_max_steps,
            "max_seq_len": CFG.self_sup_max_seq_len,
        },
        "sft": {
            "epochs": CFG.epochs_sft,
            "learning_rate": CFG.sft_learning_rate,
            "max_steps": CFG.sft_max_steps,
            "max_seq_len": CFG.sft_max_seq_len,
        },
        "dpo": {
            "epochs": CFG.epochs_dpo,
            "learning_rate": CFG.dpo_learning_rate,
            "max_steps": CFG.dpo_max_steps,
            "max_seq_len": CFG.dpo_max_seq_len,
        },
        "optimizer": CFG.optim,
        "scheduler": CFG.lr_scheduler_type,
        "save_total_limit": CFG.save_total_limit,
        "early_stopping_patience": CFG.early_stopping_patience,
    }


def _processed_rows(processed_summary: dict[str, Any], split_name: str) -> int:
    split = processed_summary.get(split_name, {})
    if not isinstance(split, dict):
        return 0
    return int(split.get("rows", 0) or 0)


def _few_shot_schema_stats() -> dict[str, Any]:
    total = len(FEW_SHOTS_CFG)
    with_user = sum(1 for row in FEW_SHOTS_CFG if row.get("user") or row.get("client") or row.get("opticien"))
    with_assistant = sum(1 for row in FEW_SHOTS_CFG if row.get("assistant") or row.get("agent"))
    return {
        "rows": total,
        "rows_with_user_side": with_user,
        "rows_with_assistant_side": with_assistant,
        "intent_counts": _intent_counts(FEW_SHOTS_CFG),
    }


def _processed_role_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    rows = _load_jsonl(path)
    invalid = 0
    for row in rows:
        messages = row.get("messages", [])
        normalized = normalize_messages(messages if isinstance(messages, list) else [])
        if not normalized or any(message["role"] not in {"system", "user", "assistant"} for message in normalized):
            invalid += 1
    return {
        "exists": True,
        "rows": len(rows),
        "invalid_role_rows": invalid,
    }


_SUPPORTED_RAG_SUFFIXES = {".md", ".txt", ".jsonl", ".csv", ".json"}


def _iter_rag_source_files() -> list[Path]:
    files: list[Path] = []
    rag_cfg = DOMAIN_CFG.get("rag", {})
    source_dirs = rag_cfg.get("source_dirs", DOMAIN_CFG.get("rag_source_dirs", ["data/rag"]))
    for source_dir in source_dirs:
        root = resolve_project_path(source_dir)
        if not root.exists():
            continue
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in _SUPPORTED_RAG_SUFFIXES
        )
    return sorted(set(files))


def _mojibake_marker_count(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="ignore")
    return sum(text.count(marker) for marker in ("Ã", "ï¿½"))


def _rag_asset_summary() -> dict[str, Any]:
    files = _iter_rag_source_files()
    normalized_paths = {path: str(path).replace("\\", "/").lower() for path in files}
    internal_files = [path for path in files if "/data/rag/internal/" in normalized_paths[path]]
    external_files = [path for path in files if "/data/rag/external/" in normalized_paths[path]]
    kb_files = [path for path in files if "/data/kb/" in normalized_paths[path]]
    file_types = Counter(path.suffix.lower() for path in files)

    required_internal = {
        "agent_playbook.md",
        "intent_slot_playbook.md",
        "multi_turn_slot_examples.md",
    }
    present_internal = {path.name for path in internal_files}
    missing_internal = sorted(required_internal - present_internal)

    delivery_rag_present = any(
        "livraison" in path.name.lower() or "agence" in normalized_paths[path]
        for path in files
    )
    lens_rag_present = any(
        "verre" in path.name.lower() or "catalog" in path.name.lower()
        for path in files
    )
    mojibake_suspected_files = sorted(
        str(path) for path in files if _mojibake_marker_count(path) > 0
    )

    return {
        "source_file_count": len(files),
        "internal_file_count": len(internal_files),
        "external_file_count": len(external_files),
        "kb_file_count": len(kb_files),
        "file_types": dict(sorted(file_types.items())),
        "internal_files": [str(path) for path in internal_files],
        "external_files": [str(path) for path in external_files],
        "kb_files": [str(path) for path in kb_files],
        "missing_internal_playbooks": missing_internal,
        "delivery_rag_present": delivery_rag_present,
        "lens_rag_present": lens_rag_present,
        "mojibake_suspected_files": mojibake_suspected_files,
    }


def _rag_training_augmentation_summary() -> dict[str, Any]:
    conversations = _build_rag_grounded_conversations()
    self_sup_texts = _load_rag_grounded_self_sup_texts()
    sample_intents = Counter()
    for conversation in conversations:
        first_user_turn = next(
            (message.get("content", "") for message in conversation if message.get("role") == "user"),
            "",
        )
        lowered = str(first_user_turn).lower()
        if "planning livraison" in lowered or "livraison" in lowered:
            sample_intents["delivery_schedule"] += 1
        elif "reference" in lowered or "code verre" in lowered:
            sample_intents["reference_confirmation"] += 1
        elif "dispo" in lowered or "disponibil" in lowered:
            sample_intents["availability_inquiry"] += 1
        elif "suivi commande" in lowered:
            sample_intents["order_tracking"] += 1
        else:
            sample_intents["other"] += 1
    return {
        "sft_conversation_count": len(conversations),
        "self_sup_text_count": len(self_sup_texts),
        "sample_intents": dict(sorted(sample_intents.items())),
        "self_sup_script_distribution": _script_distribution_from_texts(self_sup_texts[:1000]),
    }


def _intent_required_slot_coverage(commandes_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows_by_intent: dict[str, list[dict[str, Any]]] = {}
    for row in commandes_rows:
        intent = canonicalize_intent(row.get("intent"))
        rows_by_intent.setdefault(intent, []).append(row)

    coverage: dict[str, Any] = {}
    for intent_name, required_slots in DOMAIN_CFG.get("required_slots", {}).items():
        intent = canonicalize_intent(intent_name)
        rows = rows_by_intent.get(intent, [])
        canonical_rows = [canonicalize_slots(row.get("slots", {})) for row in rows]
        complete_rows = sum(
            1 for slots in canonical_rows if all(slots.get(slot_name) for slot_name in required_slots)
        )
        coverage[intent] = {
            "rows": len(rows),
            "required_slots": list(required_slots),
            "complete_rows": complete_rows,
            "coverage_pct": round(complete_rows / max(len(rows), 1) * 100, 2) if rows else 0.0,
        }
    return dict(sorted(coverage.items()))


def validate_domain_assets(write_report: bool = True) -> dict[str, Any]:
    commandes_rows = _load_jsonl(CONFIG_DIR / "commandes_intents.jsonl")
    eval_rows = _load_jsonl(EVAL_DIR / "inference_cases.jsonl")
    tool_registry = get_tool_registry()
    tool_names = {tool["name"] for tool in tool_registry.list_tools()}

    intents_with_keywords = {canonicalize_intent(intent) for intent in DOMAIN_CFG.get("intent_keywords", {}).keys()}
    intents_with_required_slots = {canonicalize_intent(intent) for intent in DOMAIN_CFG.get("required_slots", {}).keys()}
    intents_with_tool_mapping = {canonicalize_intent(intent) for intent in DOMAIN_CFG.get("intent_to_tool", {}).keys()}
    intents_from_commandes = {canonicalize_intent(row.get("intent")) for row in commandes_rows}
    intents_from_few_shots = {canonicalize_intent(row.get("intent")) for row in FEW_SHOTS_CFG}
    intents_from_eval = {canonicalize_intent(row.get("expected_intent")) for row in eval_rows}
    all_intents = {
        intent
        for intent in intents_from_commandes | intents_from_few_shots | intents_from_eval
        if intent and intent != "unknown"
    }

    tool_optional_intents = {"greeting", "thanks", "get_num_client", "unknown"}
    missing_keyword_defs = sorted(intent for intent in all_intents if intent not in intents_with_keywords)
    missing_tool_mapping = sorted(
        intent for intent in all_intents if intent not in intents_with_tool_mapping and intent not in tool_optional_intents
    )
    missing_required_slots = sorted(
        intent for intent in intents_with_tool_mapping if intent not in intents_with_required_slots
    )
    missing_tool_handlers = sorted(
        {
            tool_name
            for tool_name in DOMAIN_CFG.get("intent_to_tool", {}).values()
            if str(tool_name) not in tool_names
        }
    )

    eval_tool_mismatches = []
    for row in eval_rows:
        intent = canonicalize_intent(row.get("expected_intent"))
        expected_tool = row.get("expected_tool")
        configured_tool = DOMAIN_CFG.get("intent_to_tool", {}).get(intent)
        if expected_tool and expected_tool != configured_tool:
            eval_tool_mismatches.append(
                {
                    "id": row.get("id"),
                    "intent": intent,
                    "expected_tool": expected_tool,
                    "configured_tool": configured_tool,
                }
            )

    commandes_slot_summary = {
        "rows": len(commandes_rows),
        "intent_counts": _intent_counts(commandes_rows),
        "rows_with_num_client": sum(1 for row in commandes_rows if canonicalize_slots(row.get("slots", {})).get("num_client")),
        "rows_with_order_id": sum(1 for row in commandes_rows if canonicalize_slots(row.get("slots", {})).get("order_id")),
        "rows_with_product": sum(1 for row in commandes_rows if canonicalize_slots(row.get("slots", {})).get("product")),
        "rows_with_reference": sum(1 for row in commandes_rows if canonicalize_slots(row.get("slots", {})).get("reference")),
        "rows_with_optics": sum(
            1
            for row in commandes_rows
            if any(
                canonicalize_slots(row.get("slots", {})).get(field)
                for field in ["od_sphere", "od_cyl", "od_axis", "og_sphere", "og_cyl", "og_axis", "addition"]
            )
        ),
    }
    commandes_slot_summary["required_slot_coverage"] = _intent_required_slot_coverage(commandes_rows)

    few_shot_stats = _few_shot_schema_stats()
    rag_summary = _rag_asset_summary()
    rag_training_summary = _rag_training_augmentation_summary()
    files_summary = {
        "domain_config": str(CONFIG_DIR / "domain.json"),
        "few_shots_exists": (CONFIG_DIR / "few_shots.jsonl").exists(),
        "commandes_intents_exists": (CONFIG_DIR / "commandes_intents.jsonl").exists(),
        "eval_cases_exists": (EVAL_DIR / "inference_cases.jsonl").exists(),
        "orders_mock_exists": (KB_DIR / "orders_mock.csv").exists(),
        "lens_catalog_exists": (KB_DIR / "lens_catalog.csv").exists(),
        "stores_exists": (KB_DIR / "stores.csv").exists(),
        "policies_exists": (KB_DIR / "policies.jsonl").exists(),
    }

    raw_dataset_summary = []
    for spec in dataset_specs():
        output_file = RAW_DATA_DIR / spec.get("output_file", f"{spec.get('name', 'dataset')}.jsonl")
        raw_dataset_summary.append(
            {
                "name": spec.get("name"),
                "roles": spec.get("roles", []),
                "exists": output_file.exists(),
                "rows": _count_jsonl_rows(output_file) if output_file.exists() else 0,
            }
        )

    processed_summary = {
        "self_sup_train": {"exists": (PROCESSED_DATA_DIR / "self_sup_train.jsonl").exists(), "rows": _count_jsonl_rows(PROCESSED_DATA_DIR / "self_sup_train.jsonl")},
        "self_sup_val": {"exists": (PROCESSED_DATA_DIR / "self_sup_val.jsonl").exists(), "rows": _count_jsonl_rows(PROCESSED_DATA_DIR / "self_sup_val.jsonl")},
        "sft_train": _processed_role_stats(PROCESSED_DATA_DIR / "sft_train.jsonl"),
        "sft_val": _processed_role_stats(PROCESSED_DATA_DIR / "sft_val.jsonl"),
        "sft_test": _processed_role_stats(PROCESSED_DATA_DIR / "sft_test.jsonl"),
        "dpo_train": {"exists": (PROCESSED_DATA_DIR / "dpo_train.jsonl").exists(), "rows": _count_jsonl_rows(PROCESSED_DATA_DIR / "dpo_train.jsonl")},
        "dpo_val": {"exists": (PROCESSED_DATA_DIR / "dpo_val.jsonl").exists(), "rows": _count_jsonl_rows(PROCESSED_DATA_DIR / "dpo_val.jsonl")},
    }

    memory_cfg = DOMAIN_CFG.get("memory", {})
    session_state_path = resolve_project_path(memory_cfg.get("history_state_path", "data/history/session_state.json"))
    conversations_path = resolve_project_path(memory_cfg.get("conversation_log_path", "data/history/conversations.jsonl"))
    pending_learning_path = resolve_project_path(memory_cfg.get("pending_learning_path", "data/history/learning_pending.jsonl"))
    approved_learning_path = resolve_project_path(memory_cfg.get("learning_buffer_path", "data/history/learning_buffer.jsonl"))
    feedback_log_path = resolve_project_path(memory_cfg.get("feedback_log_path", "data/history/feedback_log.jsonl"))
    feedback_dpo_path = resolve_project_path(memory_cfg.get("approved_dpo_feedback_path", "data/history/feedback_dpo.jsonl"))
    ratings_log_path = resolve_project_path(memory_cfg.get("ratings_log_path", "data/history/ratings_log.jsonl"))
    admin_corrections_path = resolve_project_path(memory_cfg.get("admin_corrections_path", "data/history/admin_corrections.jsonl"))

    language_coverage = {
        "self_sup_train": _self_sup_script_coverage(PROCESSED_DATA_DIR / "self_sup_train.jsonl"),
        "sft_train": _sft_script_coverage(PROCESSED_DATA_DIR / "sft_train.jsonl"),
    }

    warnings: list[str] = []
    self_sup_scripts = language_coverage["self_sup_train"]["distribution"]["counts"] if language_coverage["self_sup_train"].get("exists") else {}
    sft_assistant_scripts = language_coverage["sft_train"]["assistant"]["counts"] if language_coverage["sft_train"].get("exists") else {}
    if self_sup_scripts and (self_sup_scripts.get("arabizi", 0) + self_sup_scripts.get("mixed", 0) == 0):
        warnings.append("Self-sup train has no arabizi/mixed script samples; add code-switch utterances.")
    if sft_assistant_scripts and sft_assistant_scripts.get("mixed", 0) == 0:
        warnings.append("SFT assistant data has no mixed-script turns; add professional code-switch examples.")
    if not language_coverage["self_sup_train"].get("exists"):
        warnings.append("Processed self-sup train split is absent. This is normal after a clean reset; run prepare on the VM.")
    if not language_coverage["sft_train"].get("exists"):
        warnings.append("Processed SFT train split is absent. This is normal after a clean reset; run prepare on the VM.")
    if rag_summary["missing_internal_playbooks"]:
        warnings.append(
            "Internal RAG playbooks missing: " + ", ".join(rag_summary["missing_internal_playbooks"])
        )
    if rag_summary["kb_file_count"] > 0:
        issues.append("RAG retriever sources include data/kb. Keep the official retriever grounded on data/rag only.")
    if rag_summary["mojibake_suspected_files"]:
        warnings.append(
            f"Some RAG files still contain mojibake markers: {len(rag_summary['mojibake_suspected_files'])} file(s)."
        )
    if rag_training_summary["self_sup_script_distribution"]["counts"].get("mixed", 0) == 0:
        warnings.append("RAG-grounded self-sup augmentation has no mixed-script samples; keep adding code-switch phrasing.")

    external_ssf_source = next(
        (row for row in raw_dataset_summary if str(row.get("name")) == "ssf_llm_sentiment_104k"),
        None,
    )
    if external_ssf_source and not external_ssf_source.get("exists"):
        warnings.append(
            "External SSF source 'ssf_llm_sentiment_104k' is not available. Place the file at "
            "data/tounsi_raw/enrichment/ssf_llm_sentiment_104k.jsonl for the 104k enrichment path."
        )

    kb_summary = {
        "orders_mock_missing_columns": _required_columns(
            KB_DIR / "orders_mock.csv",
            ["order_id", "phone", "city", "status", "eta_days", "created_at"],
        ),
        "lens_catalog_missing_columns": _required_columns(
            KB_DIR / "lens_catalog.csv",
            ["sku", "product", "index", "coating", "city", "price_min_dt", "price_max_dt", "stock_policy"],
        ),
        "stores_missing_columns": _required_columns(
            KB_DIR / "stores.csv",
            ["store_id", "city", "store_name", "hours_weekday", "hours_sat", "hours_sun", "phone_store"],
        ),
        "policies_rows": _count_jsonl_rows(KB_DIR / "policies.jsonl"),
    }

    issues = []
    for intent in missing_keyword_defs:
        issues.append(f"Intent missing keyword definition: {intent}")
    for intent in missing_tool_mapping:
        issues.append(f"Intent missing tool mapping: {intent}")
    for intent in missing_required_slots:
        issues.append(f"Intent missing required_slots entry: {intent}")
    for tool_name in missing_tool_handlers:
        issues.append(f"Configured tool has no handler: {tool_name}")
    for item in eval_tool_mismatches:
        issues.append(
            f"Eval/tool mismatch for {item['id']}: expected {item['expected_tool']} but config has {item['configured_tool']}"
        )
    for label, missing in kb_summary.items():
        if isinstance(missing, list) and missing:
            issues.append(f"{label} missing columns: {', '.join(missing)}")
    if not rag_summary["delivery_rag_present"]:
        issues.append("Delivery RAG sources are missing; delivery_schedule cannot be grounded.")
    if not rag_summary["lens_rag_present"]:
        issues.append("Lens/catalog RAG sources are missing; reference_confirmation and availability_inquiry cannot be grounded.")

    core_min_examples = {
        "order_tracking": 50,
        "create_order": 70,
        "delivery_schedule": 8,
        "reference_confirmation": 6,
        "availability_inquiry": 6,
    }
    low_example_intents = [
        intent
        for intent, minimum in core_min_examples.items()
        if int(commandes_slot_summary["intent_counts"].get(intent, 0)) < minimum
    ]
    if low_example_intents:
        warnings.append(
            "Some core intents still have low slot/example coverage in commandes_intents: "
            + ", ".join(
                f"{intent}<{core_min_examples[intent]}"
                for intent in low_example_intents
            )
        )

    db_backend = get_database_backend()
    configured_db_url = CFG.database_url
    configured_db_url_masked = _mask_database_url(configured_db_url)
    configured_db_url_lower = str(configured_db_url or "").lower()
    postgres_url_ok = configured_db_url_lower.startswith(
        (
            "postgresql://",
            "postgresql+psycopg://",
            "postgresql+psycopg2://",
            "postgres://",
        )
    )
    db_health = db_backend.health()
    db_counts = db_backend.counts() if db_health.get("enabled") else {}

    runtime_storage = {
        "session_state": _json_state_stats(session_state_path),
        "conversations": _jsonl_stats(conversations_path),
        "learning_pending": _jsonl_stats(pending_learning_path),
        "learning_buffer": _jsonl_stats(approved_learning_path),
        "feedback_log": _jsonl_stats(feedback_log_path),
        "feedback_dpo": _jsonl_stats(feedback_dpo_path),
        "ratings_log": _jsonl_stats(ratings_log_path),
        "admin_corrections": _jsonl_stats(admin_corrections_path),
        "database": {
            "health": db_health,
            "counts": db_counts,
        },
    }

    session_storage_ok = bool(
        runtime_storage["session_state"].get("exists")
        and runtime_storage["session_state"].get("valid_json")
        and (
            runtime_storage["conversations"].get("exists")
            or runtime_storage["conversations"].get("parent_exists")
        )
    )
    if CFG.database_required_for_production and configured_db_url_lower.startswith("sqlite"):
        warnings.append("SQLite is not allowed for production readiness; configure CALL_CENTER_DATABASE_URL with PostgreSQL.")

    if CFG.database_required_for_production and not db_health.get("enabled"):
        warnings.append("Database is required for production but not healthy/available.")
    elif not db_health.get("enabled"):
        warnings.append("Database is disabled; production deployments should provide CALL_CENTER_DATABASE_URL.")

    active_production_marker = ADAPTERS_DIR / "active_production.json"
    active_production_payload: dict[str, Any] = {}
    if active_production_marker.exists():
        try:
            active_production_payload = json.loads(active_production_marker.read_text(encoding="utf-8"))
        except Exception:
            active_production_payload = {"error": "invalid_active_production_json"}

    preflight_checks: list[dict[str, Any]] = [
        {
            "name": "coherence_issues",
            "passed": len(issues) == 0,
            "issues": len(issues),
        },
        {
            "name": "kb_assets_present",
            "passed": (
                files_summary["orders_mock_exists"]
                and files_summary["lens_catalog_exists"]
                and files_summary["stores_exists"]
                and files_summary["policies_exists"]
                and not kb_summary["orders_mock_missing_columns"]
                and not kb_summary["lens_catalog_missing_columns"]
                and not kb_summary["stores_missing_columns"]
                and kb_summary["policies_rows"] > 0
            ),
            "files": files_summary,
            "kb": kb_summary,
        },
        {
            "name": "rag_assets_present",
            "passed": (
                rag_summary["source_file_count"] > 0
                and rag_summary["delivery_rag_present"]
                and rag_summary["lens_rag_present"]
            ),
            "rag": rag_summary,
        },
        {
            "name": "internal_playbooks_present",
            "passed": not rag_summary["missing_internal_playbooks"],
            "missing_internal_playbooks": rag_summary["missing_internal_playbooks"],
        },
        {
            "name": "rag_grounded_training_augmentation",
            "passed": (
                rag_training_summary["sft_conversation_count"] >= 100
                and rag_training_summary["self_sup_text_count"] >= 200
            ),
            "rag_training": rag_training_summary,
        },
        {
            "name": "core_intent_examples",
            "passed": not low_example_intents,
            "low_example_intents": low_example_intents,
            "thresholds": core_min_examples,
        },
        {
            "name": "few_shot_examples_present",
            "passed": (
                few_shot_stats["rows"] >= 50
                and few_shot_stats["rows_with_user_side"] == few_shot_stats["rows"]
                and few_shot_stats["rows_with_assistant_side"] == few_shot_stats["rows"]
            ),
            "few_shots": few_shot_stats,
        },
        {
            "name": "target_runtime_profile_t4",
            "passed": CFG.target_runtime_profile == "t4_16gb",
            "profile": CFG.target_runtime_profile,
        },
    ]
    preflight_failing_checks = [check["name"] for check in preflight_checks if not check.get("passed")]
    preflight_readiness = {
        "go_no_go": "GO" if not preflight_failing_checks else "NO_GO",
        "failing_checks": preflight_failing_checks,
        "checks": preflight_checks,
    }

    self_sup_rows = _processed_rows(processed_summary, "self_sup_train")
    sft_rows = _processed_rows(processed_summary, "sft_train")
    dpo_rows = _processed_rows(processed_summary, "dpo_train")
    readiness_checks: list[dict[str, Any]] = [
        {
            "name": "coherence_issues",
            "passed": len(issues) == 0,
            "issues": len(issues),
        },
        {
            "name": "self_sup_train_min_rows",
            "passed": self_sup_rows >= CFG.min_self_sup_train_rows,
            "rows": self_sup_rows,
            "required": CFG.min_self_sup_train_rows,
        },
        {
            "name": "sft_train_min_rows",
            "passed": sft_rows >= CFG.min_sft_train_rows,
            "rows": sft_rows,
            "required": CFG.min_sft_train_rows,
        },
        {
            "name": "dpo_train_min_rows",
            "passed": dpo_rows >= CFG.min_dpo_train_rows,
            "rows": dpo_rows,
            "required": CFG.min_dpo_train_rows,
        },
        {
            "name": "production_adapter_promoted",
            "passed": active_production_marker.exists(),
            "marker": str(active_production_marker),
        },
        {
            "name": "session_conversation_storage",
            "passed": session_storage_ok,
            "session_state": runtime_storage["session_state"],
            "conversations": runtime_storage["conversations"],
        },
        {
            "name": "database_engine_postgres",
            "passed": postgres_url_ok if CFG.database_required_for_production else True,
            "required": CFG.database_required_for_production,
            "configured_url": configured_db_url_masked,
        },
        {
            "name": "database_health",
            "passed": bool(db_health.get("enabled")) if CFG.database_required_for_production else True,
            "required": CFG.database_required_for_production,
            "health": db_health,
        },
    ]
    failing_checks = [check["name"] for check in readiness_checks if not check.get("passed")]
    production_readiness = {
        "go_no_go": "GO" if not failing_checks else "NO_GO",
        "failing_checks": failing_checks,
        "checks": readiness_checks,
        "thresholds": {
            "min_self_sup_train_rows": CFG.min_self_sup_train_rows,
            "min_sft_train_rows": CFG.min_sft_train_rows,
            "min_dpo_train_rows": CFG.min_dpo_train_rows,
            "database_required_for_production": CFG.database_required_for_production,
        },
        "database": db_health,
        "active_production": active_production_payload,
    }

    report = {
        "summary": {
            "issue_count": len(issues),
            "warnings_count": len(warnings),
            "all_intents": sorted(all_intents),
            "tools_available": sorted(tool_names),
            "preflight_go_no_go": preflight_readiness["go_no_go"],
            "go_no_go": production_readiness["go_no_go"],
        },
        "files": files_summary,
        "training_profile": _training_profile_summary(),
        "language_coverage": language_coverage,
        "few_shots": few_shot_stats,
        "commandes_intents": commandes_slot_summary,
        "eval_cases": {
            "rows": len(eval_rows),
            "intent_counts": _intent_counts(eval_rows, field="expected_intent"),
            "tool_mismatches": eval_tool_mismatches,
        },
        "raw_datasets": raw_dataset_summary,
        "processed": processed_summary,
        "kb": kb_summary,
        "rag": rag_summary,
        "rag_training": rag_training_summary,
        "runtime_storage": runtime_storage,
        "preflight_readiness": preflight_readiness,
        "production_readiness": production_readiness,
        "warnings": warnings,
        "issues": issues,
    }

    if write_report:
        json_path = REPORTS_DIR / "validation_report.json"
        md_path = REPORTS_DIR / "validation_report.md"
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        lines = [
            "# Validation Report",
            "",
            f"- issue_count: {report['summary']['issue_count']}",
            f"- preflight_go_no_go: {report['summary']['preflight_go_no_go']}",
            f"- go_no_go: {report['summary']['go_no_go']}",
            f"- intents: {', '.join(report['summary']['all_intents'])}",
            f"- tools_available: {', '.join(report['summary']['tools_available'])}",
            "",
            "## Issues",
            "",
        ]
        if issues:
            for issue in issues:
                lines.append(f"- {issue}")
        else:
            lines.append("- No blocking coherence issue detected.")
        lines.extend(
            [
                "",
                "## Warnings",
                "",
            ]
        )
        if warnings:
            for warning in warnings:
                lines.append(f"- {warning}")
        else:
            lines.append("- No warning.")
        lines.extend(
            [
                "",
                "## Preflight Readiness",
                "",
                f"- verdict: {preflight_readiness['go_no_go']}",
            ]
        )
        if preflight_readiness["failing_checks"]:
            lines.append(f"- failing_checks: {', '.join(preflight_readiness['failing_checks'])}")
        else:
            lines.append("- failing_checks: none")
        for check in preflight_readiness["checks"]:
            lines.append(f"- {check['name']}: passed={check['passed']} details={check}")
        lines.extend(
            [
                "",
                "## Training Profile",
                "",
                f"- profile: {report['training_profile']['target_runtime_profile']}",
                f"- hardware: {report['training_profile']['hardware_expectation']}",
                f"- batch_size: {report['training_profile']['batch_size']}",
                f"- grad_accum: {report['training_profile']['gradient_accumulation_steps']}",
                f"- self_sup: {report['training_profile']['self_sup']}",
                f"- sft: {report['training_profile']['sft']}",
                f"- dpo: {report['training_profile']['dpo']}",
                "",
                "## RAG Assets",
                "",
                f"- rag: {rag_summary}",
                "",
                "## Processed Data",
                "",
            ]
        )
        for split_name, stats in processed_summary.items():
            lines.append(f"- {split_name}: {stats}")
        lines.extend(
            [
                "",
                "## Production Readiness",
                "",
                f"- verdict: {production_readiness['go_no_go']}",
            ]
        )
        if production_readiness["failing_checks"]:
            lines.append(f"- failing_checks: {', '.join(production_readiness['failing_checks'])}")
        else:
            lines.append("- failing_checks: none")
        for check in production_readiness["checks"]:
            lines.append(f"- {check['name']}: passed={check['passed']} details={check}")
        md_path.write_text("\n".join(lines), encoding="utf-8")

    return report
