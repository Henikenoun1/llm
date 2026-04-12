"""
Comprehensive data and inference evaluation utilities.
"""
from __future__ import annotations

import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any

from rouge_score import rouge_scorer

from .config import EVAL_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, logger
from .corrections import LiveCorrectionStore
from .data_prep import FUSHA_MARKERS, _load_jsonl, is_clean_tounsi
from .domain_utils import canonicalize_intent, canonicalize_slots, normalize_messages
from .inference import production_infer
from .memory import ConversationMemoryStore
from .rag import VectorRAGRetriever
from .tools import get_tool_registry
from .validation import validate_domain_assets

try:  # pragma: no cover - optional dependency
    import sacrebleu
except Exception:  # pragma: no cover - optional dependency
    sacrebleu = None


_CHINESE_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)


DEFAULT_EVAL_CASES = [
    {
        "id": "greeting_basic",
        "input": "عسلامة شنحوالك",
        "expected_intent": "greeting",
        "reference_responses": ["عسلامة، مرحبا بيك. كيفاش نجم نعاونك؟"],
        "expected_human_review": False,
    },
    {
        "id": "price_progressive",
        "input": "قداش سوم البلارات البروقرسيف 1.67 في تونس",
        "expected_intent": "price_inquiry",
        "expected_slots": {"product": "progressive", "index": "1.67", "city": "Tunis"},
        "expected_tool": "get_price",
        "reference_responses": ["نجم نعطيك السعر حسب البرودوي والانديس في تونس."],
        "expected_human_review": False,
    },
    {
        "id": "track_order",
        "input": "نحب نتبع الكوموند ORD-ABC12345",
        "expected_intent": "order_tracking",
        "expected_slots": {"order_id": "ORD-ABC12345"},
        "expected_tool": "track_order",
        "reference_responses": ["نجم نتثبتلك في حالة الكوموند متاعك."],
        "expected_human_review": False,
    },
    {
        "id": "order_missing_phone",
        "input": "نحب نعمل كوموند progressive 1.67 في تونس",
        "expected_intent": "create_order",
        "expected_slots": {"product": "progressive", "index": "1.67", "city": "Tunis"},
        "expected_tool": "create_order",
        "expected_missing_slots": ["num_client"],
        "expected_human_review": True,
        "reference_responses": ["يلزمني num client باش نكمل الطلبية."],
    },
]


def _tool_precision_recall_f1(rows: list[dict[str, Any]]) -> dict[str, float]:
    expected = sum(1 for row in rows if row.get("tool_expected"))
    predicted = sum(1 for row in rows if row.get("tool_detected"))
    true_positive = sum(
        1
        for row in rows
        if row.get("tool_expected")
        and row.get("tool_detected")
        and row.get("tool_expected") == row.get("tool_detected")
    )

    precision = true_positive / max(predicted, 1)
    recall = true_positive / max(expected, 1)
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    return {
        "tool_precision": round(precision, 4),
        "tool_recall": round(recall, 4),
        "tool_f1": round(f1, 4),
    }


def _intent_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for row in rows:
        expected_intent = str(row.get("intent_expected") or "unknown")
        bucket = summary.setdefault(
            expected_intent,
            {
                "cases": 0,
                "intent_accuracy": 0.0,
                "slot_f1": 0.0,
                "tool_accuracy": 0.0,
                "human_review_accuracy": 0.0,
            },
        )
        bucket["cases"] += 1
        bucket["intent_accuracy"] += 1.0 if row.get("intent_correct") else 0.0
        bucket["slot_f1"] += float(row.get("slot_f1", 0.0))
        bucket["tool_accuracy"] += 1.0 if row.get("tool_correct") else 0.0
        bucket["human_review_accuracy"] += 1.0 if row.get("human_review_correct") else 0.0

    for bucket in summary.values():
        cases = max(int(bucket["cases"]), 1)
        bucket["intent_accuracy"] = round(bucket["intent_accuracy"] / cases * 100, 2)
        bucket["slot_f1"] = round(bucket["slot_f1"] / cases, 4)
        bucket["tool_accuracy"] = round(bucket["tool_accuracy"] / cases * 100, 2)
        bucket["human_review_accuracy"] = round(bucket["human_review_accuracy"] / cases * 100, 2)
    return dict(sorted(summary.items()))


def _intent_confusion(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for row in rows:
        expected_intent = str(row.get("intent_expected") or "unknown")
        detected_intent = str(row.get("intent_detected") or "unknown")
        matrix.setdefault(expected_intent, {})
        matrix[expected_intent][detected_intent] = matrix[expected_intent].get(detected_intent, 0) + 1
    return {
        expected: dict(sorted(predicted.items()))
        for expected, predicted in sorted(matrix.items(), key=lambda item: item[0])
    }


def _safe_mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _flatten_messages(messages: list[dict[str, Any]]) -> tuple[str, str]:
    user_parts = []
    assistant_parts = []
    for message in normalize_messages(messages):
        role = message.get("role")
        content = str(message.get("content", ""))
        if role == "user":
            user_parts.append(content)
        elif role == "assistant":
            assistant_parts.append(content)
    return " ".join(user_parts).strip(), " ".join(assistant_parts).strip()


def _count_tokens(text: str) -> int:
    return len(str(text).split())


def _normalized_fingerprint(text: str) -> str:
    return " ".join(re.sub(r"\s+", " ", str(text).strip().lower()).split())


def _data_stats_for_self_sup(path: Path) -> dict[str, Any]:
    rows = _load_jsonl(path)
    texts = [str(row.get("text", "")) for row in rows if row.get("text")]
    duplicates = len(texts) - len({_normalized_fingerprint(text) for text in texts})
    return {
        "examples": len(texts),
        "avg_tokens": round(statistics.mean(_count_tokens(text) for text in texts), 2) if texts else 0,
        "avg_chars": round(statistics.mean(len(text) for text in texts), 2) if texts else 0,
        "duplicate_examples": duplicates,
        "clean_tounsi_rate": round(sum(is_clean_tounsi(text, max_fusha=3) for text in texts) / max(len(texts), 1) * 100, 2),
    }


def _data_stats_for_sft(path: Path) -> dict[str, Any]:
    rows = _load_jsonl(path)
    turns = []
    assistant_texts = []
    user_texts = []
    fingerprints = []
    for row in rows:
        messages = row.get("messages", [])
        if not isinstance(messages, list):
            continue
        turns.append(len(messages))
        user_text, assistant_text = _flatten_messages(messages)
        if user_text:
            user_texts.append(user_text)
        if assistant_text:
            assistant_texts.append(assistant_text)
        fingerprints.append(_normalized_fingerprint(f"{user_text} || {assistant_text}"))

    duplicates = len(fingerprints) - len(set(fingerprints))
    return {
        "examples": len(rows),
        "avg_turns": round(statistics.mean(turns), 2) if turns else 0,
        "avg_user_tokens": round(statistics.mean(_count_tokens(text) for text in user_texts), 2) if user_texts else 0,
        "avg_assistant_tokens": round(statistics.mean(_count_tokens(text) for text in assistant_texts), 2) if assistant_texts else 0,
        "assistant_clean_tounsi_rate": round(
            sum(is_clean_tounsi(text, max_fusha=2) for text in assistant_texts) / max(len(assistant_texts), 1) * 100,
            2,
        ),
        "duplicate_examples": duplicates,
    }


def _data_stats_for_dpo(path: Path) -> dict[str, Any]:
    rows = _load_jsonl(path)
    prompts = [str(row.get("prompt", "")) for row in rows if row.get("prompt")]
    chosen = [str(row.get("chosen", "")) for row in rows if row.get("chosen")]
    rejected = [str(row.get("rejected", "")) for row in rows if row.get("rejected")]
    malformed = sum(1 for row in rows if not (row.get("prompt") and row.get("chosen") and row.get("rejected")))
    return {
        "examples": len(rows),
        "avg_prompt_tokens": round(statistics.mean(_count_tokens(text) for text in prompts), 2) if prompts else 0,
        "avg_chosen_tokens": round(statistics.mean(_count_tokens(text) for text in chosen), 2) if chosen else 0,
        "avg_rejected_tokens": round(statistics.mean(_count_tokens(text) for text in rejected), 2) if rejected else 0,
        "malformed_examples": malformed,
    }


def evaluate_processed_data() -> dict[str, Any]:
    splits: dict[str, Any] = {}
    self_sup_train = PROCESSED_DATA_DIR / "self_sup_train.jsonl"
    self_sup_val = PROCESSED_DATA_DIR / "self_sup_val.jsonl"
    sft_train = PROCESSED_DATA_DIR / "sft_train.jsonl"
    sft_val = PROCESSED_DATA_DIR / "sft_val.jsonl"
    sft_test = PROCESSED_DATA_DIR / "sft_test.jsonl"
    dpo_train = PROCESSED_DATA_DIR / "dpo_train.jsonl"
    dpo_val = PROCESSED_DATA_DIR / "dpo_val.jsonl"

    if self_sup_train.exists():
        splits["self_sup_train"] = _data_stats_for_self_sup(self_sup_train)
    if self_sup_val.exists():
        splits["self_sup_val"] = _data_stats_for_self_sup(self_sup_val)
    if sft_train.exists():
        splits["sft_train"] = _data_stats_for_sft(sft_train)
    if sft_val.exists():
        splits["sft_val"] = _data_stats_for_sft(sft_val)
    if sft_test.exists():
        splits["sft_test"] = _data_stats_for_sft(sft_test)
    if dpo_train.exists():
        splits["dpo_train"] = _data_stats_for_dpo(dpo_train)
    if dpo_val.exists():
        splits["dpo_val"] = _data_stats_for_dpo(dpo_val)

    return {"processed_data": splits}


def _load_eval_cases(cases_path: Path | None = None) -> list[dict[str, Any]]:
    path = cases_path or (EVAL_DIR / "inference_cases.jsonl")
    if path.exists():
        rows = _load_jsonl(path)
        if rows:
            return rows
    return DEFAULT_EVAL_CASES


def _slot_metrics(expected: dict[str, Any], predicted: dict[str, Any]) -> dict[str, float]:
    expected_items = {key: str(value) for key, value in canonicalize_slots(expected).items()}
    predicted_items = {
        key: str(value) for key, value in canonicalize_slots(predicted).items() if value not in (None, "", [])
    }
    true_positive = sum(1 for key, value in predicted_items.items() if expected_items.get(key) == value)
    precision = true_positive / max(len(predicted_items), 1)
    recall = true_positive / max(len(expected_items), 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _best_text_overlap(prediction: str, references: list[str]) -> dict[str, float]:
    if not references:
        return {"rougeL_f1": 0.0, "rouge1_f1": 0.0, "rouge2_f1": 0.0, "bleu": 0.0, "chrf": 0.0}

    rouge_scores = []
    for reference in references:
        score = _ROUGE.score(reference, prediction)
        rouge_scores.append(
            {
                "rouge1_f1": score["rouge1"].fmeasure,
                "rouge2_f1": score["rouge2"].fmeasure,
                "rougeL_f1": score["rougeL"].fmeasure,
            }
        )

    best_rouge = max(rouge_scores, key=lambda item: item["rougeL_f1"])
    bleu = 0.0
    chrf = 0.0
    if sacrebleu is not None:  # pragma: no branch - optional dependency
        try:
            bleu = sacrebleu.corpus_bleu([prediction], [[ref] for ref in references]).score / 100
            chrf = sacrebleu.corpus_chrf([prediction], [[ref] for ref in references]).score / 100
        except Exception:
            bleu = 0.0
            chrf = 0.0
    return {
        "rouge1_f1": round(best_rouge["rouge1_f1"], 4),
        "rouge2_f1": round(best_rouge["rouge2_f1"], 4),
        "rougeL_f1": round(best_rouge["rougeL_f1"], 4),
        "bleu": round(bleu, 4),
        "chrf": round(chrf, 4),
    }


def evaluate_inference(
    *,
    model_variant: str = "prod",
    runtime_mode: str = "autonomous",
    cases_path: Path | None = None,
    max_cases: int | None = None,
) -> dict[str, Any]:
    cases = _load_eval_cases(cases_path=cases_path)
    if max_cases:
        cases = cases[:max_cases]

    retriever = VectorRAGRetriever(refresh=True)
    memory_store = ConversationMemoryStore()
    correction_store = LiveCorrectionStore()
    tool_registry = get_tool_registry()

    details = []
    metric_rows = []
    for index, case in enumerate(cases):
        session_id = f"eval-case-{index}"
        memory_store.reset(session_id=session_id)
        started = time.perf_counter()
        try:
            output = production_infer(
                user_text=str(case.get("input", "")),
                retriever=retriever,
                history=[],
                session_id=session_id,
                model_variant=model_variant,
                runtime_mode=str(case.get("runtime_mode", runtime_mode)),
                memory_store=memory_store,
                tool_registry=tool_registry,
                correction_store=correction_store,
            )
        except Exception as exc:
            details.append({"id": case.get("id", f"case_{index}"), "input": case.get("input", ""), "error": str(exc)})
            continue

        latency = round((time.perf_counter() - started) * 1000, 2)
        response = str(output.get("response", ""))
        predicted_intent = canonicalize_intent(output.get("intent", ""))
        predicted_slots = canonicalize_slots(output.get("slots", {})) if isinstance(output.get("slots"), dict) else {}
        predicted_tool = None
        if isinstance(output.get("tool_call"), dict):
            predicted_tool = output["tool_call"].get("name")

        expected_intent = canonicalize_intent(case.get("expected_intent"))
        expected_slots = canonicalize_slots(case.get("expected_slots", {})) if isinstance(case.get("expected_slots"), dict) else {}
        slot_scores = _slot_metrics(expected_slots, predicted_slots)
        overlap = _best_text_overlap(response, [str(item) for item in case.get("reference_responses", [])])
        fusha_count = sum(1 for marker in FUSHA_MARKERS if marker in response)

        expected_keywords = [str(item).lower() for item in case.get("required_response_keywords", [])]
        forbidden_keywords = [str(item).lower() for item in case.get("forbidden_response_keywords", [])]
        response_lower = response.lower()
        keyword_pass = all(keyword in response_lower for keyword in expected_keywords) if expected_keywords else True
        forbidden_pass = not any(keyword in response_lower for keyword in forbidden_keywords)

        detail = {
            "id": case.get("id", f"case_{index}"),
            "input": case.get("input", ""),
            "response": response,
            "intent_expected": expected_intent,
            "intent_detected": predicted_intent,
            "intent_correct": predicted_intent == expected_intent,
            "tool_expected": case.get("expected_tool"),
            "tool_detected": predicted_tool,
            "tool_correct": (predicted_tool == case.get("expected_tool")) if case.get("expected_tool") else predicted_tool is None,
            "slots_expected": expected_slots,
            "slots_detected": predicted_slots,
            "slot_precision": slot_scores["precision"],
            "slot_recall": slot_scores["recall"],
            "slot_f1": slot_scores["f1"],
            "slot_exact_match": expected_slots == predicted_slots,
            "expected_human_review": bool(case.get("expected_human_review", False)),
            "predicted_human_review": bool(output.get("needs_human_review", False)),
            "human_review_correct": bool(output.get("needs_human_review", False)) == bool(case.get("expected_human_review", False)),
            "latency_ms": latency,
            "response_tokens": _count_tokens(response),
            "response_chars": len(response),
            "is_tounsi": is_clean_tounsi(response, max_fusha=3),
            "has_chinese": bool(_CHINESE_RE.search(response)),
            "fusha_count": fusha_count,
            "keyword_pass": keyword_pass,
            "forbidden_pass": forbidden_pass,
            "response_metrics": overlap,
            "runtime_mode": output.get("runtime_mode", runtime_mode),
            "model_variant": output.get("model_variant", model_variant),
        }
        metric_rows.append(detail)
        details.append(detail)

    valid = [row for row in metric_rows if "intent_correct" in row]
    tool_prf = _tool_precision_recall_f1(valid)
    summary = {
        "cases_total": len(cases),
        "cases_scored": len(valid),
        "errors": len(details) - len(valid),
        "intent_accuracy": round(sum(row["intent_correct"] for row in valid) / max(len(valid), 1) * 100, 2),
        "tool_accuracy": round(sum(row["tool_correct"] for row in valid) / max(len(valid), 1) * 100, 2),
        **tool_prf,
        "human_review_accuracy": round(sum(row["human_review_correct"] for row in valid) / max(len(valid), 1) * 100, 2),
        "slot_precision": round(sum(row["slot_precision"] for row in valid) / max(len(valid), 1), 4),
        "slot_recall": round(sum(row["slot_recall"] for row in valid) / max(len(valid), 1), 4),
        "slot_f1": round(sum(row["slot_f1"] for row in valid) / max(len(valid), 1), 4),
        "slot_exact_match_rate": round(sum(row["slot_exact_match"] for row in valid) / max(len(valid), 1) * 100, 2),
        "response_rougeL_f1": round(sum(row["response_metrics"]["rougeL_f1"] for row in valid) / max(len(valid), 1), 4),
        "response_rouge1_f1": round(sum(row["response_metrics"]["rouge1_f1"] for row in valid) / max(len(valid), 1), 4),
        "response_rouge2_f1": round(sum(row["response_metrics"]["rouge2_f1"] for row in valid) / max(len(valid), 1), 4),
        "response_bleu": round(sum(row["response_metrics"]["bleu"] for row in valid) / max(len(valid), 1), 4),
        "response_chrf": round(sum(row["response_metrics"]["chrf"] for row in valid) / max(len(valid), 1), 4),
        "avg_response_tokens": round(sum(row["response_tokens"] for row in valid) / max(len(valid), 1), 2),
        "avg_response_chars": round(sum(row["response_chars"] for row in valid) / max(len(valid), 1), 2),
        "tounsi_rate": round(sum(row["is_tounsi"] for row in valid) / max(len(valid), 1) * 100, 2),
        "chinese_rate": round(sum(row["has_chinese"] for row in valid) / max(len(valid), 1) * 100, 2),
        "keyword_pass_rate": round(sum(row["keyword_pass"] for row in valid) / max(len(valid), 1) * 100, 2),
        "forbidden_pass_rate": round(sum(row["forbidden_pass"] for row in valid) / max(len(valid), 1) * 100, 2),
        "avg_fusha_markers": round(sum(row["fusha_count"] for row in valid) / max(len(valid), 1), 2),
        "avg_latency_ms": round(sum(row["latency_ms"] for row in valid) / max(len(valid), 1), 2),
        "intent_breakdown": _intent_breakdown(valid),
        "intent_confusion": _intent_confusion(valid),
        "details": details,
    }
    return summary


def _markdown_report(report: dict[str, Any]) -> str:
    validation_summary = report.get("validation_summary", {}).get("summary", {})
    data_summary = report.get("data_summary", {}).get("processed_data", {})
    inference = report.get("inference_summary", {})
    lines = [
        "# Evaluation Report",
        "",
        "## Validation Summary",
        "",
    ]
    for key, value in validation_summary.items():
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
        "## Data Summary",
        "",
        ]
    )
    for split_name, stats in data_summary.items():
        lines.append(f"### {split_name}")
        for key, value in stats.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    lines.extend(
        [
            "## Inference Summary",
            "",
        ]
    )
    for key, value in inference.items():
        if key in {"details", "intent_breakdown", "intent_confusion"}:
            continue
        lines.append(f"- {key}: {value}")

    if isinstance(inference.get("intent_breakdown"), dict):
        lines.extend(["", "### Per-Intent Breakdown", ""])
        for intent_name, metrics in inference["intent_breakdown"].items():
            lines.append(f"- {intent_name}: {metrics}")

    if isinstance(inference.get("intent_confusion"), dict):
        lines.extend(["", "### Intent Confusion", ""])
        for expected_intent, predicted in inference["intent_confusion"].items():
            lines.append(f"- {expected_intent}: {predicted}")

    lines.extend(
        [
            "",
            "## Recommendations",
            "",
            "- Track intent accuracy, slot F1, tool accuracy and human review accuracy together.",
            "- Use ROUGE-L, BLEU and chrF only for cases with stable reference responses.",
            "- Keep separate eval suites for collect mode and autonomous mode.",
            "- Review error cases in the JSON report before promoting a new production adapter.",
            "",
        ]
    )
    return "\n".join(lines)


def run_evaluation(
    *,
    model_variant: str = "prod",
    runtime_mode: str = "autonomous",
    cases_path: Path | None = None,
    max_cases: int | None = None,
) -> dict[str, Any]:
    report = {
        "generated_at": time.time(),
        "validation_summary": validate_domain_assets(write_report=True),
        "data_summary": evaluate_processed_data(),
        "inference_summary": evaluate_inference(
            model_variant=model_variant,
            runtime_mode=runtime_mode,
            cases_path=cases_path,
            max_cases=max_cases,
        ),
    }

    json_path = REPORTS_DIR / "eval_full.json"
    md_path = REPORTS_DIR / "eval_full.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    logger.info("Evaluation complete -> %s", json_path)
    return report
