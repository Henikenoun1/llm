"""
Dataset preparation utilities for:
  1. dataset download
  2. self-supervised language adaptation
  3. supervised dialogue fine-tuning
  4. DPO preference alignment
"""
from __future__ import annotations

import json
import random
import re
import shutil
from pathlib import Path
from typing import Any

from .config import (
    DOMAIN_CFG,
    HISTORY_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SEED,
    logger,
    resolve_project_path,
)


FUSHA_MARKERS = [
    "إليك",
    "يمكنني مساعدتك",
    "بالتأكيد",
    "يرجى",
    "يسعدني",
    "تفضل",
    "عذرا",
    "لا تتردد",
    "كيف يمكنني",
    "بكل سرور",
    "هل تريد",
    "هل ترغب",
]

MOROCCAN_MARKERS = [
    "ديال",
    "ديالي",
    "كاين",
    "كاينة",
    "مزيان",
    "دابا",
    "فاش",
    "واخا",
    "بزاف",
    "هادا",
    "هادي",
]

TOUNSI_MARKERS = [
    "متاع",
    "فمّا",
    "فما",
    "باهي",
    "برشا",
    "قداش",
    "وين",
    "وقتاش",
    "شنوة",
    "نحب",
    "نجم",
    "عسلامة",
    "يعيشك",
]

_CHINESE_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


SYSTEM_PROMPT = DOMAIN_CFG.get(
    "system_prompt",
    "إنت agent تخدم في مركز نداء. جاوب بطبيعة وبلهجة تونسية كي يلزم.",
)
SYSTEM_GENERAL = "تحكي بالتونسي ديما. جاوب بطريقة طبيعية وودودة."
_KNOWN_SYSTEM_PROMPTS = {SYSTEM_PROMPT, SYSTEM_GENERAL}
MEMORY_CFG = DOMAIN_CFG.get("memory", {})
APPROVED_LEARNING_PATH = resolve_project_path(
    MEMORY_CFG.get("learning_buffer_path", HISTORY_DIR / "learning_buffer.jsonl")
)
APPROVED_FEEDBACK_DPO_PATH = resolve_project_path(
    MEMORY_CFG.get("approved_dpo_feedback_path", HISTORY_DIR / "feedback_dpo.jsonl")
)


def _save_jsonl(ds, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in ds:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def download_configured_datasets(
    cache_dir: Path | None = None,
    dataset_specs: list[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    cache = cache_dir or RAW_DATA_DIR
    cache.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, Path] = {}
    specs = dataset_specs or DOMAIN_CFG.get("datasets", [])

    for spec in specs:
        name = spec.get("name")
        if not name:
            continue

        output_path = cache / spec.get("output_file", f"{name}.jsonl")
        if output_path.exists():
            downloaded[name] = output_path
            logger.info("Dataset already cached: %s", output_path)
            continue

        if spec.get("local_path"):
            source = resolve_project_path(spec["local_path"])
            if source.exists():
                shutil.copy2(source, output_path)
                downloaded[name] = output_path
                logger.info("Copied local dataset %s -> %s", source, output_path)
            else:
                logger.warning("Configured local dataset not found: %s", source)
            continue

        if spec.get("hf_dataset"):
            try:
                import datasets as hfds

                logger.info("Downloading %s [%s]", spec["hf_dataset"], spec.get("split", "train"))
                ds = hfds.load_dataset(spec["hf_dataset"], split=spec.get("split", "train"))
                _save_jsonl(ds, output_path)
                downloaded[name] = output_path
                logger.info("  -> Saved %d rows to %s", len(ds), output_path)
            except Exception as exc:
                logger.warning("Could not download %s: %s", spec["hf_dataset"], exc)
            continue

        logger.warning("Unsupported dataset spec: %s", spec)

    return downloaded


def download_tounsi_datasets(cache_dir: Path | None = None) -> dict[str, Path]:
    return download_configured_datasets(cache_dir=cache_dir)


def is_moroccan_or_algerian(text: str) -> bool:
    if not text:
        return False
    moroccan_score = sum(1 for marker in MOROCCAN_MARKERS if marker in text)
    tounsi_score = sum(1 for marker in TOUNSI_MARKERS if marker in text)
    return moroccan_score > 0 and tounsi_score <= moroccan_score


def is_clean_tounsi(text: str, max_fusha: int = 1) -> bool:
    if not text or len(text.strip()) < 5:
        return False
    if _CHINESE_RE.search(text):
        return False
    if sum(1 for marker in FUSHA_MARKERS if marker in text) >= max_fusha:
        return False
    if is_moroccan_or_algerian(text):
        return False
    return True


def compute_quality_stats(texts: list[str]) -> dict[str, Any]:
    total = len(texts)
    if total == 0:
        return {"total": 0}
    clean = sum(1 for text in texts if is_clean_tounsi(text))
    chinese = sum(1 for text in texts if _CHINESE_RE.search(text or ""))
    marker_counts = {
        marker: sum(1 for text in texts if marker in (text or ""))
        for marker in FUSHA_MARKERS
    }
    marker_counts = {key: value for key, value in marker_counts.items() if value > 0}
    return {
        "total": total,
        "clean_tounsi": clean,
        "clean_pct": round(clean / total * 100, 1),
        "has_chinese": chinese,
        "top_fusha_markers": dict(sorted(marker_counts.items(), key=lambda item: -item[1])[:10]),
    }


def format_sft_conversation(messages: list[dict[str, Any]], system: str = SYSTEM_PROMPT) -> list[dict[str, str]]:
    existing_system = None
    cleaned: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", message.get("text", ""))
        if not content:
            continue
        if role == "system":
            existing_system = content
            continue
        cleaned.append({"role": role, "content": content})

    chosen_system = existing_system if existing_system in _KNOWN_SYSTEM_PROMPTS else system
    return [{"role": "system", "content": chosen_system}] + cleaned


def _extract_conversation(row: dict[str, Any]) -> list[dict[str, str]] | None:
    if "messages" in row and isinstance(row["messages"], list) and len(row["messages"]) >= 2:
        return format_sft_conversation(row["messages"])

    instruction = row.get("instruction") or row.get("input") or row.get("question") or row.get("prompt")
    output = row.get("output") or row.get("response") or row.get("answer") or row.get("completion")
    if instruction and output:
        return format_sft_conversation(
            [
                {"role": "user", "content": str(instruction)},
                {"role": "assistant", "content": str(output)},
            ]
        )
    return None


def _extract_dpo_pair(row: dict[str, Any]) -> dict[str, str] | None:
    prompt = row.get("prompt", "")
    chosen = row.get("chosen", "")
    rejected = row.get("rejected", "")

    if isinstance(prompt, list):
        prompt = " ".join(item.get("content", "") for item in prompt if isinstance(item, dict))
    if isinstance(chosen, list):
        chosen = " ".join(item.get("content", "") for item in chosen if isinstance(item, dict))
    if isinstance(rejected, list):
        rejected = " ".join(item.get("content", "") for item in rejected if isinstance(item, dict))

    if prompt and chosen and rejected:
        return {"prompt": str(prompt), "chosen": str(chosen), "rejected": str(rejected)}
    return None


def _extract_text_candidates(row: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    if "messages" in row and isinstance(row["messages"], list):
        for message in row["messages"]:
            if not isinstance(message, dict):
                continue
            text = message.get("content") or message.get("text")
            if text:
                texts.append(str(text))
    else:
        for key in ["instruction", "input", "question", "prompt", "output", "response", "answer", "completion", "text"]:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value)
    return texts


def _filtered_conversations(raw_paths: dict[str, Path]) -> list[list[dict[str, str]]]:
    all_conversations: list[list[dict[str, str]]] = []
    for name, path in raw_paths.items():
        before = len(all_conversations)
        rows = _load_jsonl(path)
        for row in rows:
            conv = _extract_conversation(row)
            if conv:
                all_conversations.append(conv)
        logger.info("Loaded %d conversations from %s", len(all_conversations) - before, name)

    custom_path = PROCESSED_DATA_DIR / "custom_tounsi_conversations.jsonl"
    if custom_path.exists():
        for row in _load_jsonl(custom_path):
            conv = _extract_conversation(row)
            if conv:
                all_conversations.extend([conv] * 3)

    enrichment_path = RAW_DATA_DIR / "enrichment" / "enriched_sft_conversations.jsonl"
    if enrichment_path.exists():
        for row in _load_jsonl(enrichment_path):
            conv = _extract_conversation(row)
            if conv:
                all_conversations.append(conv)

    if APPROVED_LEARNING_PATH.exists():
        for row in _load_jsonl(APPROVED_LEARNING_PATH):
            conv = _extract_conversation(row)
            if conv:
                all_conversations.append(conv)

    clean_conversations: list[list[dict[str, str]]] = []
    for conv in all_conversations:
        assistant_texts = [message["content"] for message in conv if message["role"] == "assistant"]
        if not assistant_texts:
            continue
        if any(len(text.strip()) < 10 for text in assistant_texts):
            continue
        if any(not is_clean_tounsi(text, max_fusha=2) for text in assistant_texts):
            continue
        clean_conversations.append(conv)

    return clean_conversations


def prepare_self_supervised_data(
    raw_paths: dict[str, Path],
    output_dir: Path | None = None,
    max_texts: int | None = None,
) -> dict[str, Path]:
    output_dir = output_dir or PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    texts: list[str] = []
    for path in raw_paths.values():
        for row in _load_jsonl(path):
            for text in _extract_text_candidates(row):
                if len(text.strip()) < 20:
                    continue
                if "```" in text:
                    continue
                if is_clean_tounsi(text, max_fusha=3):
                    texts.append(text.strip())

    enrichment_all = RAW_DATA_DIR / "enrichment" / "all_clean_tounsi.jsonl"
    if enrichment_all.exists():
        for row in _load_jsonl(enrichment_all):
            for text in _extract_text_candidates(row):
                if len(text.strip()) >= 20:
                    texts.append(text.strip())

    if max_texts and len(texts) > max_texts:
        random.seed(SEED)
        texts = random.sample(texts, max_texts)

    random.seed(SEED)
    random.shuffle(texts)
    split_at = max(10, int(len(texts) * 0.95))
    train_texts = texts[:split_at]
    val_texts = texts[split_at:]

    paths = {
        "train": output_dir / "self_sup_train.jsonl",
        "val": output_dir / "self_sup_val.jsonl",
    }
    for split_name, split_texts in [("train", train_texts), ("val", val_texts)]:
        with open(paths[split_name], "w", encoding="utf-8") as handle:
            for text in split_texts:
                handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        logger.info("Self-supervised %s: %d texts -> %s", split_name, len(split_texts), paths[split_name])
    return paths


def prepare_sft_data(
    raw_paths: dict[str, Path],
    output_dir: Path | None = None,
    max_samples: int | None = None,
) -> dict[str, Path]:
    output_dir = output_dir or PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_conversations = _filtered_conversations(raw_paths)
    if max_samples and len(clean_conversations) > max_samples:
        random.seed(SEED)
        clean_conversations = random.sample(clean_conversations, max_samples)

    random.seed(SEED)
    random.shuffle(clean_conversations)

    n = len(clean_conversations)
    if n <= 2:
        train = clean_conversations
        val = []
        test = []
    else:
        n_val = max(1, int(n * 0.05))
        n_test = max(1, int(n * 0.05))
        while n - n_val - n_test < 1 and (n_val > 1 or n_test > 1):
            if n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1

        train = clean_conversations[: max(1, n - n_val - n_test)]
        val = clean_conversations[len(train) : len(train) + n_val]
        test = clean_conversations[len(train) + len(val) :]

    if n > 2 and not val:
        val = test[:1]
        test = test[1:]

    paths = {}
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = output_dir / f"sft_{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for conv in split_data:
                handle.write(json.dumps({"messages": conv}, ensure_ascii=False) + "\n")
        paths[split_name] = path
        logger.info("SFT %s: %d conversations -> %s", split_name, len(split_data), path)
    return paths


def prepare_dpo_data(
    raw_paths: dict[str, Path],
    output_dir: Path | None = None,
) -> dict[str, Path]:
    output_dir = output_dir or PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[dict[str, str]] = []
    dpo_path = raw_paths.get("dpo")
    if dpo_path and dpo_path.exists():
        for row in _load_jsonl(dpo_path):
            pair = _extract_dpo_pair(row)
            if not pair:
                continue
            if len(pair["chosen"].strip()) < 15 or len(pair["rejected"].strip()) < 15:
                continue
            if is_moroccan_or_algerian(pair["prompt"] + " " + pair["chosen"] + " " + pair["rejected"]):
                continue
            pairs.append(pair)

    if APPROVED_FEEDBACK_DPO_PATH.exists():
        for row in _load_jsonl(APPROVED_FEEDBACK_DPO_PATH):
            pair = _extract_dpo_pair(row)
            if pair:
                pairs.append(pair)

    if not pairs:
        logger.warning("No DPO pairs found after filtering.")
        return {}

    random.seed(SEED)
    random.shuffle(pairs)
    if len(pairs) <= 2:
        train = pairs
        val = []
    else:
        n_val = max(1, int(len(pairs) * 0.1))
        if len(pairs) - n_val < 1:
            n_val = 1
        train = pairs[:-n_val]
        val = pairs[-n_val:]

    paths = {}
    for split_name, split_data in [("train", train), ("val", val)]:
        path = output_dir / f"dpo_{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for pair in split_data:
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
        paths[split_name] = path
        logger.info("DPO %s: %d pairs -> %s", split_name, len(split_data), path)
    return paths


def prepare_all_data(
    raw_paths: dict[str, Path],
    output_dir: Path | None = None,
    max_sft_samples: int | None = None,
    max_self_sup_texts: int | None = None,
) -> dict[str, dict[str, Path]]:
    return {
        "self_sup": prepare_self_supervised_data(raw_paths, output_dir=output_dir, max_texts=max_self_sup_texts),
        "sft": prepare_sft_data(raw_paths, output_dir=output_dir, max_samples=max_sft_samples),
        "dpo": prepare_dpo_data(raw_paths, output_dir=output_dir),
    }
