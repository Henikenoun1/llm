"""
Live admin corrections applied at inference time without weight retraining.
"""
from __future__ import annotations

import json
import time
import unicodedata
from pathlib import Path
from typing import Any

from .config import DOMAIN_CFG, HISTORY_DIR, resolve_project_path
from .storage import get_database_backend


def _normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower()
    cleaned = []
    for char in text:
        if char.isalnum() or char.isspace() or ("\u0600" <= char <= "\u06FF"):
            cleaned.append(char)
        else:
            cleaned.append(" ")
    return " ".join("".join(cleaned).split())


def _token_overlap(a: str, b: str) -> float:
    a_tokens = set(_normalize(a).split())
    b_tokens = set(_normalize(b).split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(len(a_tokens), len(b_tokens))


class LiveCorrectionStore:
    def __init__(self) -> None:
        memory_cfg = DOMAIN_CFG.get("memory", {})
        self.path = resolve_project_path(
            memory_cfg.get("admin_corrections_path", HISTORY_DIR / "admin_corrections.jsonl")
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = get_database_backend()
        self._entries: list[dict[str, Any]] = []
        self.reload()

    def reload(self) -> None:
        entries: list[dict[str, Any]] = []
        if self.path.exists():
            with open(self.path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        for row in self.db.load_admin_corrections():
            entries.append(row)

        deduped: list[dict[str, Any]] = []
        seen = set()
        for entry in entries:
            key = (
                entry.get("normalized_pattern"),
                entry.get("intent"),
                entry.get("runtime_mode"),
                entry.get("corrected_response"),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(entry)
        self._entries = deduped

    def add_correction(
        self,
        *,
        pattern_text: str,
        corrected_response: str,
        intent: str | None = None,
        slots: dict[str, Any] | None = None,
        runtime_mode: str | None = None,
        action: str = "replace",
        reviewer_id: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        entry = {
            "timestamp": time.time(),
            "pattern_text": pattern_text,
            "normalized_pattern": _normalize(pattern_text),
            "intent": intent,
            "slots": slots or {},
            "runtime_mode": runtime_mode,
            "corrected_response": corrected_response,
            "action": action,
            "reviewer_id": reviewer_id,
            "notes": notes or "",
        }
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.db.record_admin_correction(entry)
        self._entries.append(entry)
        return entry

    def find_best(
        self,
        *,
        user_text: str,
        intent: str,
        slots: dict[str, Any],
        runtime_mode: str,
    ) -> dict[str, Any] | None:
        normalized_user = _normalize(user_text)
        best_entry: dict[str, Any] | None = None
        best_score = 0.0

        for entry in self._entries:
            pattern = str(entry.get("normalized_pattern") or entry.get("pattern_text") or "")
            if not pattern:
                continue

            score = _token_overlap(normalized_user, pattern)
            if pattern == normalized_user:
                score += 2.0
            entry_intent = str(entry.get("intent") or "")
            if entry_intent and entry_intent == intent:
                score += 1.0
            entry_mode = str(entry.get("runtime_mode") or "")
            if entry_mode and entry_mode == runtime_mode:
                score += 0.5

            entry_slots = entry.get("slots", {}) if isinstance(entry.get("slots"), dict) else {}
            if entry_slots:
                matched = 0
                for key, value in entry_slots.items():
                    if slots.get(key) == value:
                        matched += 1
                score += matched * 0.35
                if matched == len(entry_slots):
                    score += 0.4

            if score > best_score:
                best_entry = entry
                best_score = score

        if best_entry is None:
            return None
        if best_score < 1.4:
            return None
        result = dict(best_entry)
        result["score"] = round(best_score, 4)
        return result
