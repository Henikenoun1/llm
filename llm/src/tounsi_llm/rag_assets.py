"""
Shared loaders for official business RAG assets.

This module centralizes access to the curated `data/rag` knowledge so the
runtime, training augmentation, and validation layers all use the same source
of truth.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from .config import RAG_DIR, logger


_DELIVERY_CACHE: list[dict[str, Any]] | None = None
_LENS_CACHE: list[dict[str, Any]] | None = None
_RAG_TEXT_CACHE: list[str] | None = None

_TIME_SLOT_RE = re.compile(r"\b(\d{1,2})[:h](\d{2})\b", re.IGNORECASE)


def normalize_lookup(value: str | None) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9\u0600-\u06FF]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _read_text_with_fallback(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def iter_official_rag_files(*, suffixes: set[str] | None = None) -> list[Path]:
    suffixes = suffixes or {".jsonl", ".md", ".txt"}
    if not RAG_DIR.exists():
        return []
    return sorted(
        path
        for path in RAG_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping invalid RAG JSONL line in %s", path)
    return rows


def time_slot_to_minutes(value: str | None) -> int | None:
    raw = str(value or "").strip().lower().replace("h", ":")
    if not raw:
        return None
    match = _TIME_SLOT_RE.search(raw)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour > 23 or minute > 59:
        return None
    return hour * 60 + minute


def next_time_slot_after(slots: list[str], requested_slot: str | None, *, inclusive: bool = True) -> str | None:
    clean_slots = [str(slot).strip() for slot in slots if str(slot).strip()]
    if not clean_slots:
        return None
    requested_minutes = time_slot_to_minutes(requested_slot)
    if requested_minutes is None:
        return clean_slots[0]

    scored: list[tuple[int, str]] = []
    for slot in clean_slots:
        minutes = time_slot_to_minutes(slot)
        if minutes is None:
            continue
        if inclusive and minutes >= requested_minutes:
            scored.append((minutes, slot))
        elif not inclusive and minutes > requested_minutes:
            scored.append((minutes, slot))

    if scored:
        scored.sort(key=lambda item: item[0])
        return scored[0][1]
    return clean_slots[0]


def load_delivery_rag_entries() -> list[dict[str, Any]]:
    global _DELIVERY_CACHE
    if _DELIVERY_CACHE is not None:
        return [dict(entry) for entry in _DELIVERY_CACHE]

    seen: set[str] = set()
    entries: list[dict[str, Any]] = []
    for path in iter_official_rag_files(suffixes={".jsonl"}):
        for row in _read_jsonl(path):
            metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
            row_type = str(row.get("type") or metadata.get("type") or "").strip().lower()
            if row_type not in {"fiche_agence", "fiche_secteur", "agence", "secteur"}:
                continue

            slots = metadata.get("tous_creneaux", [])
            if isinstance(slots, str):
                slots = [slots]
            if not isinstance(slots, list):
                slots = []
            slots = [str(item).strip() for item in slots if str(item).strip()]

            entry = {
                "id": row.get("id"),
                "entry_type": row_type,
                "agence": str(metadata.get("agence", "")).strip(),
                "secteur": str(metadata.get("secteur", "")).strip(),
                "nb_livraisons_jour": metadata.get("nb_livraisons_jour"),
                "premier_creneau": metadata.get("premier_creneau") or (slots[0] if slots else None),
                "tous_creneaux": slots,
                "frequence_continue": bool(metadata.get("frequence_continue", False)),
                "text": str(row.get("text", "")).strip(),
                "source": str(path),
            }
            if not entry["agence"] and not entry["secteur"]:
                continue

            dedup_key = "|".join(
                [
                    normalize_lookup(entry["agence"]),
                    normalize_lookup(entry["secteur"]),
                    ",".join(entry["tous_creneaux"]),
                ]
            )
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            entries.append(entry)

    _DELIVERY_CACHE = entries
    return [dict(entry) for entry in entries]


def load_lens_rag_entries() -> list[dict[str, Any]]:
    global _LENS_CACHE
    if _LENS_CACHE is not None:
        return [dict(entry) for entry in _LENS_CACHE]

    seen: set[str] = set()
    entries: list[dict[str, Any]] = []
    for path in iter_official_rag_files(suffixes={".jsonl"}):
        for row in _read_jsonl(path):
            metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
            row_type = str(row.get("type") or metadata.get("type") or "").strip().lower()
            if row_type not in {"fiche_verre", "verre"}:
                continue

            entry = {
                "id": row.get("id"),
                "code": str(metadata.get("code", "")).strip().upper(),
                "name": str(metadata.get("nom", "")).strip(),
                "brand": str(metadata.get("marque", "")).strip(),
                "geometry": str(metadata.get("geometrie", "")).strip(),
                "material": str(metadata.get("matiere", "")).strip(),
                "photochromic": str(metadata.get("photochromique", "")).strip(),
                "diameter": str(metadata.get("diametre", "")).strip(),
                "text": str(row.get("text", "")).strip(),
                "source": str(path),
            }
            if not entry["code"] and not entry["name"]:
                continue

            dedup_key = "|".join(
                [
                    normalize_lookup(entry["code"]),
                    normalize_lookup(entry["name"]),
                    normalize_lookup(entry["diameter"]),
                ]
            )
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            entries.append(entry)

    _LENS_CACHE = entries
    return [dict(entry) for entry in entries]


def load_official_rag_texts() -> list[str]:
    global _RAG_TEXT_CACHE
    if _RAG_TEXT_CACHE is not None:
        return list(_RAG_TEXT_CACHE)

    texts: list[str] = []
    seen: set[str] = set()

    for entry in load_delivery_rag_entries():
        text = str(entry.get("text", "")).strip()
        if text and text not in seen:
            seen.add(text)
            texts.append(text)

    for entry in load_lens_rag_entries():
        text = str(entry.get("text", "")).strip()
        if text and text not in seen:
            seen.add(text)
            texts.append(text)

    for path in iter_official_rag_files(suffixes={".md", ".txt"}):
        text = _read_text_with_fallback(path).strip()
        if text and text not in seen:
            seen.add(text)
            texts.append(text)

    _RAG_TEXT_CACHE = texts
    return list(texts)
