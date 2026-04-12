"""
Dataset ingestion utilities with provenance manifest generation.

This module is intentionally self-contained so it can be reused by both
preprocessing and audit steps without touching training logic.
"""
from __future__ import annotations

import json
import re
import shutil
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
from csv import DictReader
from pathlib import Path
from typing import Any

from .config import CACHE_DIR, DOMAIN_CFG, RAW_DATA_DIR, REPORTS_DIR, logger, resolve_project_path


COMMON_TEXT_FIELDS = [
    "text",
    "sentence",
    "utterance",
    "content",
    "comment",
    "tweet",
    "review",
    "transcript",
    "normalized_text",
    "prompt",
    "response",
    "answer",
    "question",
]

COMMON_LABEL_FIELDS = ["label", "class"]

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_ARABIZI_DIGIT_RE = re.compile(r"\b\w*[235679]\w*\b", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+", flags=re.UNICODE)
_HASHTAG_RE = re.compile(r"#(\w+)", flags=re.UNICODE)
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]",
    flags=re.UNICODE,
)
_REPEAT_PUNCT_RE = re.compile(r"([!?.,;:\u061b\u061f])\1{1,}")
_REPEAT_CHAR_RE = re.compile(r"(.)\1{4,}", flags=re.UNICODE)

TOUNSI_MARKERS = {
    "شنوة",
    "شنية",
    "عسلامة",
    "قداش",
    "فما",
    "نجم",
    "برشا",
    "برشة",
    "يعيشك",
    "كيفاش",
    "علاش",
    "توة",
    "توا",
    "موش",
    "خاطر",
    "بالله",
    "ديما",
    "aslema",
    "chnowa",
    "chnia",
    "kifech",
    "kifach",
    "famma",
    "najjem",
    "nejjem",
    "n3awnek",
    "9adech",
    "3lech",
    "tawa",
    "mouch",
    "barsha",
    "barcha",
    "khater",
    "belehi",
    "bilehi",
    "3aychek",
}

NON_TUNSI_MARKERS = {
    "عايز",
    "عاوزه",
    "ازاي",
    "إزاي",
    "دلوقتي",
    "ليه",
    "واش",
    "بزاف",
    "دابا",
    "ديال",
    "هاد",
    "كاين",
    "كاينة",
    "wach",
    "bzaf",
    "daba",
    "dial",
    "kayen",
}


def dataset_specs() -> list[dict[str, Any]]:
    return list(DOMAIN_CFG.get("datasets", []))


def dataset_spec_map() -> dict[str, dict[str, Any]]:
    return {spec.get("name", ""): spec for spec in dataset_specs() if spec.get("name")}


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def normalize_text(text: str) -> str:
    text = safe_text(text)
    text = unicodedata.normalize("NFKC", text)
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(r"\1", text)
    text = _EMOJI_RE.sub(" ", text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    text = text.replace("\u0640", "")
    text = _REPEAT_PUNCT_RE.sub(r"\1", text)
    text = _REPEAT_CHAR_RE.sub(lambda m: m.group(1) * 3, text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_for_dedup(text: str) -> str:
    text = normalize_text(text).lower()
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\u0600-\u06FF]+", " ", text, flags=re.UNICODE)
    return _WHITESPACE_RE.sub(" ", text).strip()


def detect_script(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return "other"
    arabic_count = len(_ARABIC_RE.findall(text))
    latin_count = len(_LATIN_RE.findall(text))
    arabizi_digit = len(_ARABIZI_DIGIT_RE.findall(text))

    if arabic_count and not latin_count:
        return "arabic"
    if latin_count and not arabic_count:
        return "arabizi" if arabizi_digit or latin_count >= 3 else "latin"
    if arabic_count and latin_count:
        return "mixed"
    return "other"


def looks_tunisian(text: str, strict: bool = False) -> bool:
    normalized = normalize_text(text)
    if len(normalized) < 4:
        return False
    lowered = normalized.lower()
    has_tounsi_marker = any(marker in normalized or marker in lowered for marker in TOUNSI_MARKERS)

    if strict:
        has_non_tounsi = any(marker in normalized or marker in lowered for marker in NON_TUNSI_MARKERS)
        if has_non_tounsi and not has_tounsi_marker:
            return False
        return has_tounsi_marker

    if has_tounsi_marker:
        return True
    return detect_script(normalized) in {"arabic", "arabizi", "mixed"}


def _save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _read_csv_like(path: Path, delimiter: str = ",") -> list[dict[str, Any]]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(DictReader(handle, delimiter=delimiter))


def _read_xlsx(path: Path) -> list[dict[str, Any]]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", ns):
                shared_strings.append("".join(node.text or "" for node in item.findall(".//a:t", ns)))

        worksheet = "xl/worksheets/sheet1.xml"
        if worksheet not in archive.namelist():
            raise FileNotFoundError(f"Worksheet not found in {path}")
        root = ET.fromstring(archive.read(worksheet))
        sheet_rows = root.findall(".//a:sheetData/a:row", ns)

        def _cell_value(cell) -> str:
            kind = cell.attrib.get("t")
            value_node = cell.find("a:v", ns)
            if value_node is None:
                inline_node = cell.find("a:is", ns)
                if inline_node is None:
                    return ""
                return "".join(node.text or "" for node in inline_node.findall(".//a:t", ns))
            if kind == "s":
                return shared_strings[int(value_node.text)] if value_node.text is not None else ""
            return value_node.text or ""

        records: list[dict[str, Any]] = []
        headers: list[str] = []
        for idx, row in enumerate(sheet_rows):
            values = [_cell_value(cell) for cell in row.findall("a:c", ns)]
            if idx == 0:
                headers = values
                continue
            item = {headers[i]: values[i] if i < len(values) else "" for i in range(len(headers))}
            if any(str(value).strip() for value in item.values()):
                records.append(item)
        return records


def _convert_local_table_to_jsonl(source: Path, destination: Path) -> None:
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        shutil.copy2(source, destination)
        return
    if suffix == ".csv":
        _save_jsonl(_read_csv_like(source, delimiter=","), destination)
        return
    if suffix == ".tsv":
        _save_jsonl(_read_csv_like(source, delimiter="\t"), destination)
        return
    if suffix in {".xlsx", ".xlsm"}:
        _save_jsonl(_read_xlsx(source), destination)
        return
    raise ValueError(f"Unsupported local dataset format: {source.suffix}")


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _iter_nested_strings(value: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(value, str):
        text = normalize_text(value)
        if text:
            texts.append(text)
    elif isinstance(value, dict):
        for child in value.values():
            texts.extend(_iter_nested_strings(child))
    elif isinstance(value, list):
        for child in value:
            texts.extend(_iter_nested_strings(child))
    return texts


def extract_text_candidates(row: dict[str, Any], spec: dict[str, Any] | None = None) -> list[str]:
    spec = spec or {}
    keys = list(dict.fromkeys([*(spec.get("text_fields", []) or []), *COMMON_TEXT_FIELDS]))
    texts: list[str] = []

    for key in keys:
        texts.extend(_iter_nested_strings(row.get(key)))

    if "messages" in row and isinstance(row["messages"], list):
        for message in row["messages"]:
            if isinstance(message, dict):
                texts.extend(_iter_nested_strings(message.get("content") or message.get("text")))

    if "conversation" in row:
        texts.extend(_iter_nested_strings(row["conversation"]))

    deduped: list[str] = []
    seen: set[str] = set()
    for text in texts:
        if text and text not in seen:
            deduped.append(text)
            seen.add(text)
    return deduped


def extract_label(row: dict[str, Any], spec: dict[str, Any] | None = None) -> str | None:
    spec = spec or {}
    fields = list(dict.fromkeys([*(spec.get("label_fields", []) or []), *COMMON_LABEL_FIELDS]))
    mapping = {str(k).strip().lower(): str(v).strip().lower() for k, v in (spec.get("label_mapping") or {}).items()}

    for field in fields:
        if field not in row:
            continue
        value = row.get(field)
        if value is None:
            continue
        key = str(value).strip().lower()
        mapped = mapping.get(key, key)
        if mapped in {"positive", "negative", "neutral"}:
            return mapped
    return None


def download_configured_datasets(
    cache_dir: Path | None = None,
    dataset_specs_override: list[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    cache = cache_dir or RAW_DATA_DIR
    cache.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    specs = dataset_specs_override or dataset_specs()
    downloaded: dict[str, Path] = {}
    manifest: list[dict[str, Any]] = []

    for spec in specs:
        name = spec.get("name")
        if not name:
            continue

        output_path = cache / spec.get("output_file", f"{name}.jsonl")
        manifest.append(
            {
                "name": name,
                "roles": spec.get("roles", []),
                "type": spec.get("type"),
                "license": spec.get("license"),
                "path": str(output_path),
                "source": spec.get("hf_dataset") or spec.get("url") or spec.get("local_path"),
                "split": spec.get("split", "train"),
            }
        )

        if output_path.exists():
            downloaded[name] = output_path
            logger.info("Dataset already cached: %s", output_path)
            continue

        if spec.get("local_path"):
            source = resolve_project_path(spec["local_path"])
            if not source.exists():
                logger.warning("Configured local dataset not found: %s", source)
                continue
            _convert_local_table_to_jsonl(source, output_path)
            downloaded[name] = output_path
            logger.info("Converted local dataset %s -> %s", source, output_path)
            continue

        if spec.get("hf_dataset"):
            try:
                import datasets as hfds  # type: ignore[import-not-found]

                logger.info("Downloading %s [%s]", spec["hf_dataset"], spec.get("split", "train"))
                ds = hfds.load_dataset(
                    spec["hf_dataset"],
                    spec.get("hf_name"),
                    split=spec.get("split", "train"),
                    cache_dir=str(CACHE_DIR),
                    trust_remote_code=bool(spec.get("trust_remote_code", False)),
                )
                if spec.get("max_rows"):
                    ds = ds.select(range(min(int(spec["max_rows"]), len(ds))))
                rows = []
                for row in ds:
                    item = dict(row)
                    item.setdefault("__source__", name)
                    rows.append(item)
                _save_jsonl(rows, output_path)
                downloaded[name] = output_path
                logger.info("  -> Saved %d rows to %s", len(rows), output_path)
            except Exception as exc:
                logger.warning("Could not download %s: %s", spec["hf_dataset"], exc)
            continue

        if spec.get("url"):
            try:
                import requests

                response = requests.get(spec["url"], timeout=120)
                response.raise_for_status()
                output_path.write_bytes(response.content)
                downloaded[name] = output_path
                logger.info("Downloaded %s -> %s", spec["url"], output_path)
            except Exception as exc:
                logger.warning("Could not download URL %s: %s", spec["url"], exc)
            continue

        logger.warning("Unsupported dataset spec: %s", spec)

    manifest_path = REPORTS_DIR / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Dataset manifest written to %s", manifest_path)
    return downloaded
