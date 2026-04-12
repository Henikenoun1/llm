"""
Raw dataset audit utilities.

Produces machine-readable and markdown summaries in reports/.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from .config import REPORTS_DIR, logger
from .data_sources import (
    dataset_spec_map,
    detect_script,
    extract_label,
    extract_text_candidates,
    load_jsonl_rows,
    normalize_for_dedup,
)


def _dataset_stats(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    spec = dataset_spec_map().get(name, {})
    texts: list[str] = []
    labels: dict[str, int] = {}
    scripts: dict[str, int] = {}

    for row in rows:
        row_texts = extract_text_candidates(row, spec=spec)
        texts.extend(row_texts)

        label = extract_label(row, spec=spec)
        if label:
            labels[label] = labels.get(label, 0) + 1

        if row_texts:
            script = detect_script(row_texts[0])
            scripts[script] = scripts.get(script, 0) + 1

    lengths = [len(text.split()) for text in texts if text]
    unique_texts = {normalize_for_dedup(text) for text in texts if text}

    return {
        "rows": len(rows),
        "text_records": len(texts),
        "unique_texts": len(unique_texts),
        "avg_tokens": round(mean(lengths), 2) if lengths else 0.0,
        "scripts": scripts,
        "labels": labels,
        "license": spec.get("license"),
        "roles": spec.get("roles", []),
        "type": spec.get("type"),
    }


def audit_raw_datasets(raw_paths: dict[str, Path]) -> dict[str, str]:
    summary: dict[str, Any] = {
        "datasets": {},
        "totals": {
            "rows": 0,
            "text_records": 0,
            "datasets": len(raw_paths),
        },
    }

    for name, path in raw_paths.items():
        rows = load_jsonl_rows(path)
        stats = _dataset_stats(name, rows)
        summary["datasets"][name] = stats
        summary["totals"]["rows"] += stats["rows"]
        summary["totals"]["text_records"] += stats["text_records"]
        logger.info("Audited %s: %s", name, stats)

    json_report = REPORTS_DIR / "data_audit.json"
    md_report = REPORTS_DIR / "data_audit.md"
    json_report.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Data Audit",
        "",
        f"- datasets: {summary['totals']['datasets']}",
        f"- total rows: {summary['totals']['rows']}",
        f"- total text records: {summary['totals']['text_records']}",
        "",
    ]

    for name, stats in summary["datasets"].items():
        lines.extend(
            [
                f"## {name}",
                f"- rows: {stats['rows']}",
                f"- text records: {stats['text_records']}",
                f"- unique texts: {stats['unique_texts']}",
                f"- avg tokens: {stats['avg_tokens']}",
                f"- scripts: {json.dumps(stats['scripts'], ensure_ascii=False)}",
                f"- labels: {json.dumps(stats['labels'], ensure_ascii=False)}",
                f"- license: {stats.get('license')}",
                f"- type: {stats.get('type')}",
                f"- roles: {json.dumps(stats.get('roles', []), ensure_ascii=False)}",
                "",
            ]
        )

    md_report.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Data audit reports written: %s | %s", json_report, md_report)

    return {
        "json_report": str(json_report),
        "markdown_report": str(md_report),
    }
