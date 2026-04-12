"""
Sync external RAG JSONL files into the project workspace without altering content.

This script performs byte-level copies only. It does not parse or rewrite JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_SOURCE_ROOT = Path(os.getenv("OFFICIAL_RAG_SOURCE_ROOT", "external_rag_source"))
DEFAULT_TARGET_ROOT = Path("data/rag/external")
DEFAULT_MANIFEST_PATH = Path("reports/rag_external_sync_manifest.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_jsonl_files(source_root: Path) -> list[Path]:
    return sorted(path for path in source_root.rglob("*.jsonl") if path.is_file())


def sync_external_rag(
    source_root: Path,
    target_root: Path,
    manifest_path: Path,
    overwrite: bool,
) -> dict:
    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    target_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    copied: list[dict] = []
    skipped: list[dict] = []

    for source_file in _collect_jsonl_files(source_root):
        relative = source_file.relative_to(source_root)
        target_file = target_root / relative
        target_file.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "source": str(source_file.resolve()),
            "target": str(target_file.resolve()),
            "relative": str(relative).replace("\\", "/"),
            "size_bytes": source_file.stat().st_size,
            "sha256": _sha256(source_file),
        }

        if target_file.exists() and not overwrite:
            skipped.append(entry)
            continue

        shutil.copy2(source_file, target_file)
        copied.append(entry)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root.resolve()),
        "target_root": str(target_root.resolve()),
        "overwrite": overwrite,
        "copied_count": len(copied),
        "skipped_count": len(skipped),
        "copied": copied,
        "skipped": skipped,
    }
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync external RAG JSONL files (byte-preserving copy).")
    parser.add_argument(
        "--source-root",
        default=str(DEFAULT_SOURCE_ROOT),
        help="Root directory that contains the official RAG JSONL files. Can also be set with OFFICIAL_RAG_SOURCE_ROOT.",
    )
    parser.add_argument("--target-root", default=str(DEFAULT_TARGET_ROOT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite files if they already exist.")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    target_root = Path(args.target_root)
    manifest_path = Path(args.manifest)

    try:
        report = sync_external_rag(
            source_root=source_root,
            target_root=target_root,
            manifest_path=manifest_path,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"[sync_external_rag] error: {exc}")
        return 1

    print(
        "[sync_external_rag] done "
        f"copied={report['copied_count']} skipped={report['skipped_count']} "
        f"manifest={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
