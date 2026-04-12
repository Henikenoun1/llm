"""
Import the official Scenario Reco DOCX into the internal RAG corpus.

Usage:
  python scripts/import_scenario_reco.py
  python scripts/import_scenario_reco.py --source "C:/path/Scenario reco.docx"
"""
from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


DEFAULT_SOURCE = Path(r"C:\Users\Heni\Downloads\Scenario reco.docx")
DEFAULT_TARGET = Path("data/rag/internal/scenario_reco_officiel.md")


def extract_docx_paragraphs(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        document_xml = archive.read("word/document.xml")

    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    root = ET.fromstring(document_xml)

    paragraphs: list[str] = []
    for para in root.findall(".//w:p", ns):
        parts = []
        for node in para.findall(".//w:t", ns):
            if node.text:
                parts.append(node.text)
        text = "".join(parts).strip()
        if text:
            text = re.sub(r"\s+", " ", text)
            paragraphs.append(text)
    return paragraphs


def to_markdown(paragraphs: list[str], source: Path) -> str:
    lines = [
        "# Scenario Recommandation Officiel",
        "",
        f"Source: {source}",
        "",
        "Document importe automatiquement depuis le fichier DOCX officiel.",
        "",
    ]
    for idx, paragraph in enumerate(paragraphs, start=1):
        lines.append(f"{idx:03d}. {paragraph}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Import Scenario Reco DOCX into internal RAG markdown")
    parser.add_argument("--source", type=str, default=str(DEFAULT_SOURCE))
    parser.add_argument("--target", type=str, default=str(DEFAULT_TARGET))
    args = parser.parse_args()

    source = Path(args.source)
    target = Path(args.target)

    if not source.exists():
        print(f"[import_scenario_reco] source not found: {source}")
        return 1

    paragraphs = extract_docx_paragraphs(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(to_markdown(paragraphs, source), encoding="utf-8")

    print(f"[import_scenario_reco] paragraphs={len(paragraphs)}")
    print(f"[import_scenario_reco] wrote: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
