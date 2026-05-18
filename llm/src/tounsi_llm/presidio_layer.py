# Presidio privacy layer (middleware)
# This file is a standalone layer for PII detection/anonymization using Presidio
# Usage: presidio_filter(text) -> anonymized_text, entities

import re
from typing import Tuple, List, Dict

try:
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine
except ImportError:
    AnalyzerEngine = None
    AnonymizerEngine = None
    Pattern = None
    PatternRecognizer = None

analyzer = AnalyzerEngine() if AnalyzerEngine else None
anonymizer = AnonymizerEngine() if AnonymizerEngine else None

_FALLBACK_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "PHONE_NUMBER",
        re.compile(r"\+?216[\s\-]?\d{2}[\s\-]?\d{3}[\s\-]?\d{3}")
    ),
    (
        "EMAIL_ADDRESS",
        re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
    ),
    (
        "ORDER_ID",
        re.compile(r"\b(?:ORD|CMD|IMPORT)[-_]?[A-Z0-9]{3,}\b|\bimport[-_ ]?\d{3,8}\b", re.IGNORECASE)
    ),
    (
        "CUSTOMER_ID",
        re.compile(r"\b(?:CLI|CLT|CLIENT|CUST)[-_]?\s*0*(\d{3,6})\b", re.IGNORECASE)
    ),
]


def _register_custom_recognizers() -> None:
    if not analyzer or not PatternRecognizer or not Pattern:
        return

    recognizers: list[PatternRecognizer] = []
    for lang in ("en", "fr"):
        recognizers.extend(
            [
                PatternRecognizer(
                    supported_entity="PHONE_NUMBER",
                    patterns=[
                        Pattern(
                            "tn_phone",
                            r"\+?216[\s\-]?\d{2}[\s\-]?\d{3}[\s\-]?\d{3}",
                            0.6,
                        )
                    ],
                    context=["tel", "telephone", "phone", "numero", "num"],
                    supported_language=lang,
                ),
                PatternRecognizer(
                    supported_entity="ORDER_ID",
                    patterns=[
                        Pattern(
                            "order_id",
                            r"\b(?:ORD|CMD|IMPORT)[-_]?[A-Z0-9]{3,}\b|\bimport[-_ ]?\d{3,8}\b",
                            0.55,
                        )
                    ],
                    context=["commande", "order", "cmd", "ord", "import"],
                    supported_language=lang,
                ),
                PatternRecognizer(
                    supported_entity="CUSTOMER_ID",
                    patterns=[
                        Pattern(
                            "customer_id",
                            r"\b(?:CLI|CLT|CLIENT|CUST)[-_]?\s*0*(\d{3,6})\b",
                            0.5,
                        )
                    ],
                    context=["client", "customer", "cli", "clt"],
                    supported_language=lang,
                ),
            ]
        )

    for recognizer in recognizers:
        try:
            analyzer.registry.add_recognizer(recognizer)
        except Exception:
            continue


def _fallback_regex_anonymize(text: str) -> Tuple[str, List[Dict]]:
    if not text:
        return text, []

    spans: list[tuple[int, int, str]] = []
    for entity_type, pattern in _FALLBACK_PATTERNS:
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end(), entity_type))

    if not spans:
        return text, []

    anonymized = text
    for start, end, entity_type in sorted(spans, key=lambda item: item[0], reverse=True):
        anonymized = f"{anonymized[:start]}<{entity_type}>{anonymized[end:]}"

    entities = [
        {"entity_type": entity_type, "start": start, "end": end, "score": 0.5}
        for start, end, entity_type in sorted(spans, key=lambda item: item[0])
    ]
    return anonymized, entities


def _safe_analyze(text: str, language: str) -> list:
    if not analyzer:
        return []
    try:
        return analyzer.analyze(text=text, language=language)
    except Exception:
        if language != "en":
            try:
                return analyzer.analyze(text=text, language="en")
            except Exception:
                return []
        return []


_register_custom_recognizers()


def presidio_filter(text: str, language: str = "en") -> Tuple[str, List[Dict]]:
    """
    Analyze and anonymize sensitive entities in the text using Presidio.
    Returns (anonymized_text, entities_list)
    """
    if not analyzer or not anonymizer:
        # Presidio not installed, use regex fallback
        return _fallback_regex_anonymize(text)
    results = _safe_analyze(text=text, language=language)
    if not results:
        return text, []
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text, [r.to_dict() for r in results]

# Example usage:
# anonymized, entities = presidio_filter("My phone is 123-456-7890 and my name is Ahmed.")
# print(anonymized, entities)
