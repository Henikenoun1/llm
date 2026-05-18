# Presidio privacy layer (middleware)
# This file is a standalone layer for PII detection/anonymization using Presidio
# Usage: presidio_filter(text) -> anonymized_text, entities

from typing import Tuple, List, Dict

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
except ImportError:
    AnalyzerEngine = None
    AnonymizerEngine = None

analyzer = AnalyzerEngine() if AnalyzerEngine else None
anonymizer = AnonymizerEngine() if AnonymizerEngine else None


def presidio_filter(text: str, language: str = "en") -> Tuple[str, List[Dict]]:
    """
    Analyze and anonymize sensitive entities in the text using Presidio.
    Returns (anonymized_text, entities_list)
    """
    if not analyzer or not anonymizer:
        # Presidio not installed, return as is
        return text, []
    results = analyzer.analyze(text=text, language=language)
    if not results:
        return text, []
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text, [r.to_dict() for r in results]

# Example usage:
# anonymized, entities = presidio_filter("My phone is 123-456-7890 and my name is Ahmed.")
# print(anonymized, entities)
