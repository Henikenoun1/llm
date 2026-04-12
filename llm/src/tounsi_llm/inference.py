"""
Inference pipeline with:
  - configurable model variants
  - intent/slot extraction
  - external tools
  - vector RAG
  - few-shot reinforcement
  - persistent conversation memory
"""
from __future__ import annotations

import json
import random
import re
import threading
import time
import unicodedata
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .corrections import LiveCorrectionStore
from .config import CFG, CONFIG_DIR, DOMAIN_CFG, FEW_SHOTS_CFG, logger
from .domain_utils import canonicalize_intent, canonicalize_slots, words_to_number
from .memory import ConversationMemoryStore
from .rag import VectorRAGRetriever
from .rag_assets import load_delivery_rag_entries, load_lens_rag_entries
from .tools import ToolRegistry, get_tool_registry


PHONE_RE = re.compile(r"\+?216[\s\-]?\d{2}[\s\-]?\d{3}[\s\-]?\d{3}")
ORDER_RE = re.compile(r"\b(?:ORD|CMD)[-_]?[A-Z0-9]{4,}\b|\bimport[-_ ]?\d{3,8}\b", re.IGNORECASE)
DATE_RE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")
TIME_RE = re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b")
_ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
CUSTOMER_ID_RE = re.compile(r"\b(?:CLI|CLT|CLIENT|CUST)[-_]?\s*0*(\d{3,6})\b", re.IGNORECASE)
STANDALONE_CLIENT_NUMBER_RE = re.compile(r"\b\d{3,6}\b")
INDEX_RE = re.compile(r"\b1\.(?:50|56|60|67)\b")
INDEX_LABEL_RE = re.compile(r"\b(?:index|indice|ind)\s*(1\.(?:50|56|60|67))\b", re.IGNORECASE)
REFERENCE_RE = re.compile(r"\b\d{2}\s?\d{2}\b")
LENS_CODE_CONTEXT_RE = re.compile(
    r"\b(?:code(?:\s+verre)?|réf(?:érence)?|reference)\s*[:#-]?\s*([A-Z0-9]{5,8})\b",
    re.IGNORECASE,
)
BON_NUMBER_RE = re.compile(
    r"\b(?:bon(?:\s+de\s+commande)?|commande|كوموند)\b\s*[:#-]?\s*((?:ORD|CMD|IMPORT)[-_]?[A-Z0-9]{3,}|\d{4,8})\b",
    re.IGNORECASE,
)
SIGNED_NUMBER_RE = re.compile(r"(plus|moins|\+|-)?\s*(\d{1,2}(?:[.,]\d{1,2})?)", re.IGNORECASE)
AXIS_RE = re.compile(r"(?:axe|axis)\s*(\d{1,3})", re.IGNORECASE)
ADDITION_RE = re.compile(r"(?:addition|add)\s*(plus|moins|\+|-)?\s*(\d{1,2}(?:[.,]\d{1,2})?)", re.IGNORECASE)
_DOMAIN_VALUE_CACHE: dict[str, list[str]] | None = None

_ARABIZI_STYLE_MARKERS = {
    "aslema",
    "marhbe",
    "marhba",
    "labes",
    "chnowa",
    "chnia",
    "kifech",
    "kifach",
    "nheb",
    "najjem",
    "nejjem",
    "ma3louma",
    "mouch",
    "barsha",
    "3lech",
    "ya3tik",
    "3aychek",
}

_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
_MODEL_LOCK = threading.Lock()


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _transliterate_franco(text)


def _transliterate_franco(text: str) -> str:
    franco_map = DOMAIN_CFG.get("franco_map", {})
    latin_chars = sum(1 for char in text if char.isascii() and char.isalpha())
    arabic_chars = sum(1 for char in text if "\u0600" <= char <= "\u06FF")
    if latin_chars <= arabic_chars:
        return text

    converted = []
    for word in text.split():
        stripped = word.lower().strip(".,!?؟")
        converted.append(franco_map.get(stripped, word))
    return " ".join(converted)


def _norm_for_matching(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.lower()
    kept_chars = []
    for char in normalized:
        if (
            char.isdigit()
            or ("a" <= char <= "z")
            or ("\u0600" <= char <= "\u06FF")
            or char.isspace()
            or char in {".", ",", "+", "-"}
        ):
            kept_chars.append(char)
        else:
            kept_chars.append(" ")
    return re.sub(r"\s+", " ", "".join(kept_chars)).strip()


def _detect_script_like(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return "other"
    has_arabic = bool(_ARABIC_CHAR_RE.search(normalized))
    has_latin = bool(_LATIN_CHAR_RE.search(normalized))
    if has_arabic and not has_latin:
        return "arabic"
    if has_latin and not has_arabic:
        return "arabizi"
    if has_arabic and has_latin:
        return "mixed"
    return "other"


def _preferred_response_script(user_text: str) -> str:
    detected = _detect_script_like(user_text)
    if detected in {"arabizi", "latin"}:
        return "arabizi"
    if detected == "arabic":
        return "arabic"
    if detected == "mixed":
        lowered = normalize_text(user_text).lower()
        if any(marker in lowered for marker in _ARABIZI_STYLE_MARKERS):
            return "arabizi"
        arabic_count = len(_ARABIC_CHAR_RE.findall(lowered))
        latin_count = len(_LATIN_CHAR_RE.findall(lowered))
        return "arabizi" if latin_count >= arabic_count else "arabic"
    return "arabic"


def _script_instruction(target_script: str) -> str:
    if target_script == "arabizi":
        return (
            "Reply in Tunisian Arabizi (Latin script) with a professional call-center tone. "
            "Keep the answer concise and do not switch to MSA Arabic unless the user asked for it."
        )
    return (
        "جاوب باللهجة التونسية بالحروف العربية وبأسلوب مهني متاع مركز نداء. "
        "خلي الجواب مختصر وواضح."
    )


def _configured_system_prompts() -> dict[str, str]:
    prompts = DOMAIN_CFG.get("system_prompts", {})
    if not isinstance(prompts, dict):
        return {}

    normalized: dict[str, str] = {}
    for key, value in prompts.items():
        text = str(value or "").strip()
        if not text:
            continue
        normalized_key = str(key).strip().lower().replace("-", "_")
        normalized[normalized_key] = text
    return normalized


def _select_prompt_style(user_script: str, target_script: str) -> str:
    if user_script == "mixed":
        return "code_switch"
    if target_script == "arabizi":
        return "arabizi"
    if user_script == "arabic":
        return "arabic"
    return "default"


def _resolve_system_prompt_messages(*, user_script: str, target_script: str, intent: str) -> list[str]:
    configured = _configured_system_prompts()
    prompt_keys = ["default", _select_prompt_style(user_script, target_script)]

    if intent in {
        "create_order",
        "order_creation",
        "order_tracking",
        "delivery_schedule",
        "availability_inquiry",
        "reference_confirmation",
    }:
        prompt_keys.append("business_critical")

    resolved: list[str] = []
    seen: set[str] = set()

    for key in prompt_keys:
        text = str(configured.get(key, "")).strip()
        if text and text not in seen:
            resolved.append(text)
            seen.add(text)

    for fallback in [DOMAIN_CFG.get("system_prompt", ""), DOMAIN_CFG.get("humanized_prompt", "")]:
        text = str(fallback or "").strip()
        if text and text not in seen:
            resolved.append(text)
            seen.add(text)

    return resolved


def _is_script_mismatch(text: str, target_script: str) -> bool:
    normalized = normalize_text(text)
    if len(normalized.split()) < 4:
        return False
    has_arabic = bool(_ARABIC_CHAR_RE.search(normalized))
    has_latin = bool(_LATIN_CHAR_RE.search(normalized))
    if target_script == "arabizi":
        return has_arabic and not has_latin
    return has_latin and not has_arabic


def _build_script_retry_messages(messages: list[dict[str, str]], target_script: str) -> list[dict[str, str]]:
    retry_messages = list(messages)
    retry_instruction = (
        "Your previous answer script does not match the user style. Regenerate strictly in Tunisian Arabizi (Latin letters)."
        if target_script == "arabizi"
        else "المسودة السابقة ما تحترمش سكريبت المستخدم. أعد الجواب باللهجة التونسية بالحروف العربية فقط."
    )
    retry_messages.insert(1, {"role": "system", "content": retry_instruction})
    return retry_messages


def _load_domain_value_lexicon() -> dict[str, list[str]]:
    global _DOMAIN_VALUE_CACHE
    if _DOMAIN_VALUE_CACHE is not None:
        return _DOMAIN_VALUE_CACHE

    values = {
        "product": set(),
        "material": set(),
        "treatment": set(),
        "color": set(),
        "priority": set(),
    }

    def simplify_rag_material(value: str) -> str:
        text = str(value or "").strip()
        text = re.sub(r"\s*\([^)]*\)", "", text).strip(" -/")
        return text

    def allow_value(field: str, value: str) -> bool:
        normalized_value = _norm_for_matching(value)
        if not normalized_value:
            return False
        if field == "material":
            if normalized_value in {"marron", "gris", "blanc", "brun", "bleu", "green"}:
                return False
            if re.fullmatch(r"1\.(50|56|60|67)", normalized_value):
                return False
        return True

    for canonical, aliases in DOMAIN_CFG.get("product_aliases", {}).items():
        values["product"].add(str(canonical))
        for alias in aliases:
            values["product"].add(str(alias))

    for row in load_lens_rag_entries():
        if row.get("name") and allow_value("product", str(row["name"])):
            values["product"].add(str(row["name"]))
        simplified_material = simplify_rag_material(str(row.get("material", "")))
        if simplified_material and allow_value("material", simplified_material):
            values["material"].add(simplified_material)
        if row.get("photochromic") and str(row["photochromic"]).strip().lower() not in {"non", "none", "no"}:
            values["treatment"].add(str(row["photochromic"]).strip())

    path = CONFIG_DIR / "commandes_intents.jsonl"
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                slots = canonicalize_slots(row.get("slots", {}))
                for field in values:
                    if slots.get(field) and allow_value(field, str(slots[field])):
                        values[field].add(str(slots[field]))

    _DOMAIN_VALUE_CACHE = {
        field: sorted((value for value in items if value), key=lambda item: len(_norm_for_matching(item)), reverse=True)
        for field, items in values.items()
    }
    return _DOMAIN_VALUE_CACHE


def _extract_known_lens_code(text: str) -> str | None:
    known_codes = {
        str(row.get("code", "")).strip().upper()
        for row in load_lens_rag_entries()
        if str(row.get("code", "")).strip()
    }
    if not known_codes:
        return None
    for token in re.findall(r"\b[A-Z0-9]{4,8}\b", text.upper()):
        if token in known_codes:
            return token
    return None


def _extract_rag_delivery_slots(text: str) -> dict[str, Any]:
    normalized = _norm_for_matching(text)
    if not normalized:
        return {}

    best_agence = None
    best_agence_len = 0
    best_sector_entry: dict[str, Any] | None = None
    best_sector_len = 0

    for entry in load_delivery_rag_entries():
        agence = str(entry.get("agence", "")).strip()
        secteur = str(entry.get("secteur", "")).strip()
        agence_norm = _norm_for_matching(agence)
        secteur_norm = _norm_for_matching(secteur)

        if agence_norm and agence_norm in normalized and len(agence_norm) > best_agence_len:
            best_agence = agence
            best_agence_len = len(agence_norm)

        if secteur_norm and secteur_norm in normalized and len(secteur_norm) > best_sector_len:
            best_sector_entry = entry
            best_sector_len = len(secteur_norm)

    updates: dict[str, Any] = {}
    if best_sector_entry is not None:
        if best_sector_entry.get("secteur"):
            updates["secteur"] = best_sector_entry["secteur"]
        if best_sector_entry.get("agence"):
            updates["agence"] = best_sector_entry["agence"]
    elif best_agence:
        updates["agence"] = best_agence

    if best_agence and updates.get("agence") and _norm_for_matching(str(updates["agence"])) != _norm_for_matching(best_agence):
        updates["agence"] = best_agence
    elif best_agence and not updates.get("agence"):
        updates["agence"] = best_agence

    return canonicalize_slots(updates)


def _extract_lexicon_value(text: str, field: str) -> str | None:
    lowered = _norm_for_matching(text)
    for candidate in _load_domain_value_lexicon().get(field, []):
        normalized_candidate = _norm_for_matching(candidate)
        if not normalized_candidate:
            continue
        candidate_tokens = [re.escape(token) for token in normalized_candidate.split() if token]
        if not candidate_tokens:
            continue
        pattern = r"\\b" + r"\\s+".join(candidate_tokens) + r"\\b"
        if re.search(pattern, lowered):
            return candidate
    return None


def _normalize_signed_value(sign: str | None, raw_number: str) -> str:
    number = raw_number.replace(",", ".").strip()
    if sign:
        sign = sign.strip().lower()
        if sign in {"moins", "-"} and not number.startswith("-"):
            return f"-{number.lstrip('+')}"
        if sign in {"plus", "+"} and not number.startswith(("+", "-")):
            return f"+{number}"
    return number


def _extract_num_client(text: str) -> str | None:
    if match := CUSTOMER_ID_RE.search(text):
        return str(int(match.group(1)))

    normalized = _norm_for_matching(text)
    if any(marker in normalized for marker in ["num client", "معاك", "client", "cli"]):
        spoken = words_to_number(normalized)
        if spoken is not None and 100 <= spoken <= 999999:
            return str(spoken)

    candidates = []
    for match in STANDALONE_CLIENT_NUMBER_RE.finditer(text):
        candidate = match.group(0)
        prefix = _norm_for_matching(text[max(0, match.start() - 24) : match.start()])
        if re.search(
            r"(bon(?: de commande)?|commande|كوموند|cmd|ord|import[-_ ]?|diametre|diameter|axe|axis|addition|add|index)$",
            prefix,
            flags=re.IGNORECASE,
        ):
            continue
        candidates.append(candidate)
    short_candidates = [
        candidate
        for candidate in candidates
        if 3 <= len(candidate.lstrip("0") or "0") <= 4 and candidate not in {"2026", "2025", "2048"}
    ]
    if short_candidates:
        return short_candidates[0].lstrip("0") or "0"
    if len(candidates) == 1:
        candidate = candidates[0]
        if candidate not in {"2026", "2025", "2048"} and candidate not in {"150", "156", "160", "167"}:
            return candidate.lstrip("0") or "0"
    return None


def _extract_order_id(text: str) -> str | None:
    if match := ORDER_RE.search(text):
        return match.group(0).upper()
    if match := BON_NUMBER_RE.search(text):
        value = match.group(1).strip()
        return value.upper() if not value.isdigit() else value
    return None


def _extract_index(text: str) -> str | None:
    normalized = _norm_for_matching(text)
    if match := INDEX_LABEL_RE.search(normalized):
        return match.group(1)

    candidates = INDEX_RE.findall(text)
    if len(candidates) != 1:
        return None

    optical_markers = ["od", "og", "sphere", "cyl", "cylindre", "axe", "axis", "addition", "plus", "moins"]
    if any(marker in normalized for marker in optical_markers):
        return None
    return candidates[0]


def _extract_quantity(text: str) -> str | None:
    normalized = _norm_for_matching(text)
    if "كعبتين" in normalized or "deux pieces" in normalized or "deux piece" in normalized:
        return "2"
    if "كعبة" in normalized or "قطعة" in normalized or "piece" in normalized:
        if "2" not in normalized:
            return "1"
    if match := re.search(r"\b([12])\s*(?:كعبة|pi[eè]ce|pieces?)\b", text, flags=re.IGNORECASE):
        return match.group(1)
    return None


def _extract_priority(text: str) -> str | None:
    normalized = _norm_for_matching(text)
    if any(token in normalized for token in ["illico", "urgent", "urgence"]):
        return "illico"
    if "import" in normalized or "commande import" in normalized:
        return "import"
    return None


def _extract_reference(text: str, excluded: set[str] | None = None) -> str | None:
    excluded = {str(value).replace(" ", "") for value in (excluded or set()) if value}
    normalized = _norm_for_matching(text)
    if not any(token in normalized for token in ["reference", "référence", "ref", "بلارات", "blarat", "réf"]):
        return None
    for match in REFERENCE_RE.finditer(text):
        candidate = match.group(0).replace(" ", "")
        if candidate in excluded:
            continue
        return candidate
    return None


def _extract_optical_slots(text: str) -> dict[str, Any]:
    normalized = _norm_for_matching(text)
    slots: dict[str, Any] = {}

    alias_patterns = {
        "od": ["od", "oeil droit", "droit", "right", "عاليمين", "اليمنى"],
        "og": ["og", "oeil gauche", "gauche", "left", "عاليسار", "الثانية", "اليسرى"],
    }
    next_markers = alias_patterns["od"] + alias_patterns["og"]

    for eye, aliases in alias_patterns.items():
        for alias in aliases:
            marker = _norm_for_matching(alias)
            match_obj = re.search(rf"\b{re.escape(marker)}\b", normalized)
            index = match_obj.start() if match_obj else normalized.find(marker) if len(marker) > 3 else -1
            if index == -1:
                continue
            tail = normalized[index + len(marker) :].strip()
            end_positions = [tail.find(_norm_for_matching(item)) for item in next_markers if _norm_for_matching(item) in tail]
            end_positions = [pos for pos in end_positions if pos > 0]
            block = tail[: min(end_positions)] if end_positions else tail
            axis_match = AXIS_RE.search(block)
            if axis_match:
                slots[f"{eye}_axis"] = axis_match.group(1)
                block = AXIS_RE.sub(" ", block)
            values = SIGNED_NUMBER_RE.findall(block)
            if values:
                slots[f"{eye}_sphere"] = _normalize_signed_value(*values[0])
            if len(values) > 1:
                slots[f"{eye}_cyl"] = _normalize_signed_value(*values[1])
            break

    if addition_match := ADDITION_RE.search(normalized):
        slots["addition"] = _normalize_signed_value(addition_match.group(1), addition_match.group(2))

    return slots


def _recover_missing_slots_from_turn(
    text: str,
    extracted_slots: dict[str, Any],
    session_state: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(session_state, dict):
        return {}

    missing = [str(item) for item in session_state.get("missing_slots", []) if item]
    if not missing:
        return {}

    normalized = _norm_for_matching(text)
    compact = normalized.replace(" ", "")
    updates: dict[str, Any] = {}

    if "num_client" in missing and not extracted_slots.get("num_client"):
        if num_client := _extract_num_client(text):
            updates["num_client"] = num_client
        elif compact.isdigit() and 3 <= len(compact) <= 6:
            updates["num_client"] = str(int(compact))

    if "order_id" in missing and not extracted_slots.get("order_id"):
        if order_id := _extract_order_id(text):
            updates["order_id"] = order_id
        elif compact.isdigit() and 4 <= len(compact) <= 8:
            updates["order_id"] = compact

    if "reference" in missing and not extracted_slots.get("reference"):
        if reference := _extract_reference(text):
            updates["reference"] = reference
        elif compact.isdigit() and len(compact) == 4:
            updates["reference"] = compact

    if "quantity" in missing and not extracted_slots.get("quantity") and compact in {"1", "2"}:
        updates["quantity"] = compact

    if "priority" in missing and not extracted_slots.get("priority"):
        if priority := _extract_priority(text):
            updates["priority"] = priority

    if "phone" in missing and not extracted_slots.get("phone"):
        if phone_match := PHONE_RE.search(text):
            updates["phone"] = phone_match.group(0)

    if "date" in missing and not extracted_slots.get("date"):
        if date_match := DATE_RE.search(text):
            updates["date"] = date_match.group(0)

    if "time_slot" in missing and not extracted_slots.get("time_slot"):
        if time_match := TIME_RE.search(text):
            updates["time_slot"] = time_match.group(0)

    if "diameter" in missing and not extracted_slots.get("diameter"):
        if diameter_match := re.search(r"\b(65|70|75)\b", compact):
            updates["diameter"] = diameter_match.group(1)

    return canonicalize_slots(updates)


def infer_intent(text: str, extracted_slots: dict[str, Any] | None = None) -> str:
    extracted_slots = canonicalize_slots(extracted_slots)
    normalized = _norm_for_matching(text)
    words = set(normalized.split())

    def has_keyword(keywords: list[str]) -> bool:
        for keyword in keywords:
            keyword = _norm_for_matching(keyword)
            if not keyword:
                continue
            if keyword in normalized:
                return True
            if keyword in words:
                return True
        return False

    intent_keywords = DOMAIN_CFG.get("intent_keywords", {})
    business_keywords = DOMAIN_CFG.get("business_keywords", [])
    has_num_client = bool(extracted_slots.get("num_client"))
    has_order = bool(extracted_slots.get("order_id"))
    has_greeting = has_keyword(intent_keywords.get("greeting", []))
    schedule_signal = has_keyword(intent_keywords.get("delivery_schedule", [])) or any(
        token in normalized
        for token in ["livraison", "agence", "secteur", "créneau", "creneau", "horaire"]
    )
    availability_signal = any(
        token in normalized
        for token in ["availability", "disponibilite", "disponibilité", "disponible", "stock", "dispo", "mawjoud", "fabrication", "متوفرة"]
    )
    reference_signal = bool(extracted_slots.get("reference") or extracted_slots.get("lens_code")) and any(
        token in normalized
        for token in ["reference", "référence", "ref", "code", "code verre", "ordonnance", "blarat", "بلارات", "ثبتلي"]
    )
    has_product_signal = bool(
        extracted_slots.get("product")
        or extracted_slots.get("od_sphere")
        or extracted_slots.get("og_sphere")
    )
    create_keywords = intent_keywords.get("create_order", [])
    order_creation_keywords = intent_keywords.get("order_creation", [])
    strong_create_keywords = [
        keyword
        for keyword in [*create_keywords, *order_creation_keywords]
        if _norm_for_matching(keyword) not in {"illico"}
    ]
    create_signal = has_product_signal or has_keyword(strong_create_keywords)
    tracking_keywords = [
        keyword
        for keyword in intent_keywords.get("order_tracking", [])
        if _norm_for_matching(keyword) not in {"commande", "كوموند"}
    ]
    business_opening = has_keyword(business_keywords) or any(
        token in normalized for token in ["commande", "bon de commande", "ordonnance", "verre", "traitement", "suivi"]
    )
    tracking_question_signal = any(
        token in normalized for token in ["وين", "وصلت", "وقتاش", "تجي", "حالة", "statut", "livraison", "suivi"]
    )
    tracking_signal = has_order or has_keyword(tracking_keywords) or (tracking_question_signal and business_opening)
    price_signal = has_keyword(intent_keywords.get("price_inquiry", []))

    if not has_num_client and business_opening and not create_signal and not tracking_signal and not availability_signal and not reference_signal and not schedule_signal:
        return "get_num_client"

    if has_greeting and has_num_client and len(words) <= 4 and not (
        tracking_signal or create_signal or availability_signal or reference_signal
    ):
        return "get_num_client"

    if availability_signal and not create_signal:
        return "availability_inquiry"
    if schedule_signal and not create_signal and not has_order:
        return "delivery_schedule"
    if reference_signal and not create_signal:
        return "reference_confirmation"
    if tracking_signal:
        return "order_tracking"
    if create_signal:
        return "create_order"
    if price_signal:
        return "price_inquiry"
    if has_num_client and not (tracking_signal or create_signal or availability_signal or reference_signal):
        return "get_num_client"
    if has_greeting and not has_keyword(business_keywords):
        return "greeting"
    if has_num_client and len(words) <= 4 and not (tracking_signal or create_signal or availability_signal or reference_signal):
        return "get_num_client"

    ordered_intents = ["delivery_schedule", "appointment_booking", "store_info", "product_info", "thanks"]
    for intent in ordered_intents:
        if has_keyword(intent_keywords.get(intent, [])):
            return canonicalize_intent(intent)
    return "unknown"


def _extract_alias(text: str, alias_map: dict[str, list[str]]) -> str | None:
    lowered = _norm_for_matching(text)
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            normalized_alias = _norm_for_matching(alias)
            if normalized_alias and normalized_alias in lowered:
                return canonical
    return None


def extract_slots(text: str) -> dict[str, Any]:
    slots: dict[str, Any] = {}

    if num_client := _extract_num_client(text):
        slots["num_client"] = num_client
    if order_id := _extract_order_id(text):
        slots["order_id"] = order_id
    if index := _extract_index(text):
        slots["index"] = index
    if match := PHONE_RE.search(text):
        slots["phone"] = match.group(0)
    if match := DATE_RE.search(text):
        slots["date"] = match.group(0)
    if match := TIME_RE.search(text):
        slots["time_slot"] = match.group(0)
    if quantity := _extract_quantity(text):
        slots["quantity"] = quantity
    if priority := _extract_priority(text):
        slots["priority"] = priority
    if reference := _extract_reference(text, excluded={slots.get("num_client"), slots.get("order_id")}):
        slots["reference"] = reference
    if match := LENS_CODE_CONTEXT_RE.search(text):
        detected_code = match.group(1).upper()
        slots["lens_code"] = detected_code
        if not slots.get("reference"):
            slots["reference"] = detected_code
    elif detected_code := _extract_known_lens_code(text):
        slots["lens_code"] = detected_code
        if not slots.get("reference"):
            slots["reference"] = detected_code

    for slot_name, pattern in DOMAIN_CFG.get("slot_patterns", {}).items():
        if slot_name in slots:
            continue
        try:
            compiled = re.compile(str(pattern), re.IGNORECASE)
        except re.error:
            logger.warning("Invalid slot pattern for %s: %s", slot_name, pattern)
            continue
        if match := compiled.search(text):
            slots[slot_name] = match.group(1) if match.groups() else match.group(0)

    city = _extract_alias(text, DOMAIN_CFG.get("location_aliases", {}))
    if city:
        slots["city"] = city
    for key, value in _extract_rag_delivery_slots(text).items():
        if not slots.get(key):
            slots[key] = value

    normalized = _norm_for_matching(text)
    if "orma blanc" in normalized:
        slots["material"] = "orma blanc"

    product = _extract_lexicon_value(text, "product") or _extract_alias(text, DOMAIN_CFG.get("product_aliases", {}))
    if product:
        slots["product"] = product

    for field in ["material", "treatment", "color"]:
        if not slots.get(field):
            value = _extract_lexicon_value(text, field)
            if value:
                slots[field] = value

    if not slots.get("color"):
        color_match = re.search(r"\b(marron|gris|blanc|brun|bleu)\b", _norm_for_matching(text), flags=re.IGNORECASE)
        if color_match:
            slots["color"] = color_match.group(1).lower()

    if slots.get("material") == "orma blanc":
        if slots.get("treatment") == "blanc":
            slots.pop("treatment", None)
        if slots.get("color") == "blanc":
            slots.pop("color", None)

    slots.update(_extract_optical_slots(text))
    return canonicalize_slots(slots)


def _merge_slots(base: dict[str, Any] | None, updates: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(canonicalize_slots(base))
    for key, value in canonicalize_slots(updates).items():
        if value not in (None, "", []):
            merged[key] = value
    return merged


def _resolve_turn_state(
    text: str,
    intent: str,
    extracted_slots: dict[str, Any],
    session_state: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    session_state = session_state or {}
    active_intent = canonicalize_intent(session_state.get("active_intent"))
    open_form = bool(session_state.get("open_form"))
    known_slots = session_state.get("slots", {}) if isinstance(session_state.get("slots"), dict) else {}

    resolved_intent = canonicalize_intent(intent)
    carried_slots: dict[str, Any] = {}
    if active_intent and open_form and resolved_intent in {"unknown", "get_num_client", active_intent}:
        resolved_intent = active_intent
        carried_slots = known_slots
    elif resolved_intent != "unknown" and resolved_intent == active_intent and open_form:
        carried_slots = known_slots
    elif resolved_intent == "unknown" and active_intent and len(text.split()) <= 8 and extracted_slots:
        resolved_intent = active_intent
        carried_slots = known_slots

    return resolved_intent, _merge_slots(carried_slots, extracted_slots)


def _summarize_session_state(session_state: dict[str, Any] | None) -> dict[str, Any]:
    session_state = session_state or {}
    summary = {
        "active_intent": session_state.get("active_intent"),
        "known_slots": session_state.get("slots", {}) if isinstance(session_state.get("slots"), dict) else {},
        "missing_slots": list(session_state.get("missing_slots", []) or []),
        "open_form": bool(session_state.get("open_form")),
        "review_required": bool(session_state.get("review_required")),
        "customer_context": session_state.get("customer_context", {})
        if isinstance(session_state.get("customer_context"), dict)
        else {},
        "last_tool_call": session_state.get("last_tool_call"),
    }
    return summary


def route_to_tool(intent: str, slots: dict[str, Any]) -> tuple[str | None, dict[str, Any], list[str]]:
    intent = canonicalize_intent(intent)
    slots = canonicalize_slots(slots)
    required = DOMAIN_CFG.get("required_slots", {}).get(intent, [])
    missing = [slot for slot in required if not slots.get(slot)]
    tool_name = DOMAIN_CFG.get("intent_to_tool", {}).get(intent)
    if missing or not tool_name:
        return None, {}, missing

    tool_args = {slot: slots.get(slot) for slot in required}
    for optional in [
        "treatment",
        "color",
        "material",
        "index",
        "lens_code",
        "city",
        "agence",
        "secteur",
        "delivery_slot",
        "requested_slot",
        "phone",
        "date",
        "date_from",
        "date_to",
        "time_slot",
        "priority",
        "quantity",
        "reference",
        "diameter",
        "od_sphere",
        "od_cyl",
        "od_axis",
        "og_sphere",
        "og_cyl",
        "og_axis",
        "addition",
    ]:
        if slots.get(optional):
            tool_args[optional] = slots[optional]
    return tool_name, tool_args, missing


def _should_execute_tool(intent: str, missing_slots: list[str]) -> bool:
    if missing_slots:
        return False
    policy = str(DOMAIN_CFG.get("tool_execution_policy", {}).get(intent, "auto")).lower()
    return policy == "auto"


def _resolve_runtime_mode(runtime_mode: str | None) -> str:
    configured_default = str(DOMAIN_CFG.get("runtime_modes", {}).get("default", "speak")).lower()
    mode = str(runtime_mode or configured_default or "speak").lower()
    if mode not in {"collect_execute", "speak", "autonomous"}:
        return configured_default if configured_default in {"collect_execute", "speak", "autonomous"} else "speak"
    return mode


def _should_execute_tool_for_mode(intent: str, missing_slots: list[str], runtime_mode: str) -> bool:
    if missing_slots:
        return False
    policy = str(DOMAIN_CFG.get("tool_execution_policy", {}).get(intent, "auto")).lower()
    if runtime_mode == "autonomous":
        return policy in {"auto", "draft"}
    return policy == "auto"


def _mode_instruction(runtime_mode: str) -> str:
    if runtime_mode == "collect_execute":
        return (
            "You are in collect_execute mode. Keep answers short and operational. "
            "Focus on extracting fields, asking only for missing fields, or confirming the detected action."
        )
    if runtime_mode == "autonomous":
        return (
            "You are in autonomous mode. Speak naturally, take initiative when the request is clear, "
            "and confirm executed actions clearly."
        )
    return (
        "You are in speak mode. Speak naturally and professionally, but stay careful with business actions "
        "that still need human review."
    )


def _render_collect_response(
    *,
    intent: str,
    slots: dict[str, Any],
    missing_slots: list[str],
    tool_name: str | None,
    tool_result: dict[str, Any] | None,
    auto_execute_tool: bool,
) -> str:
    if intent == "unknown":
        return "Demande captee, mais l'intent n'est pas encore clair. Merci de reformuler ou de donner plus de contexte."
    if intent == "get_num_client":
        return "Intent detecte: get_num_client. Champ manquant: num_client."
    if missing_slots:
        return (
            f"Intent detecte: {intent}. "
            f"Champs detectes: {json.dumps(slots, ensure_ascii=False)}. "
            f"Champs manquants: {', '.join(missing_slots)}."
        )
    if tool_name and tool_result is not None and auto_execute_tool:
        return (
            f"Intent detecte: {intent}. "
            f"Action executee: {tool_name}. "
            f"Resultat: {json.dumps(tool_result, ensure_ascii=False)}."
        )
    if tool_name:
        return (
            f"Intent detecte: {intent}. "
            f"Action preparee: {tool_name}. "
            f"Parametres: {json.dumps(slots, ensure_ascii=False)}. "
            "Validation humaine requise."
        )
    return f"Intent detecte: {intent}. Slots: {json.dumps(slots, ensure_ascii=False)}."


def _select_few_shots(user_text: str, intent: str) -> list[dict[str, str]]:
    if not FEW_SHOTS_CFG:
        return []
    intent = canonicalize_intent(intent)

    def score(example: dict[str, Any]) -> float:
        example_user = str(example.get("user") or example.get("client") or example.get("opticien") or "")
        example_intent = canonicalize_intent(example.get("intent", ""))
        base = 1.0 if example_intent == intent else 0.0
        a_tokens = set(_norm_for_matching(user_text).split())
        b_tokens = set(_norm_for_matching(example_user).split())
        overlap = len(a_tokens & b_tokens) / max(len(a_tokens), len(b_tokens), 1)
        return base + overlap

    ranked = sorted(FEW_SHOTS_CFG, key=score, reverse=True)
    selected = []
    for example in ranked[: CFG.few_shot_top_k]:
        user_text = example.get("user") or example.get("client") or example.get("opticien")
        assistant_text = example.get("assistant") or example.get("agent")
        if user_text and assistant_text:
            selected.append(
                {"role": "user", "content": str(user_text)}
            )
            selected.append(
                {"role": "assistant", "content": str(assistant_text)}
            )
    return selected


def _build_grounding_context(
    *,
    intent: str,
    slots: dict[str, Any],
    missing_slots: list[str],
    tool_result: dict[str, Any] | None,
    rag_results: list[dict[str, Any]],
    memory_hits: list[dict[str, Any]],
    session_state: dict[str, Any] | None = None,
) -> str:
    parts: list[str] = []

    if session_state:
        parts.append(f"[session_state]\n{json.dumps(session_state, ensure_ascii=False)}")
    if tool_result:
        parts.append(f"[tool_result]\n{json.dumps(tool_result, ensure_ascii=False)}")
    if missing_slots:
        parts.append(f"[missing_slots]\n{', '.join(missing_slots)}")
    if memory_hits:
        joined_memory = "\n\n".join(hit["text"] for hit in memory_hits)
        parts.append(f"[memory]\n{joined_memory}")
    if rag_results:
        joined_rag = "\n\n".join(f"{item['text']}" for item in rag_results[: CFG.retrieval_top_k])
        parts.append(f"[retrieval]\n{joined_rag}")
    if intent != "unknown":
        parts.append(f"[intent]\n{intent}")
    if slots:
        parts.append(f"[slots]\n{json.dumps(slots, ensure_ascii=False)}")
    return "\n\n".join(parts)


def _adapter_key(model_variant: str) -> str:
    adapter_dir = CFG.resolve_adapter_dir(
        model_variant,
        allow_fallback=CFG.should_allow_variant_fallback(model_variant),
    )
    return f"{model_variant}:{adapter_dir or 'base'}"


def load_llm(model_variant: str = "prod") -> tuple[Any, Any]:
    cache_key = _adapter_key(model_variant)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    with _MODEL_LOCK:
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        allow_fallback = CFG.should_allow_variant_fallback(model_variant)
        adapter_dir = CFG.resolve_adapter_dir(model_variant, allow_fallback=allow_fallback)
        if model_variant == "prod" and not allow_fallback and adapter_dir is None:
            raise FileNotFoundError("Production adapter not found. Promote a production adapter before serving.")

        logger.info("Loading LLM variant=%s", model_variant)
        tokenizer = AutoTokenizer.from_pretrained(
            CFG.base_model,
            use_fast=True,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if CFG.use_4bit and torch.cuda.is_available():
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
            if torch.cuda.is_available():
                kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(CFG.base_model, **kwargs)

        if adapter_dir:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_dir))
            logger.info("Loaded adapter for variant=%s from %s", model_variant, adapter_dir)

        model.config.pad_token_id = tokenizer.pad_token_id
        model.eval()
        _MODEL_CACHE[cache_key] = (model, tokenizer)
        return model, tokenizer


def unload_llm(model_variant: str | None = None) -> None:
    keys = list(_MODEL_CACHE.keys())
    for key in keys:
        if model_variant and not key.startswith(f"{model_variant}:"):
            continue
        model, tokenizer = _MODEL_CACHE.pop(key)
        del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _generate(messages: list[dict[str, str]], model_variant: str = "prod") -> str:
    model, tokenizer = load_llm(model_variant=model_variant)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=CFG.max_seq_len - CFG.max_new_tokens,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=CFG.max_new_tokens,
            do_sample=True,
            temperature=CFG.temperature,
            top_p=CFG.top_p,
            repetition_penalty=CFG.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


_PHONE_OUT_RE = re.compile(r"\+?216[\s\-]?\d[\d\s\-]{6,12}")
_ORDER_OUT_RE = re.compile(r"ORD-[A-Z0-9]{5,}")
_ENGLISH_RE = re.compile(r"\b[A-Za-z]{4,}\b")
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def _scrub_pii(text: str, slots: dict[str, Any]) -> str:
    real_phone = str(slots.get("phone", "")).replace(" ", "").replace("-", "")
    real_order = str(slots.get("order_id", ""))

    def keep_phone(match: re.Match[str]) -> str:
        raw = match.group(0).replace(" ", "").replace("-", "")
        return match.group(0) if real_phone and raw in real_phone else ""

    def keep_order(match: re.Match[str]) -> str:
        return match.group(0) if real_order and match.group(0) == real_order else ""

    text = _PHONE_OUT_RE.sub(keep_phone, text)
    text = _ORDER_OUT_RE.sub(keep_order, text)
    return re.sub(r"  +", " ", text).strip()


def _strip_leaked_english(text: str, target_script: str) -> str:
    if target_script == "arabizi":
        return text.strip()

    allowed = {"sivo", "dt", "indice", "progressive", "photochromic", "index"}

    def replace_word(match: re.Match[str]) -> str:
        word = match.group(0)
        return word if word.lower() in allowed else ""

    text = _ENGLISH_RE.sub(replace_word, text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _truncate(text: str, max_sentences: int = 6) -> str:
    sentences = re.split(r"(?<=[.!؟])\s+", text)
    return " ".join(sentences[:max_sentences]).strip()


def _humanize(text: str) -> str:
    assistant_name = DOMAIN_CFG.get("assistant_name", "agent")
    text = re.sub(r"\bassistant\b", assistant_name, text, flags=re.IGNORECASE)
    text = re.sub(r"\bbot\b", assistant_name, text, flags=re.IGNORECASE)
    text = re.sub(r"أنا مساعد ذكي", f"أنا {assistant_name}", text)
    return text.strip()


def _is_garbage_response(text: str) -> bool:
    if not text or len(text.strip()) < 8:
        return True
    if _CJK_RE.search(text):
        return True
    if text.count(text[:10]) > 4 if len(text) >= 10 else False:
        return True
    return False


def _get_fallback_response(kind: str = "unclear") -> str:
    organization = DOMAIN_CFG.get("organization", "your company")
    dont_know = [
        f"والله هالحكاية موش من خدمة {organization}. إذا تحب نعاونك في الموضوع اللي يخصنا قولي.",
        "سامحني، ما عنديش معلومة صحيحة على الموضوع هذا وما نحبش نخمن.",
    ]
    unclear = [
        "عاود قلي شنية تحتاج بالضبط باش نجم نعاونك.",
        "المعلومة موش واضحة برشا، تنجم تفسرلي أكثر؟",
    ]
    return random.choice(dont_know if kind == "dont_know" else unclear)


def production_infer(
    user_text: str,
    retriever: VectorRAGRetriever,
    history: list[dict[str, Any]] | None = None,
    *,
    session_id: str | None = None,
    model_variant: str = "prod",
    runtime_mode: str | None = None,
    memory_store: ConversationMemoryStore | None = None,
    tool_registry: ToolRegistry | None = None,
    correction_store: LiveCorrectionStore | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    tool_registry = tool_registry or get_tool_registry()
    history = history or []
    runtime_mode = _resolve_runtime_mode(runtime_mode)
    user_script = _detect_script_like(user_text)
    target_script = _preferred_response_script(user_text)

    text = normalize_text(user_text)
    session_state = memory_store.get_session_state(session_id) if memory_store and session_id else {}
    extracted_slots = extract_slots(text)
    recovered_slots = _recover_missing_slots_from_turn(text, extracted_slots, session_state)
    if recovered_slots:
        extracted_slots = _merge_slots(extracted_slots, recovered_slots)
    raw_intent = infer_intent(text, extracted_slots=extracted_slots)
    intent, slots = _resolve_turn_state(text, raw_intent, extracted_slots, session_state)
    state_summary = _summarize_session_state(session_state)

    normalized_text = _norm_for_matching(text)
    raw_normalized_text = _norm_for_matching(user_text)
    out_of_domain = (
        intent == "unknown"
        and not state_summary.get("open_form")
        and any(keyword in normalized_text or keyword in raw_normalized_text for keyword in DOMAIN_CFG.get("out_of_domain_keywords", []))
    )
    if out_of_domain:
        latency = round((time.perf_counter() - start) * 1000, 1)
        return {
            "response": _get_fallback_response("dont_know"),
            "intent": intent,
            "slots": slots,
            "tool_call": None,
            "tool_result": None,
            "rag_results": [],
            "memory_hits": [],
            "missing_slots": [],
            "session_state": state_summary,
            "needs_human_review": False,
            "latency_ms": latency,
            "model_variant": model_variant,
            "runtime_mode": runtime_mode,
            "correction_applied": False,
        }

    if (
        all(char.isascii() or char.isspace() for char in user_text.strip())
        and len(user_text.split()) <= 2
        and intent == "unknown"
    ):
        latency = round((time.perf_counter() - start) * 1000, 1)
        return {
            "response": _get_fallback_response("unclear"),
            "intent": intent,
            "slots": slots,
            "tool_call": None,
            "tool_result": None,
            "rag_results": [],
            "memory_hits": [],
            "missing_slots": [],
            "session_state": state_summary,
            "needs_human_review": False,
            "latency_ms": latency,
            "model_variant": model_variant,
            "runtime_mode": runtime_mode,
            "correction_applied": False,
        }

    tool_name, tool_args, missing_slots = route_to_tool(intent, slots)
    if intent == "get_num_client" and not slots.get("num_client"):
        missing_slots = ["num_client"]
    auto_execute_tool = _should_execute_tool_for_mode(intent, missing_slots, runtime_mode)
    tool_result = tool_registry.execute(tool_name, tool_args) if tool_name and auto_execute_tool else None
    if isinstance(tool_result, dict):
        tool_missing = [str(item) for item in tool_result.get("missing_fields", []) if item]
        if tool_missing:
            missing_slots = list(dict.fromkeys([*missing_slots, *tool_missing]))

    rag_results = []
    if intent not in {"greeting", "thanks"}:
        rag_results = retriever.search(text, top_k=CFG.retrieval_top_k)

    memory_hits = []
    if memory_store and session_id:
        memory_hits = memory_store.retrieve_relevant(session_id, text, top_k=CFG.memory_top_k)

    correction_match = None
    if correction_store is not None:
        correction_match = correction_store.find_best(
            user_text=text,
            intent=intent,
            slots=slots,
            runtime_mode=runtime_mode,
        )

    messages: list[dict[str, str]] = []
    for prompt in _resolve_system_prompt_messages(
        user_script=user_script,
        target_script=target_script,
        intent=intent,
    ):
        messages.append({"role": "system", "content": prompt})
    messages.extend(
        [
            {"role": "system", "content": _mode_instruction(runtime_mode)},
            {"role": "system", "content": _script_instruction(target_script)},
        ]
    )
    messages.extend(_select_few_shots(text, intent))

    if history:
        for message in history[-CFG.max_history_messages :]:
            role = message.get("role")
            content = message.get("content")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})

    context = _build_grounding_context(
        intent=intent,
        slots=slots,
        missing_slots=missing_slots,
        tool_result=tool_result,
        rag_results=rag_results,
        memory_hits=memory_hits,
        session_state=state_summary,
    )
    if context:
        messages.append({"role": "system", "content": context})

    if correction_match and correction_match.get("action", "replace") != "replace":
        messages.append(
            {
                "role": "system",
                "content": f"[admin_correction_hint]\n{correction_match.get('corrected_response', '')}",
            }
        )

    if intent == "unknown":
        messages.append(
            {
                "role": "system",
                "content": "إذا ما فهمتش الطلب، اطلب توضيح مختصر وواضح. ما تجاوبش على حاجة موش واضحة.",
            }
        )
    elif intent == "get_num_client":
        messages.append(
            {
                "role": "system",
                "content": "Ask only for num_client first. Do not continue order tracking or order creation until num_client is confirmed.",
            }
        )

    messages.append({"role": "user", "content": text})

    correction_applied = False
    if correction_match and correction_match.get("action", "replace") == "replace":
        response = str(correction_match.get("corrected_response", "")).strip()
        correction_applied = True
    elif runtime_mode == "collect_execute":
        response = _render_collect_response(
            intent=intent,
            slots=slots,
            missing_slots=missing_slots,
            tool_name=tool_name,
            tool_result=tool_result,
            auto_execute_tool=auto_execute_tool,
        )
    else:
        response = _generate(messages, model_variant=model_variant)
        response = _scrub_pii(response, slots)
        response = _strip_leaked_english(response, target_script=target_script)
        response = _humanize(response)
        if _is_script_mismatch(response, target_script):
            retry_messages = _build_script_retry_messages(messages, target_script)
            response = _generate(retry_messages, model_variant=model_variant)
            response = _scrub_pii(response, slots)
            response = _strip_leaked_english(response, target_script=target_script)
            response = _humanize(response)
        response = _truncate(response)

    if _is_garbage_response(response):
        response = _get_fallback_response("unclear")

    tool_status = str((tool_result or {}).get("status", "")).lower() if isinstance(tool_result, dict) else ""
    needs_human_review = bool(missing_slots) or bool(tool_name and not auto_execute_tool and runtime_mode != "collect_execute")
    if tool_status in {"error", "verification_failed", "not_found"}:
        needs_human_review = True
    updated_state = session_state
    if memory_store and session_id:
        updated_state = memory_store.update_session_state(
            session_id,
            intent=intent,
            slots=slots,
            missing_slots=missing_slots,
            review_required=needs_human_review,
            tool_call={"name": tool_name, "args": tool_args} if tool_name else None,
            tool_result=tool_result,
        )

    latency = round((time.perf_counter() - start) * 1000, 1)
    logger.info(
        "infer variant=%s intent=%s tool=%s latency=%.1fms",
        model_variant,
        intent,
        tool_name,
        latency,
    )

    return {
        "response": response,
        "intent": intent,
        "slots": slots,
        "tool_call": {"name": tool_name, "args": tool_args} if tool_name else None,
        "tool_result": tool_result,
        "rag_results": rag_results,
        "memory_hits": memory_hits,
        "missing_slots": missing_slots,
        "session_state": _summarize_session_state(updated_state),
        "needs_human_review": needs_human_review,
        "latency_ms": latency,
        "model_variant": model_variant,
        "runtime_mode": runtime_mode,
        "response_script_target": target_script,
        "response_script_detected": _detect_script_like(response),
        "correction_applied": correction_applied,
    }
