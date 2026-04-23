"""
Shared domain normalization helpers.

These helpers keep the data-prep, runtime, evaluation, and validation layers
aligned on the same intent, role, and slot vocabulary.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any


ROLE_ALIASES = {
    "system": "system",
    "user": "user",
    "client": "user",
    "customer": "user",
    "opticien": "user",
    "caller": "user",
    "assistant": "assistant",
    "agent": "assistant",
    "advisor": "assistant",
    "support": "assistant",
}

INTENT_ALIASES = {
    "order_creation": "create_order",
    "creation_commande": "create_order",
    "commande_creation": "create_order",
}

NON_STICKY_INTENTS = {
    "unknown",
    "greeting",
    "thanks",
    "get_num_client",
}

SLOT_NAME_ALIASES = {
    "customer_id": "num_client",
    "customerid": "num_client",
    "codeclient": "num_client",
    "code_client": "num_client",
    "client_code": "num_client",
    "code_client_opticien": "num_client",
    "numcmd": "order_id",
    "num_cmd": "order_id",
    "numerobon": "order_id",
    "numero_bon": "order_id",
    "diametre": "diameter",
    "traitement": "treatment",
    "coating": "treatment",
    "coloration": "color",
    "couleur": "color",
    "typecommande": "order_type",
    "type_commande": "order_type",
    "statut": "status",
    "etat": "status",
    "etatcommande": "status",
    "statuscommande": "status",
    "nomporteur": "customer_name",
    "nom_porteur": "customer_name",
    "datefrom": "date_from",
    "dateto": "date_to",
    "timeslot": "time_slot",
    "mode": "priority",
    "agence_livraison": "agence",
    "sector": "secteur",
    "secteur_livraison": "secteur",
    "creneau": "delivery_slot",
    "creneau_livraison": "delivery_slot",
    "codeverre": "lens_code",
    "code_verre": "lens_code",
    "verre_code": "lens_code",
}

OPTICAL_SLOT_ALIASES = {
    "sphere": "sphere",
    "sph": "sphere",
    "cyl": "cyl",
    "cylindre": "cyl",
    "cylinder": "cyl",
    "axe": "axis",
    "axis": "axis",
    "add": "addition",
    "addition": "addition",
}

_NON_DIGIT_RE = re.compile(r"\D+")
_WHITESPACE_RE = re.compile(r"\s+")

_NUMBER_WORDS = {
    "zero": 0,
    "zéro": 0,
    "صفر": 0,
    "واحد": 1,
    "وحدة": 1,
    "واحدة": 1,
    "un": 1,
    "une": 1,
    "اثنين": 2,
    "اثنان": 2,
    "ثنين": 2,
    "زوز": 2,
    "زوج": 2,
    "deux": 2,
    "ثلاثة": 3,
    "ثلاث": 3,
    "ثلاثةة": 3,
    "trois": 3,
    "اربعة": 4,
    "أربعة": 4,
    "اربعه": 4,
    "quatre": 4,
    "خمسة": 5,
    "خمسه": 5,
    "cinq": 5,
    "ستة": 6,
    "سته": 6,
    "six": 6,
    "سبعة": 7,
    "سبعه": 7,
    "sept": 7,
    "ثمانية": 8,
    "ثمانيه": 8,
    "ثماني": 8,
    "huit": 8,
    "تسعة": 9,
    "تسعه": 9,
    "neuf": 9,
    "عشرة": 10,
    "عشره": 10,
    "dix": 10,
    "احداش": 11,
    "إحداش": 11,
    "onze": 11,
    "اثناش": 12,
    "douze": 12,
    "ثلاثطاش": 13,
    "treize": 13,
    "اربعطاش": 14,
    "quatorze": 14,
    "خمسطاش": 15,
    "quinze": 15,
    "ستاش": 16,
    "seize": 16,
    "سبعتاش": 17,
    "dixsept": 17,
    "ثمانتاش": 18,
    "dixhuit": 18,
    "تسعتاش": 19,
    "dixneuf": 19,
    "عشرين": 20,
    "vingt": 20,
    "ثلاثين": 30,
    "trente": 30,
    "اربعين": 40,
    "quarante": 40,
    "خمسين": 50,
    "cinquante": 50,
    "ستين": 60,
    "soixante": 60,
    "سبعين": 70,
    "soixantedix": 70,
    "ثمانين": 80,
    "quatrevingt": 80,
    "quatrevingts": 80,
    "تسعين": 90,
    "quatrevingtdix": 90,
    "مية": 100,
    "مئة": 100,
    "cent": 100,
    "الف": 1000,
    "ألف": 1000,
    "mille": 1000,
}


def normalize_role(role: Any) -> str:
    raw = str(role or "user").strip().lower()
    return ROLE_ALIASES.get(raw, "user")


def message_content(message: dict[str, Any]) -> str:
    return str(message.get("content", message.get("text", "")) or "").strip()


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message_content(message)
        if not content:
            continue
        normalized.append(
            {
                "role": normalize_role(message.get("role")),
                "content": content,
            }
        )
    return normalized


def canonicalize_intent(intent: Any) -> str:
    raw = str(intent or "").strip()
    if not raw:
        return "unknown"
    lowered = raw.lower()
    return INTENT_ALIASES.get(lowered, lowered)


def is_sticky_intent(intent: Any) -> bool:
    return canonicalize_intent(intent) not in NON_STICKY_INTENTS


def canonicalize_slot_name(name: Any) -> str:
    raw = str(name or "").strip().replace("-", "_")
    if not raw:
        return ""
    lowered = unicodedata.normalize("NFKD", raw)
    lowered = "".join(ch for ch in lowered if not unicodedata.combining(ch))
    lowered = lowered.lower()
    return SLOT_NAME_ALIASES.get(lowered, lowered)


def _normalize_number_token(token: str) -> str:
    token = unicodedata.normalize("NFKD", token)
    token = "".join(ch for ch in token if not unicodedata.combining(ch))
    token = token.lower()
    token = token.replace("-", "")
    return _WHITESPACE_RE.sub("", token)


def words_to_number(text: str) -> int | None:
    if not isinstance(text, str) or not text.strip():
        return None

    tokens = []
    for raw in re.split(r"[\s,/]+", text):
        token = _normalize_number_token(raw)
        if not token or token in {"و", "et", "and"}:
            continue
        tokens.append(token)

    if not tokens:
        return None

    total = 0
    current = 0
    matched = False
    for token in tokens:
        value = _NUMBER_WORDS.get(token)
        if value is None:
            continue
        matched = True
        if value == 100:
            current = max(1, current) * 100
        elif value == 1000:
            total += max(1, current) * 1000
            current = 0
        else:
            current += value

    if not matched:
        return None
    return total + current


def canonicalize_slot_value(name: str, value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip()
    if value in (None, "", []):
        return value

    if name == "num_client":
        digits = _NON_DIGIT_RE.sub("", str(value))
        if digits:
            return str(int(digits))
        return str(value)

    if name == "order_id":
        text = str(value).strip()
        if text.isdigit():
            return text
        return text.upper()

    if name == "reference":
        return str(value).replace(" ", "")

    if name in {"quantity", "axis", "od_axis", "og_axis", "diameter"}:
        digits = _NON_DIGIT_RE.sub("", str(value))
        if digits:
            return digits

    if name in {"od_sphere", "od_cyl", "og_sphere", "og_cyl"}:
        text = str(value).replace(",", ".").strip()
        try:
            numeric = float(text)
        except ValueError:
            return value
        return f"{numeric:+.2f}"

    if name == "addition":
        text = str(value).replace(",", ".").strip().lstrip("+")
        try:
            numeric = float(text)
        except ValueError:
            return value
        return f"{numeric:.2f}"

    if name == "priority":
        lowered = str(value).strip().lower()
        if lowered in {"urgence", "urgent", "illico"}:
            return "illico"
        if lowered in {"import", "commande import"}:
            return "import"
        if lowered in {"normal", "normale"}:
            return "normale"
        return lowered

    return value


def canonicalize_slots(payload: dict[str, Any] | None) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    if not isinstance(payload, dict):
        return normalized

    for raw_key, raw_value in payload.items():
        key = canonicalize_slot_name(raw_key)
        if not key or raw_value in (None, "", []):
            continue

        if key in {"od", "og"} and isinstance(raw_value, dict):
            prefix = key
            for nested_key, nested_value in raw_value.items():
                nested_name = OPTICAL_SLOT_ALIASES.get(
                    canonicalize_slot_name(nested_key),
                    canonicalize_slot_name(nested_key),
                )
                final_name = f"{prefix}_{nested_name}"
                normalized[final_name] = canonicalize_slot_value(final_name, nested_value)
            continue

        normalized[key] = canonicalize_slot_value(key, raw_value)

    return normalized
