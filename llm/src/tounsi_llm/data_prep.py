"""
Dataset preparation utilities for:
  1. dataset download
  2. self-supervised language adaptation
  3. supervised dialogue fine-tuning
  4. DPO preference alignment
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .config import (
    CFG,
    CONFIG_DIR,
    DOMAIN_CFG,
    FEW_SHOTS_CFG,
    HISTORY_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    SEED,
    logger,
    resolve_project_path,
)
from .data_sources import (
    dataset_spec_map,
    detect_script,
    download_configured_datasets as _download_configured_datasets,
    extract_text_candidates as extract_text_candidates_generic,
    looks_tunisian,
    normalize_for_dedup,
    normalize_text,
)
from .domain_utils import canonicalize_intent, normalize_messages
from .domain_utils import canonicalize_slots
from .rag_assets import load_delivery_rag_entries, load_lens_rag_entries


FUSHA_MARKERS = [
    "إليك",
    "يمكنني مساعدتك",
    "بالتأكيد",
    "يرجى",
    "يسعدني",
    "تفضل",
    "عذرا",
    "لا تتردد",
    "كيف يمكنني",
    "بكل سرور",
    "هل تريد",
    "هل ترغب",
]

MOROCCAN_MARKERS = [
    "ديال",
    "ديالي",
    "ديالك",
    "ديالكم",
    "كاين",
    "كاينة",
    "كاينين",
    "مزيان",
    "دابا",
    "فاش",
    "واخا",
    "بزاف",
    "هادا",
    "هادي",
    "هادشي",
    "عافاك",
    "غادي",
    "حيت",
]

TOUNSI_MARKERS = [
    "متاع",
    "فمّا",
    "فما",
    "باهي",
    "برشا",
    "قداش",
    "وين",
    "وقتاش",
    "شنوة",
    "نحب",
    "نجم",
    "عسلامة",
    "يعيشك",
]

_TOXIC_MARKERS = {
    "سبان",
    "سب",
    "إرهاب",
    "ارهاب",
    "حقير",
    "حيوان",
    "نكره",
    "نكرهو",
    "meskhou",
    "msakh",
    "terror",
    "terrorist",
    "hate",
    "raciste",
    "racism",
}
_TOXIC_MARKERS = {
    normalize_for_dedup(marker)
    for marker in _TOXIC_MARKERS
    if normalize_for_dedup(marker)
}

_CHINESE_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


_RAW_SYSTEM_PROMPTS = DOMAIN_CFG.get("system_prompts", {})
SYSTEM_PROMPTS = {
    str(key).strip().lower().replace("-", "_"): str(value).strip()
    for key, value in (_RAW_SYSTEM_PROMPTS.items() if isinstance(_RAW_SYSTEM_PROMPTS, dict) else [])
    if str(value).strip()
}

SYSTEM_PROMPT = SYSTEM_PROMPTS.get(
    "default",
    str(
        DOMAIN_CFG.get(
            "system_prompt",
            "إنت agent تخدم في مركز نداء. جاوب بطبيعة وبلهجة تونسية كي يلزم.",
        )
    ).strip(),
)
SYSTEM_PROMPT_ARABIC = SYSTEM_PROMPTS.get("arabic", SYSTEM_PROMPT)
SYSTEM_PROMPT_ARABIZI = SYSTEM_PROMPTS.get("arabizi", SYSTEM_PROMPT)
SYSTEM_PROMPT_CODE_SWITCH = SYSTEM_PROMPTS.get("code_switch", SYSTEM_PROMPT)
SYSTEM_GENERAL = "تحكي بالتونسي ديما. جاوب بطريقة طبيعية وودودة."
_KNOWN_SYSTEM_PROMPTS = {
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_ARABIC,
    SYSTEM_PROMPT_ARABIZI,
    SYSTEM_PROMPT_CODE_SWITCH,
    SYSTEM_GENERAL,
    str(DOMAIN_CFG.get("humanized_prompt", "")).strip(),
}
_DOMAIN_REQUIRED_SOURCES = {
    "commandes_intents",
    "commandes_augmented",
    "multiturn_collection",
    "slot_bootstrap",
    "rag_grounded",
    "few_shots",
    "approved_learning",
}
_DOMAIN_REQUIRED_SOURCES |= {
    str(item).strip()
    for item in DOMAIN_CFG.get("sft_domain_bypass_sources", [])
    if str(item).strip()
}
_SFT_RELAXED_STYLE_SOURCES = {
    str(item).strip()
    for item in DOMAIN_CFG.get("sft_relaxed_style_sources", [])
    if str(item).strip()
}
_SFT_NO_FILTER_SOURCES = {
    str(item).strip()
    for item in DOMAIN_CFG.get("sft_no_filter_sources", [])
    if str(item).strip()
}
_DOMAIN_STRONG_MARKERS = {
    "commande",
    "كوموند",
    "suivi",
    "statut",
    "num client",
    "code client",
    "dossier",
    "livraison",
    "agence",
    "secteur",
    "creneau",
    "créneau",
    "reference",
    "référence",
    "catalogue",
    "catalog",
    "verre",
    "lens",
    "stock",
    "dispo",
    "disponibilite",
    "availability",
    "fabrication",
    "opticien",
    "optique",
    "varilux",
    "crizal",
    "prevencia",
    "stellest",
    "eyezen",
    "bifocal",
    "sphere",
    "sphère",
    "cyl",
    "cylindre",
    "axe",
    "addition",
    "diametre",
    "diamètre",
}
_DOMAIN_WEAK_MARKERS = {"client", "prix", "سوم", "product", "produit", "patient", "rdv", "rendez vous"}
_GENERIC_QUERY_MARKERS = {
    "قداش",
    "شنوة",
    "وقتاش",
    "وين",
    "نحب",
    "نجم",
    "سلام",
    "عسلامة",
    "مرحبا",
    "bonjour",
    "hello",
    "allo",
}
_NORMALIZED_GENERIC_QUERY_MARKERS = {
    normalize_for_dedup(marker)
    for marker in _GENERIC_QUERY_MARKERS
    if normalize_for_dedup(marker)
}
for item in DOMAIN_CFG.get("business_keywords", []):
    marker = normalize_for_dedup(str(item))
    if marker:
        _DOMAIN_STRONG_MARKERS.add(marker)
for intent_name, values in DOMAIN_CFG.get("intent_keywords", {}).items():
    if canonicalize_intent(intent_name) in {"greeting", "thanks"}:
        continue
    if not isinstance(values, list):
        continue
    for item in values:
        marker = normalize_for_dedup(str(item))
        if marker:
            if marker in _NORMALIZED_GENERIC_QUERY_MARKERS:
                continue
            if marker in {"client", "prix", "سوم", "price", "product", "produit"}:
                _DOMAIN_WEAK_MARKERS.add(marker)
            else:
                _DOMAIN_STRONG_MARKERS.add(marker)
_DOMAIN_STRONG_MARKERS = {normalize_for_dedup(marker) for marker in _DOMAIN_STRONG_MARKERS if normalize_for_dedup(marker)}
_DOMAIN_WEAK_MARKERS = {normalize_for_dedup(marker) for marker in _DOMAIN_WEAK_MARKERS if normalize_for_dedup(marker)}
_DOMAIN_STRONG_MARKERS -= _NORMALIZED_GENERIC_QUERY_MARKERS
_DOMAIN_WEAK_MARKERS -= _NORMALIZED_GENERIC_QUERY_MARKERS
_STRICT_DPO_DOMAIN_MARKERS = {
    "commande",
    "كوموند",
    "suivi",
    "statut",
    "livraison",
    "agence",
    "secteur",
    "creneau",
    "créneau",
    "reference",
    "référence",
    "catalog",
    "catalogue",
    "verre",
    "lens",
    "stock",
    "dispo",
    "disponibilite",
    "availability",
    "fabrication",
    "opticien",
    "optique",
    "varilux",
    "crizal",
    "prevencia",
    "stellest",
    "eyezen",
    "bifocal",
    "sphere",
    "sphère",
    "cyl",
    "cylindre",
    "axe",
    "addition",
    "diametre",
    "diamètre",
    "illico",
    "anti reflet",
    "anti-reflet",
    "progressive",
    "precal",
    "mineral",
    "minéral",
    "orma",
}
MEMORY_CFG = DOMAIN_CFG.get("memory", {})
APPROVED_LEARNING_PATH = resolve_project_path(
    MEMORY_CFG.get("learning_buffer_path", HISTORY_DIR / "learning_buffer.jsonl")
)
APPROVED_FEEDBACK_DPO_PATH = resolve_project_path(
    MEMORY_CFG.get("approved_dpo_feedback_path", HISTORY_DIR / "feedback_dpo.jsonl")
)
COMMANDES_INTENTS_PATH = CONFIG_DIR / "commandes_intents.jsonl"
_NUM_CLIENT_TOKEN_RE = re.compile(r"\b(?:CLI|CLT|CLIENT|CUST)[-_]?\s*0*\d{3,6}\b", re.IGNORECASE)
_OPTION_MARKER_RE = re.compile(r"\b[A-D]\.\s")
_ORDER_ID_RE = re.compile(r"\bCMD[-_ ]?\d{4,}\b", re.IGNORECASE)
_CLIENT_REFERENCE_RE = re.compile(
    r"\b(?:num[_ ]?client|code[_ ]?client|client|cli|clt|cust)[-_ ]?\s*0*\d{3,6}\b",
    re.IGNORECASE,
)
_OPTICS_MARKER_RE = re.compile(
    r"\b(?:od|og|sphere|sphère|cyl|cylindre|axe|addition|diam[eè]tre)\b",
    re.IGNORECASE,
)

_AUGMENTED_INTENT_CACHE: dict[int, list[list[dict[str, str]]]] = {}
_SLOT_BOOTSTRAP_CACHE: dict[int, list[list[dict[str, str]]]] = {}
_RAG_GROUNDED_CACHE: dict[tuple[int, int], list[list[dict[str, str]]]] = {}
_RAG_SELF_SUP_CACHE: list[str] | None = None

_INSTRUCTIONAL_NOISE_PATTERNS = [
    "ترجم",
    "translate this to",
    "translate to tunisian arabic",
    "متعدد الخيارات",
    "قرا هاذا النص",
    "انطلاقا من",
    "على حساب الإحصاء",
    "الجواب:",
    "هذا المقال",
    "يمكنك",
    "أيضاً",
    "تماماً",
    "سنطرق",
    "ما هو موقع",
]
_ROLE_PREFIX_PATTERNS = ["System:", "User:", "Assistant:"]


def _contains_code_switch_business_terms(text: str) -> bool:
    normalized = normalize_for_dedup(text)
    if not normalized:
        return False

    static_terms = {
        "commande",
        "suivi",
        "statut",
        "livraison",
        "verre",
        "optique",
        "num client",
        "code client",
        "dossier",
        "rdv",
        "rendez vous",
        "urgent",
        "illico",
        "progressive",
        "varilux",
        "crizal",
        "prevencia",
        "stellest",
        "bifocal",
    }
    business_tokens = {
        normalize_for_dedup(str(item))
        for item in DOMAIN_CFG.get("business_keywords", [])
        if str(item).strip()
    }
    for token in static_terms | business_tokens:
        if token and token in normalized:
            return True
    return False


def _domain_signal_score(text: str) -> int:
    normalized = normalize_for_dedup(text)
    if not normalized:
        return 0

    score = 0
    if _ORDER_ID_RE.search(text):
        score += 3
    if _NUM_CLIENT_TOKEN_RE.search(text) or _CLIENT_REFERENCE_RE.search(text):
        score += 2
    if _OPTICS_MARKER_RE.search(text):
        score += 2
    if "sivo" in normalized or "essilor" in normalized:
        score += 2

    for marker in _DOMAIN_STRONG_MARKERS:
        if marker and marker in normalized:
            score += 2
    for marker in _DOMAIN_WEAK_MARKERS:
        if marker and marker in normalized:
            score += 1
    return score


def conversation_domain_ok(messages: list[dict[str, str]], source: str = "unknown") -> bool:
    if source in _DOMAIN_REQUIRED_SOURCES:
        return True

    joined = " ".join(
        normalize_text(message.get("content", ""))
        for message in normalize_messages(messages)
        if message.get("role") in {"user", "assistant"}
    )
    return _domain_signal_score(joined) >= 2


def _self_sup_language_ok(text: str, *, strict_tunisian: bool) -> bool:
    if _contains_toxic_or_abusive_content(text):
        return False
    if _contains_instructional_noise(text):
        return False
    if strict_tunisian:
        return looks_tunisian(text, strict=True)

    if looks_tunisian(text, strict=False):
        return True

    script = detect_script(text)
    if script in {"arabizi", "mixed"}:
        return True
    if script == "latin" and _contains_code_switch_business_terms(text):
        return True
    return False


def _contains_instructional_noise(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    lowered = normalized.lower()
    if any(pattern in lowered for pattern in _INSTRUCTIONAL_NOISE_PATTERNS):
        return True
    if any(lowered.startswith(pattern.lower()) for pattern in _ROLE_PREFIX_PATTERNS):
        return True
    return bool(_OPTION_MARKER_RE.search(normalized))


def _contains_toxic_or_abusive_content(text: str) -> bool:
    if not CFG.block_toxic_content:
        return False

    normalized = normalize_for_dedup(text)
    if not normalized:
        return False
    return any(marker in normalized for marker in _TOXIC_MARKERS)


def _load_commandes_intent_conversations(path: Path = COMMANDES_INTENTS_PATH) -> list[list[dict[str, str]]]:
    """Load operator intent examples from config JSONL and normalize schema variants."""
    conversations: list[list[dict[str, str]]] = []
    if not path.exists():
        return conversations

    seen: set[str] = set()
    for row in _load_jsonl(path):
        intent = canonicalize_intent(row.get("intent"))

        user_text = row.get("user") or row.get("client") or row.get("opticien")
        assistant_text = row.get("assistant") or row.get("agent")
        if not isinstance(user_text, str) or not user_text.strip():
            continue
        if not isinstance(assistant_text, str) or not assistant_text.strip():
            continue

        dedup_key = normalize_for_dedup(f"{intent}||{user_text}||{assistant_text}")
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        conversations.append(
            format_sft_conversation(
                [
                    {"role": "user", "content": user_text.strip()},
                    {"role": "assistant", "content": assistant_text.strip()},
                ]
            )
        )

    logger.info("Loaded %d config intent conversations from %s", len(conversations), path)
    return conversations


def _strip_num_client_reference(text: str, num_client: str | None) -> str:
    stripped = _NUM_CLIENT_TOKEN_RE.sub(" ", str(text or ""))
    if num_client:
        stripped = re.sub(rf"\b0*{re.escape(str(num_client))}\b", " ", stripped)
    stripped = re.sub(r"\b(?:معاك|num client|code client|client)\b", " ", stripped, flags=re.IGNORECASE)
    return normalize_text(stripped)


def _load_multiturn_collection_conversations(path: Path = COMMANDES_INTENTS_PATH) -> list[list[dict[str, str]]]:
    conversations: list[list[dict[str, str]]] = []
    ask_templates = DOMAIN_CFG.get("response_templates", {}).get("ask_num_client", [])
    fallback_ask = "عسلامة، قبل كل شي عطيني num client باش نثبت dossier متاعك."
    seen: set[str] = set()

    for row in _load_jsonl(path):
        slots = row.get("slots", {}) if isinstance(row.get("slots"), dict) else {}
        num_client = slots.get("codeClient") or slots.get("num_client") or slots.get("customer_id")
        user_text = row.get("user") or row.get("client") or row.get("opticien")
        assistant_text = row.get("assistant") or row.get("agent")
        if not (
            isinstance(user_text, str)
            and user_text.strip()
            and isinstance(assistant_text, str)
            and assistant_text.strip()
            and num_client
        ):
            continue

        initial_turn = _strip_num_client_reference(user_text, str(num_client))
        if len(initial_turn) < 8:
            continue

        dedup_key = normalize_for_dedup(f"{initial_turn}||{num_client}||{assistant_text}")
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        ask_num_client = ask_templates[len(conversations) % len(ask_templates)] if ask_templates else fallback_ask
        conversations.append(
            format_sft_conversation(
                [
                    {"role": "user", "content": initial_turn},
                    {"role": "assistant", "content": ask_num_client},
                    {"role": "user", "content": str(num_client)},
                    {"role": "assistant", "content": assistant_text.strip()},
                ]
            )
        )

    logger.info("Loaded %d synthetic multi-turn collection conversations", len(conversations))
    return conversations


def _load_few_shot_conversations() -> list[list[dict[str, str]]]:
    conversations: list[list[dict[str, str]]] = []
    seen: set[str] = set()
    for row in FEW_SHOTS_CFG:
        intent = canonicalize_intent(row.get("intent"))
        user_text = row.get("user") or row.get("client") or row.get("opticien")
        assistant_text = row.get("assistant") or row.get("agent")
        if not isinstance(user_text, str) or not user_text.strip():
            continue
        if not isinstance(assistant_text, str) or not assistant_text.strip():
            continue

        dedup_key = normalize_for_dedup(f"{intent}||{user_text}||{assistant_text}")
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        conversations.append(
            format_sft_conversation(
                [
                    {"role": "user", "content": user_text.strip()},
                    {"role": "assistant", "content": assistant_text.strip()},
                ]
            )
        )
    logger.info("Loaded %d few-shot conversations from config", len(conversations))
    return conversations


def _slot_value(slots: dict[str, Any], *names: str, default: str = "") -> str:
    for name in names:
        value = slots.get(name)
        if value not in (None, "", []):
            return str(value)
    return default


def _optics_phrase(slots: dict[str, Any]) -> str:
    parts: list[str] = []
    for eye in ["od", "og"]:
        sphere = slots.get(f"{eye}_sphere")
        cyl = slots.get(f"{eye}_cyl")
        axis = slots.get(f"{eye}_axis")
        if sphere or cyl or axis:
            eye_label = eye.upper()
            eye_parts = [eye_label]
            if sphere:
                eye_parts.append(f"sphere {sphere}")
            if cyl:
                eye_parts.append(f"cyl {cyl}")
            if axis:
                eye_parts.append(f"axe {axis}")
            parts.append(" ".join(eye_parts))
    if slots.get("addition"):
        parts.append(f"addition {slots['addition']}")
    return " | ".join(parts) if parts else "OD sphere -0.50 cyl +1.00 axe 90"


def _build_augmented_intent_conversations(
    path: Path = COMMANDES_INTENTS_PATH,
    variants_per_row: int | None = None,
) -> list[list[dict[str, str]]]:
    effective_variants = max(1, int(variants_per_row or CFG.sft_augmented_variants_per_row))
    cached = _AUGMENTED_INTENT_CACHE.get(effective_variants)
    if cached is not None:
        return list(cached)

    intent_templates: dict[str, dict[str, list[str]]] = {
        "order_tracking": {
            "user": [
                "aslema, num client {num_client}, nheb suivi bon {order_id}. livraison l'agence weslet wala mazelt?",
                "سلام، عندي client {num_client} w commande {order_id}. tnajem taatini statut + délai taqribi lel agence?",
                "bonjour, n7eb netabba3 {order_id} mta3 {num_client}, surtout timing mta3 livraison secteur {secteur}.",
            ],
            "assistant": [
                "مرحبا، ثبت num client {num_client}. تو نعمل check l {order_id} ونعطيك statut + fenêtre livraison تقريبية حسب agence/secteur.",
                "وصلت، باش نثبت bon {order_id} للclient {num_client}. ما نخممش في délai: نعطيك كان المعطيات المأكدة من النظام.",
                "أكيد، num client {num_client} واضح. بعد التثبت من {order_id} نرجعلك بالstatut ووقت تقريبي للتوصيل للوكالة.",
            ],
        },
        "create_order": {
            "user": [
                "aslema, num client {num_client}, nheb na3di commande {product} {material} {treatment} diamètre {diameter}, {optics}.",
                "سلام، {num_client} نحب create_order: {product} sur {material}, coating {treatment}, couleur {color}, diamètre {diameter}.",
                "bonjour, pour client {num_client} n7eb ordre {product} priorité {priority}, {optics}, وتسليم agence {agence}.",
            ],
            "assistant": [
                "حاضر، ثبتّ num client {num_client}. نراجع: {product} / {material} / {treatment} diamètre {diameter}. نزيد نثبت القيم {optics} قبل validation finale.",
                "وصلت، creation commande جاهزة مبدئيا. باش نأكد slots المهمة واحدة بواحدة: produit, traitement, diamètre والقيم البصرية.",
                "مريقل، نجم نكمّل draft للكوموند متاع {num_client}. ما نبعثش نهائي إلا بعد recap واضح للـ OD/OG والتسليم للوكالة.",
            ],
        },
        "delivery_schedule": {
            "user": [
                "planning livraison agence {agence}, secteur {secteur}. next slot?",
                "سلام، pour {agence} / {secteur}, وقتاش يجي camion؟",
                "nheb window taqribi lel livraison fi {agence} secteur {secteur} ba3d {delivery_slot}.",
            ],
            "assistant": [
                "بالنسبة لـ {agence} قطاع {secteur}، نجم نعطيك window تقريبية من planning. التسليم ديما للوكالة مش للمريض.",
                "وصلت. نثبت creneaux mta3 {agence}/{secteur} ونعطيك prochain passage approximatif حسب الجدول.",
                "أكيد، نرجعلك بوقت تقريبي للتسليم في {agence} ({secteur}) مع nearest slot.",
            ],
        },
        "reference_confirmation": {
            "user": [
                "tnajem tconfirmili code verre {reference} ?",
                "سلام، référence {reference} تخص انو produit بالضبط؟",
                "aslema, code {lens_code} صحيح ولا لا قبل ما نعدي commande؟",
            ],
            "assistant": [
                "نثبتلك référence {reference} من catalog/RAG ونرجعلك بالproduit الصحيح بلا تخمين.",
                "أكيد، تو نعمل confirmation للكود {reference}. إذا مش واضح نقلك صراحة ما نجمش نأكد.",
                "وصلت، نتحقق من code {lens_code} ونمدك بالمعلومة الدقيقة قبل ما تكمل الكوموند.",
            ],
        },
        "availability_inquiry": {
            "user": [
                "dispo {product} {material} diamètre {diameter} fi secteur {secteur}?",
                "سلام، nheb n3ref disponibilité {reference} قبل ما نعمل commande.",
                "aslema, code {lens_code} موجود stock wela fabrication?",
            ],
            "assistant": [
                "باش نثبت disponibilité متاع {product}/{material} بلا ما نخمن في stock. نرجعلك بمعطيات مؤكدة فقط.",
                "وصلت، نراجع référence {reference} ونعطيك réponse دقيقة: disponible, sur commande, أو délai fabrication.",
                "أكيد، نتحقق من code {lens_code} ونوضحلك availability بطريقة مهنية وواضحة.",
            ],
        },
        "get_num_client": {
            "user": [
                "aslema, nheb suivi commande ama ma andich num client tawa.",
                "bonjour, عندي demande urgente على commande, chnowa أول معلومة تلزم؟",
                "سلام، نحب نعدي كوموند جديدة, najem n9olek details direct?",
            ],
            "assistant": [
                "مرحبا، أول خطوة تعطيني num client باش نفتح dossier الصحيح وبعدها نكمل معاك الطلب.",
                "أكيد، قبل أي suivi/create_order يلزم num_client. بعد ما تمدهولي نكمل مباشرة.",
                "حاضر، مدلي num_client الأول وبعد نطلب منك كان slots الضرورية حسب نيتك.",
            ],
        },
    }

    conversations: list[list[dict[str, str]]] = []
    seen: set[str] = set()

    for index, row in enumerate(_load_jsonl(path)):
        intent = canonicalize_intent(row.get("intent"))
        template_bank = intent_templates.get(intent)
        if not template_bank:
            continue

        slots = canonicalize_slots(row.get("slots", {}) if isinstance(row.get("slots"), dict) else {})
        num_client = _slot_value(slots, "num_client", default=str(3000 + (index % 6500)))
        order_id = _slot_value(slots, "order_id", default=f"CMD26{index:08d}")
        product = _slot_value(slots, "product", default="Varilux liberty short")
        material = _slot_value(slots, "material", default="orma blanc")
        treatment = _slot_value(slots, "treatment", default="crizal prevencia")
        color = _slot_value(slots, "color", default="gris")
        diameter = _slot_value(slots, "diameter", default="65")
        reference = _slot_value(slots, "reference", default="25YXSU")
        lens_code = _slot_value(slots, "lens_code", default=reference)
        agence = _slot_value(slots, "agence", default="Aouina")
        secteur = _slot_value(slots, "secteur", default="marsa / lac")
        delivery_slot = _slot_value(slots, "delivery_slot", default="12:00")
        priority = _slot_value(slots, "priority", default="normale")
        optics = _optics_phrase(slots)

        context = {
            "num_client": num_client,
            "order_id": order_id,
            "product": product,
            "material": material,
            "treatment": treatment,
            "color": color,
            "diameter": diameter,
            "reference": reference,
            "lens_code": lens_code,
            "agence": agence,
            "secteur": secteur,
            "delivery_slot": delivery_slot,
            "priority": priority,
            "optics": optics,
        }

        max_variants = min(effective_variants, len(template_bank["user"]), len(template_bank["assistant"]))
        for offset in range(max_variants):
            user_template = template_bank["user"][(index + offset) % len(template_bank["user"])]
            assistant_template = template_bank["assistant"][(index + offset) % len(template_bank["assistant"])]
            user_text = user_template.format(**context).strip()
            assistant_text = assistant_template.format(**context).strip()

            dedup_key = normalize_for_dedup(f"{intent}|{user_text}|{assistant_text}")
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            conversations.append(
                format_sft_conversation(
                    [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": assistant_text},
                    ]
                )
            )

    logger.info("Built %d augmented intent conversations (variants_per_row=%d)", len(conversations), effective_variants)
    _AUGMENTED_INTENT_CACHE[effective_variants] = conversations
    return list(conversations)


def _generate_slot_rich_bootstrap_conversations(limit: int | None = None) -> list[list[dict[str, str]]]:
    effective_limit = max(1, int(limit or CFG.sft_slot_bootstrap_limit))
    cached = _SLOT_BOOTSTRAP_CACHE.get(effective_limit)
    if cached is not None:
        return list(cached)

    clients = ["2250", "2868", "3144", "4022", "4771", "5206", "5931", "6088", "7310", "8442", "9055"]
    orders = ["CMD2612345678", "CMD2623456789", "CMD2634567890", "CMD2645678901", "CMD2656789012"]
    products = ["Varilux liberty short", "progressive premium", "top 28 bifocal", "Eyezen initial", "photochromique"]
    materials = ["orma blanc", "1.60", "1.67", "Airwear", "minéral"]
    treatments = ["crizal prevencia", "anti-reflet", "Transitions", "Sapphire", "sans traitement"]
    colors = ["gris", "marron dégradé", "blanc", "bleu dégradé", "brun"]
    diameters = ["65", "67", "70", "72", "75"]
    agencies = ["Aouina", "Ben Arous", "Sousse", "Nabeul", "Monastir"]
    sectors = ["marsa / lac", "mourouj", "trocadero", "nabeul centre ville", "centre ville"]
    refs = ["25YXSU", "366C10", "16 83", "39D7AQ", "41FBK0"]

    conversations: list[list[dict[str, str]]] = []
    seen: set[str] = set()

    for idx in range(effective_limit):
        num_client = clients[idx % len(clients)]
        order_id = orders[idx % len(orders)]
        product = products[idx % len(products)]
        material = materials[idx % len(materials)]
        treatment = treatments[idx % len(treatments)]
        color = colors[idx % len(colors)]
        diameter = diameters[idx % len(diameters)]
        agency = agencies[idx % len(agencies)]
        sector = sectors[idx % len(sectors)]
        reference = refs[idx % len(refs)]

        mode = idx % 5
        if mode == 0:
            user_text = (
                f"aslema, num client {num_client}, nheb create_order {product} {material} {treatment} "
                f"diamètre {diameter}, OD sphere -1.25 cyl +0.50 axe 90, OG sphere -0.75 cyl +0.25 axe 80, addition +1.50."
            )
            assistant_text = (
                f"حاضر، ثبت num_client {num_client}. recap: {product} / {material} / {treatment} / diamètre {diameter}. "
                "نأكد OD/OG + addition وبعدها نكمل validation finale."
            )
        elif mode == 1:
            user_text = (
                f"bonjour, suivi commande {order_id} pour client {num_client}. livraison lel agence {agency} secteur {sector} وقتاش؟"
            )
            assistant_text = (
                f"وصلت، باش نثبت {order_id} متاع {num_client} ونعطيك statut + délai تقريبي للتسليم في {agency}/{sector}."
            )
        elif mode == 2:
            user_text = f"planning livraison {agency} secteur {sector}, next slot ba3d 12:00?"
            assistant_text = (
                f"بالنسبة لـ {agency} ({sector}) نجم نعطيك fenêtre تقريبية للمرور القادم حسب planning الوكالة."
            )
        elif mode == 3:
            user_text = f"tnajem tconfirmili référence {reference} قبل ما نعدي commande client {num_client}?"
            assistant_text = (
                f"أكيد، نثبت الكود {reference} من catalogue/RAG ونرجعلك بالproduit الصحيح بلا أي تخمين."
            )
        else:
            user_text = (
                f"disponibilité {product} {material} diamètre {diameter} couleur {color} pour client {num_client}?"
            )
            assistant_text = (
                "نراجع disponibilité من المصادر المؤكدة (catalog + tools) ونرجعلك بجواب واضح: disponible, sur commande, ولا délai fabrication."
            )

        dedup_key = normalize_for_dedup(f"{user_text}|{assistant_text}")
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        conversations.append(
            format_sft_conversation(
                [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ]
            )
        )

    logger.info("Built %d slot bootstrap conversations (limit=%d)", len(conversations), effective_limit)
    _SLOT_BOOTSTRAP_CACHE[effective_limit] = conversations
    return list(conversations)


def _simplify_rag_material(value: str) -> str:
    text = normalize_text(str(value or ""))
    text = re.sub(r"\s*\([^)]*\)", "", text).strip(" -/")
    return text


def _build_rag_grounded_conversations(
    max_delivery: int | None = None,
    max_lens: int | None = None,
) -> list[list[dict[str, str]]]:
    effective_delivery = max(1, int(max_delivery or CFG.sft_rag_delivery_limit))
    effective_lens = max(1, int(max_lens or CFG.sft_rag_lens_limit))
    cache_key = (effective_delivery, effective_lens)
    cached = _RAG_GROUNDED_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)

    conversations: list[list[dict[str, str]]] = []
    seen: set[str] = set()

    delivery_entries = [entry for entry in load_delivery_rag_entries() if entry.get("secteur")]
    for index, entry in enumerate(delivery_entries[:effective_delivery]):
        agence = str(entry.get("agence", "")).strip()
        secteur = str(entry.get("secteur", "")).strip()
        slots = [str(item).strip() for item in entry.get("tous_creneaux", []) if str(item).strip()]
        slots_text = ", ".join(slots[:3]) if slots else str(entry.get("premier_creneau") or "")
        next_slot = str(entry.get("premier_creneau") or (slots[0] if slots else ""))
        nb_livraisons = entry.get("nb_livraisons_jour") or len(slots) or 1
        order_id = f"CMD26{index:08d}"
        num_client = str(3000 + index)

        delivery_pairs = [
            (
                f"planning livraison agence {agence}, secteur {secteur}. prochain creneau wa9tech?",
                f"Pour {agence} / {secteur}, fama {nb_livraisons} passages fi nhar. Les creneaux connus houma {slots_text}. Prochain passage approximatif: {next_slot}.",
            ),
            (
                f"aslema, lel agence {agence} secteur {secteur}, livraison l'opticien ta3mel fi anou wa9t?",
                f"Livraison SIVO temchi ll agence mta3 l opticien, moch ll patient. Fi {agence} / {secteur}, el fenetre el ma3roufa howa {slots_text}.",
            ),
            (
                f"bonjour, suivi commande {order_id} client {num_client}, surtout livraison lel agence {agence} secteur {secteur}",
                f"Nethabet statut commande {order_id}, w ken tal9a livraison lel agence {agence}/{secteur}, n9olek fenetre approximative {slots_text} sans promesse exacte.",
            ),
        ]

        for user_text, assistant_text in delivery_pairs:
            dedup_key = normalize_for_dedup(f"{user_text}|{assistant_text}")
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            conversations.append(
                format_sft_conversation(
                    [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": assistant_text},
                    ]
                )
            )

    lens_entries = [entry for entry in load_lens_rag_entries() if entry.get("code") and entry.get("name")]
    for entry in lens_entries[:effective_lens]:
        code = str(entry.get("code", "")).strip().upper()
        name = str(entry.get("name", "")).strip()
        material = _simplify_rag_material(str(entry.get("material", ""))) or "Orma"
        geometry = str(entry.get("geometry", "")).strip() or "verre"
        diameter = str(entry.get("diameter", "")).strip() or "standard"
        photo = str(entry.get("photochromic", "")).strip()
        photo_part = f", photo {photo}" if photo and photo.lower() not in {"non", "none", "no"} else ""

        lens_pairs = [
            (
                f"tnajem tconfirmili code verre {code} ?",
                f"Code {code} correspond a {name}, geometrie {geometry}, matiere {material}, diametre {diameter}{photo_part}. Nethabet men catalogue officiel sans تخمين.",
            ),
            (
                f"aslema, dispo {code} qbal ma na3di commande?",
                f"Nethabet code {code} men catalogue officiel: {name}, matiere {material}, diametre {diameter}. Pour dispo finale, يلزم confirmation stock/backoffice.",
            ),
            (
                f"bonjour, pour create_order nheb reference {code} {name}",
                f"Mriguel, nethabet référence {code}: {name}. Qbal validation finale, lazem num client w باقي slots optiques si nécessaires.",
            ),
        ]

        for user_text, assistant_text in lens_pairs:
            dedup_key = normalize_for_dedup(f"{user_text}|{assistant_text}")
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            conversations.append(
                format_sft_conversation(
                    [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": assistant_text},
                    ]
                )
            )

    logger.info(
        "Built %d RAG-grounded conversations (delivery=%d, lens=%d)",
        len(conversations),
        effective_delivery,
        effective_lens,
    )
    _RAG_GROUNDED_CACHE[cache_key] = conversations
    return list(conversations)


def _load_rag_grounded_self_sup_texts(
    max_delivery: int = 120,
    max_lens: int = 900,
) -> list[str]:
    global _RAG_SELF_SUP_CACHE
    if _RAG_SELF_SUP_CACHE is not None:
        return list(_RAG_SELF_SUP_CACHE)

    texts: list[str] = []
    seen: set[str] = set()

    def add_text(value: str) -> None:
        text = normalize_text(value)
        if not text or len(text) < 12:
            return
        dedup_key = normalize_for_dedup(text)
        if dedup_key in seen:
            return
        seen.add(dedup_key)
        texts.append(text)

    for entry in load_delivery_rag_entries()[:max_delivery]:
        agence = str(entry.get("agence", "")).strip()
        secteur = str(entry.get("secteur", "")).strip()
        slots = [str(item).strip() for item in entry.get("tous_creneaux", []) if str(item).strip()]
        slots_text = ", ".join(slots[:3]) if slots else str(entry.get("premier_creneau") or "")
        if agence and secteur:
            add_text(
                f"fi planning SIVO, agence {agence} secteur {secteur} عندها livraison approximative fi {slots_text}."
            )
            add_text(
                f"livraison mta3 SIVO temchi ll agence {agence}, secteur {secteur}, moch ll patient final."
            )

    for entry in load_lens_rag_entries()[:max_lens]:
        code = str(entry.get("code", "")).strip().upper()
        name = str(entry.get("name", "")).strip()
        material = _simplify_rag_material(str(entry.get("material", "")))
        geometry = str(entry.get("geometry", "")).strip()
        diameter = str(entry.get("diameter", "")).strip()
        photo = str(entry.get("photochromic", "")).strip()
        if code and name:
            line = f"fi catalogue SIVO, code {code} houa {name}"
            if material:
                line += f", matiere {material}"
            if geometry:
                line += f", geometrie {geometry}"
            if diameter:
                line += f", diametre {diameter}"
            if photo and photo.lower() not in {"non", "none", "no"}:
                line += f", photo {photo}"
            line += "."
            add_text(line)

    logger.info("Loaded %d RAG-grounded self-sup texts", len(texts))
    _RAG_SELF_SUP_CACHE = texts
    return list(texts)


def _load_synthetic_self_sup_texts() -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()

    synthetic_sets = [
        _build_augmented_intent_conversations(variants_per_row=1),
        _generate_slot_rich_bootstrap_conversations(limit=120),
        _build_rag_grounded_conversations(max_delivery=60, max_lens=120),
    ]
    for conversations in synthetic_sets:
        for conversation in conversations:
            for message in conversation:
                if message.get("role") not in {"user", "assistant"}:
                    continue
                text = normalize_text(message.get("content", ""))
                if not text:
                    continue
                dedup_key = normalize_for_dedup(text)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                texts.append(text)

    logger.info("Loaded %d synthetic utterances for self-sup enrichment", len(texts))
    return texts


def _load_config_self_sup_texts() -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()

    def add_text(value: Any) -> None:
        text = normalize_text(str(value or ""))
        if not text or len(text) < 6 or _is_low_quality_text(text):
            return
        dedup_key = normalize_for_dedup(text)
        if dedup_key in seen:
            return
        seen.add(dedup_key)
        texts.append(text)

    for row in FEW_SHOTS_CFG:
        add_text(row.get("user") or row.get("client") or row.get("opticien"))
        add_text(row.get("assistant") or row.get("agent"))

    for row in _load_jsonl(COMMANDES_INTENTS_PATH):
        add_text(row.get("user") or row.get("client") or row.get("opticien"))
        add_text(row.get("assistant") or row.get("agent"))

    logger.info("Loaded %d config utterances for self-sup enrichment", len(texts))
    return texts


def _save_jsonl(ds, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in ds:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSONL line in %s", path)
    return rows


def download_configured_datasets(
    cache_dir: Path | None = None,
    dataset_specs: list[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    return _download_configured_datasets(cache_dir=cache_dir, dataset_specs_override=dataset_specs)


def download_tounsi_datasets(cache_dir: Path | None = None) -> dict[str, Path]:
    return download_configured_datasets(cache_dir=cache_dir)


def is_moroccan_or_algerian(text: str) -> bool:
    if not text:
        return False
    moroccan_score = sum(1 for marker in MOROCCAN_MARKERS if marker in text)
    tounsi_score = sum(1 for marker in TOUNSI_MARKERS if marker in text)
    return moroccan_score > 0 and tounsi_score <= moroccan_score


def is_clean_tounsi(text: str, max_fusha: int = 1) -> bool:
    if not text or len(text.strip()) < 5:
        return False
    if _CHINESE_RE.search(text):
        return False
    if _contains_instructional_noise(text):
        return False
    if sum(1 for marker in FUSHA_MARKERS if marker in text) >= max_fusha:
        return False
    if is_moroccan_or_algerian(text):
        return False
    return True


def compute_quality_stats(texts: list[str]) -> dict[str, Any]:
    total = len(texts)
    if total == 0:
        return {"total": 0}
    clean = sum(1 for text in texts if is_clean_tounsi(text))
    strict_clean = sum(1 for text in texts if looks_tunisian(text, strict=True))
    chinese = sum(1 for text in texts if _CHINESE_RE.search(text or ""))
    code_switch = 0
    latin_business = 0
    scripts: dict[str, int] = {}
    for text in texts:
        script = detect_script(text)
        scripts[script] = scripts.get(script, 0) + 1
        if script in {"mixed", "arabizi"}:
            code_switch += 1
        if script == "latin" and _contains_code_switch_business_terms(text):
            latin_business += 1

    marker_counts = {
        marker: sum(1 for text in texts if marker in (text or ""))
        for marker in FUSHA_MARKERS
    }
    marker_counts = {key: value for key, value in marker_counts.items() if value > 0}
    return {
        "total": total,
        "clean_tounsi": clean,
        "clean_pct": round(clean / total * 100, 1),
        "strict_clean_tounsi": strict_clean,
        "strict_clean_pct": round(strict_clean / total * 100, 1),
        "has_chinese": chinese,
        "code_switch_or_arabizi": code_switch,
        "code_switch_or_arabizi_pct": round(code_switch / total * 100, 1),
        "latin_business": latin_business,
        "latin_business_pct": round(latin_business / total * 100, 1),
        "scripts": scripts,
        "top_fusha_markers": dict(sorted(marker_counts.items(), key=lambda item: -item[1])[:10]),
    }


def _choose_system_prompt_for_conversation(messages: list[dict[str, str]], fallback_system: str) -> str:
    joined = " ".join(
        normalize_text(message.get("content", ""))
        for message in messages
        if message.get("role") in {"user", "assistant"}
    )
    detected = detect_script(joined)
    if detected == "mixed":
        return SYSTEM_PROMPT_CODE_SWITCH
    if detected == "arabizi":
        return SYSTEM_PROMPT_ARABIZI
    if detected == "arabic":
        return SYSTEM_PROMPT_ARABIC
    return fallback_system


def format_sft_conversation(messages: list[dict[str, Any]], system: str = SYSTEM_PROMPT) -> list[dict[str, str]]:
    existing_system = None
    cleaned: list[dict[str, str]] = []
    for message in normalize_messages(messages):
        role = message["role"]
        content = message["content"]
        if role == "system":
            existing_system = content
            continue
        cleaned.append({"role": role, "content": content})

    chosen_system = existing_system if existing_system in _KNOWN_SYSTEM_PROMPTS else _choose_system_prompt_for_conversation(cleaned, system)
    return [{"role": "system", "content": chosen_system}] + cleaned


def _extract_conversation(row: dict[str, Any]) -> list[dict[str, str]] | None:
    if "messages" in row and isinstance(row["messages"], list) and len(row["messages"]) >= 2:
        return format_sft_conversation(row["messages"])

    instruction = row.get("instruction") or row.get("input") or row.get("question") or row.get("prompt")
    output = row.get("output") or row.get("response") or row.get("answer") or row.get("completion")
    if instruction and output:
        return format_sft_conversation(
            [
                {"role": "user", "content": str(instruction)},
                {"role": "assistant", "content": str(output)},
            ]
        )
    return None


def _extract_dpo_pair(row: dict[str, Any]) -> dict[str, str] | None:
    prompt = row.get("prompt", "")
    chosen = row.get("chosen", "")
    rejected = row.get("rejected", "")

    if isinstance(prompt, list):
        prompt = " ".join(item.get("content", "") for item in prompt if isinstance(item, dict))
    if isinstance(chosen, list):
        chosen = " ".join(item.get("content", "") for item in chosen if isinstance(item, dict))
    if isinstance(rejected, list):
        rejected = " ".join(item.get("content", "") for item in rejected if isinstance(item, dict))

    if prompt and chosen and rejected:
        return {"prompt": str(prompt), "chosen": str(chosen), "rejected": str(rejected)}
    return None


def _extract_text_candidates(row: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    if "messages" in row and isinstance(row["messages"], list):
        for message in normalize_messages(row["messages"]):
            text = message.get("content")
            if text:
                texts.append(str(text))
    else:
        for key in ["instruction", "input", "question", "prompt", "output", "response", "answer", "completion", "text"]:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value)
    return texts


def _conversation_weight(source: str) -> int:
    weights = {
        "sft": 4,
        "chatbot": 1,
        "commandes_intents": 6,
        "commandes_augmented": 7,
        "multiturn_collection": 7,
        "slot_bootstrap": 8,
        "rag_grounded": 7,
        "few_shots": 5,
        "custom": 5,
        "enrichment": 3,
        "approved_learning": 5,
    }
    return weights.get(source, 2)


def _assistant_style_ok(text: str, *, allow_relaxed_tunisian: bool = False) -> bool:
    cleaned = normalize_text(text)
    if len(cleaned.strip()) < 8:
        return False
    if _CHINESE_RE.search(cleaned):
        return False
    if _contains_toxic_or_abusive_content(cleaned):
        return False
    if _contains_instructional_noise(cleaned):
        return False
    if is_moroccan_or_algerian(cleaned):
        return False
    if is_clean_tounsi(cleaned, max_fusha=2):
        return True
    if allow_relaxed_tunisian and looks_tunisian(cleaned, strict=False):
        return True
    script = detect_script(cleaned)
    if script in {"mixed", "arabizi"}:
        return True
    if script == "latin" and _contains_code_switch_business_terms(cleaned):
        return True
    return False


def _user_style_ok(text: str) -> bool:
    cleaned = normalize_text(text)
    if len(cleaned.strip()) < 2:
        return True
    if _contains_toxic_or_abusive_content(cleaned):
        return False
    if _contains_instructional_noise(cleaned):
        return False
    if is_moroccan_or_algerian(cleaned):
        return False
    return True


def _dpo_pair_domain_ok(pair: dict[str, str]) -> bool:
    prompt = normalize_text(pair.get("prompt", ""))
    chosen = normalize_text(pair.get("chosen", ""))
    cleaned = re.sub(r"(?im)^\s*(client|agent)\s*:\s*", "", "\n".join([prompt, chosen]))
    normalized = normalize_for_dedup(cleaned)

    score = 0
    if _ORDER_ID_RE.search(cleaned):
        score += 3
    if _NUM_CLIENT_TOKEN_RE.search(cleaned):
        score += 3
    if re.search(r"\b(?:num[_ ]?client|code[_ ]?client)\b", cleaned, re.IGNORECASE):
        score += 2
    if _OPTICS_MARKER_RE.search(cleaned):
        score += 2
    if "sivo" in normalized or "essilor" in normalized:
        score += 2
    for marker in _STRICT_DPO_DOMAIN_MARKERS:
        normalized_marker = normalize_for_dedup(marker)
        if normalized_marker and normalized_marker in normalized:
            score += 2

    return score >= 2


def _dpo_pair_ok(pair: dict[str, str], *, require_domain: bool = True) -> bool:
    prompt = normalize_text(pair.get("prompt", ""))
    chosen = normalize_text(pair.get("chosen", ""))
    rejected = normalize_text(pair.get("rejected", ""))

    if not prompt or not chosen or not rejected:
        return False
    if any(_contains_toxic_or_abusive_content(value) for value in [prompt, chosen, rejected]):
        return False
    if any(_contains_instructional_noise(value) for value in [prompt, chosen, rejected]):
        return False
    if is_moroccan_or_algerian(" ".join([prompt, chosen, rejected])):
        return False
    if require_domain and not _dpo_pair_domain_ok(pair):
        return False
    if not is_clean_tounsi(chosen, max_fusha=3):
        return False
    if not is_clean_tounsi(rejected, max_fusha=4):
        return False
    return True


def _dpo_prompt_from_messages(messages: list[dict[str, str]], assistant_index: int, max_turns: int = 6) -> str:
    role_map = {"user": "Client", "assistant": "Agent"}
    lines: list[str] = []
    for item in normalize_messages(messages[max(0, assistant_index - max_turns) : assistant_index]):
        role = item.get("role")
        content = normalize_text(item.get("content", ""))
        if role not in role_map or not content:
            continue
        lines.append(f"{role_map[role]}: {content}")
    return "\n".join(lines).strip()


def _pick_negative_variant(reference: str, variants: list[str]) -> str:
    if not variants:
        return ""
    index = sum(ord(char) for char in reference) % len(variants)
    return variants[index]


def _build_rejected_dpo_response(prompt: str, chosen: str) -> str:
    normalized = normalize_for_dedup(" ".join([prompt, chosen]))

    if any(marker in normalized for marker in ["livraison", "agence", "secteur", "creneau", "créneau"]):
        return _pick_negative_variant(
            normalized,
            [
                "أكيد، التسليم يمشي مباشرة للpatient نهار الاثنين 10:00 مضبوط.",
                "مأكد 100% اللي livraison توصلك للpatient اليوم من غير ما نراجع planning.",
                "خلاص، نعدك بوقت دقيق ونهبطوه مباشرة عند المريض.",
            ],
        )
    if _OPTICS_MARKER_RE.search(prompt) or any(
        marker in normalized
        for marker in ["create_order", "commande", "product", "produit", "verre", "diametre", "diamètre"]
    ):
        return _pick_negative_variant(
            normalized,
            [
                "مريقل، نبعث commande توة من غير recap للـ OD/OG ولا ما نثبت num client.",
                "ما يلزمنا حتى تأكيد إضافي، نكمل الكوموند مباشرة حتى كان القيم ناقصة.",
                "خلاص نعديها نهائية من غير ما نراجع sphere ولا cylindre ولا addition.",
            ],
        )
    if any(marker in normalized for marker in ["reference", "référence", "catalog", "catalogue", "dispo", "stock", "availability"]):
        return _pick_negative_variant(
            normalized,
            [
                "إي نعم، الكود صحيح ومتوفر 100% من غير ما نراجع catalog ولا stock.",
                "نأكدلك المرجع هذا أكيد صحيح وموجود توة بلا حتى تثبت.",
                "متوفر ومثبت نهائيا، ما ثماش داعي نرجع للمصدر الرسمي.",
            ],
        )
    if any(marker in normalized for marker in ["suivi", "statut", "commande", "order_id"]):
        return _pick_negative_variant(
            normalized,
            [
                "أكيد commande validée وتوصل غدوة، ما يلزمش حتى num client.",
                "نجم نأكدلك statut exact ووقت الوصول من غير ما نثبت في السيستام.",
                "كل شي حاضر، نعطيك وعد نهائي بالتسليم حتى من غير dossier.",
            ],
        )
    if any(marker in normalized for marker in ["prix", "سوم", "tarif", "price"]):
        return _pick_negative_variant(
            normalized,
            [
                "السوم هذا ثابت ومؤكد من غير ما نراجع التسعيرة الرسمية.",
                "نعطيك prix نهائي من عندي من غير ما نثبت في النظام.",
                "أكيد هذا هو السعر النهائي 100% وما فماش حتى تبديل.",
            ],
        )
    return _pick_negative_variant(
        normalized,
        [
            "أكيد، كل شي صحيح ومؤكد 100% من غير ما نثبت في السيستام.",
            "نجم نوعدك بإجابة نهائية حتى من غير ما نراجع المصدر.",
            "مريقل، نعطيك confirmation كاملة من غير ما نتحقق.",
        ],
    )


def _build_synthetic_dpo_pairs(raw_paths: dict[str, Path], limit: int | None = None) -> list[dict[str, str]]:
    entries = _collapse_sft_conversation_entries(_filtered_conversations(raw_paths))
    random.seed(SEED)
    random.shuffle(entries)

    pairs: list[dict[str, str]] = []
    seen_pairs: set[str] = set()

    for entry in entries:
        messages = entry.get("messages", [])
        if not conversation_domain_ok(messages, source=str(entry.get("source", "unknown"))):
            continue

        for idx, message in enumerate(normalize_messages(messages)):
            if message.get("role") != "assistant":
                continue

            prompt = _dpo_prompt_from_messages(messages, idx)
            chosen = normalize_text(message.get("content", ""))
            if len(prompt) < 12 or not _assistant_style_ok(chosen):
                continue

            rejected = _build_rejected_dpo_response(prompt, chosen)
            pair = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
            if normalize_for_dedup(chosen) == normalize_for_dedup(rejected):
                continue
            if not _dpo_pair_ok(pair):
                continue

            dedup_key = normalize_for_dedup(
                f"{pair['prompt']} || {pair['chosen']} || {pair['rejected']}"
            )
            if dedup_key in seen_pairs:
                continue
            seen_pairs.add(dedup_key)
            pairs.append(pair)
            if limit is not None and len(pairs) >= limit:
                return pairs

    return pairs


def _filtered_conversations(raw_paths: dict[str, Path]) -> list[dict[str, Any]]:
    all_conversations: list[dict[str, Any]] = []
    spec_map = dataset_spec_map()
    for name, path in raw_paths.items():
        if not _dataset_supports_sft(spec_map.get(name, {})):
            logger.info("Skipping %s for SFT: no sft/conversation role", name)
            continue
        before = len(all_conversations)
        rows = _load_jsonl(path)
        for row in rows:
            conv = _extract_conversation(row)
            if conv:
                all_conversations.append(
                    {"messages": conv, "source": name, "weight": _conversation_weight(name)}
                )
        logger.info("Loaded %d conversations from %s", len(all_conversations) - before, name)

    custom_path = PROCESSED_DATA_DIR / "custom_tounsi_conversations.jsonl"
    if custom_path.exists():
        for row in _load_jsonl(custom_path):
            conv = _extract_conversation(row)
            if conv:
                all_conversations.append(
                    {"messages": conv, "source": "custom", "weight": _conversation_weight("custom")}
                )

    enrichment_path = RAW_DATA_DIR / "enrichment" / "enriched_sft_conversations.jsonl"
    if enrichment_path.exists():
        for row in _load_jsonl(enrichment_path):
            conv = _extract_conversation(row)
            if conv:
                all_conversations.append(
                    {"messages": conv, "source": "enrichment", "weight": _conversation_weight("enrichment")}
                )

    if APPROVED_LEARNING_PATH.exists():
        for row in _load_jsonl(APPROVED_LEARNING_PATH):
            conv = _extract_conversation(row)
            if conv:
                all_conversations.append(
                    {
                        "messages": conv,
                        "source": "approved_learning",
                        "weight": _conversation_weight("approved_learning"),
                    }
                )

    for conv in _load_commandes_intent_conversations():
        all_conversations.append(
            {"messages": conv, "source": "commandes_intents", "weight": _conversation_weight("commandes_intents")}
        )
    for conv in _build_augmented_intent_conversations():
        all_conversations.append(
            {"messages": conv, "source": "commandes_augmented", "weight": _conversation_weight("commandes_augmented")}
        )
    for conv in _load_multiturn_collection_conversations():
        all_conversations.append(
            {
                "messages": conv,
                "source": "multiturn_collection",
                "weight": _conversation_weight("multiturn_collection"),
            }
        )
    for conv in _generate_slot_rich_bootstrap_conversations():
        all_conversations.append(
            {
                "messages": conv,
                "source": "slot_bootstrap",
                "weight": _conversation_weight("slot_bootstrap"),
            }
        )
    for conv in _load_few_shot_conversations():
        all_conversations.append(
            {"messages": conv, "source": "few_shots", "weight": _conversation_weight("few_shots")}
        )
    for conv in _build_rag_grounded_conversations():
        all_conversations.append(
            {"messages": conv, "source": "rag_grounded", "weight": _conversation_weight("rag_grounded")}
        )

    clean_conversations: list[dict[str, Any]] = []
    for entry in all_conversations:
        conv = entry["messages"]
        source = str(entry.get("source", "unknown"))
        if source in _SFT_NO_FILTER_SOURCES:
            if any(message.get("role") == "assistant" and str(message.get("content", "")).strip() for message in conv):
                clean_conversations.append(entry)
            continue
        allow_relaxed_style = source in _SFT_RELAXED_STYLE_SOURCES
        user_texts = [message["content"] for message in conv if message["role"] == "user"]
        assistant_texts = [message["content"] for message in conv if message["role"] == "assistant"]
        if not assistant_texts:
            continue
        if any(not _user_style_ok(text) for text in user_texts):
            continue
        if any(not _assistant_style_ok(text, allow_relaxed_tunisian=allow_relaxed_style) for text in assistant_texts):
            continue
        if not conversation_domain_ok(conv, source=source):
            continue
        clean_conversations.append(entry)

    return clean_conversations


def _is_low_quality_text(text: str) -> bool:
    normalized = normalize_text(text)
    if len(normalized) < 8:
        return True
    tokens = normalized.split()
    if len(tokens) < 2:
        return True
    if len(tokens) > 80:
        return True
    unique_ratio = len(set(tokens)) / max(1, len(tokens))
    if len(tokens) >= 8 and unique_ratio < 0.35:
        return True
    punctuation_ratio = sum(1 for ch in normalized if ch in "!?.,;:\u061b\u061f") / max(1, len(normalized))
    if punctuation_ratio > 0.35:
        return True
    return False


def _format_chat_messages(messages: list[dict[str, str]]) -> str:
    role_map = {"system": "System", "user": "User", "assistant": "Assistant"}
    lines: list[str] = []
    for message in messages:
        role = message.get("role")
        content = normalize_text(message.get("content", ""))
        if role not in role_map or not content:
            continue
        lines.append(f"{role_map[role]}: {content}")
    return "\n".join(lines).strip()


def _conversation_fingerprint(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for message in normalize_messages(messages):
        role = message.get("role")
        if role == "system":
            continue
        content = normalize_for_dedup(message.get("content", ""))
        if role in {"user", "assistant"} and content:
            parts.append(f"{role}:{content}")
    return " || ".join(parts)


def _collapse_sft_conversation_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    collapsed: dict[str, dict[str, Any]] = {}
    for entry in entries:
        messages = entry.get("messages", [])
        fingerprint = _conversation_fingerprint(messages)
        if not fingerprint:
            continue

        source = str(entry.get("source", "unknown"))
        weight = max(1, int(entry.get("weight", 1)))
        existing = collapsed.get(fingerprint)
        if existing is None:
            collapsed[fingerprint] = {
                "messages": messages,
                "source": source,
                "sources": [source],
                "weight": weight,
            }
            continue

        if source not in existing["sources"]:
            existing["sources"].append(source)
        if weight >= int(existing.get("weight", 1)):
            existing["messages"] = messages
            existing["source"] = source
        existing["weight"] = max(int(existing.get("weight", 1)), weight)

    return list(collapsed.values())


def _repeat_count_from_weight(weight: int) -> int:
    repeats = 1
    if weight >= 4:
        repeats += 1
    if weight >= 7:
        repeats += 1
    return min(CFG.max_sft_train_repeats, max(1, repeats))


def _materialize_weighted_train_conversations(entries: list[dict[str, Any]]) -> list[list[dict[str, str]]]:
    materialized: list[list[dict[str, str]]] = []
    for entry in entries:
        repeats = _repeat_count_from_weight(int(entry.get("weight", 1)))
        materialized.extend([entry["messages"]] * repeats)
    return materialized


def _script_distribution_from_conversations(conversations: list[list[dict[str, str]]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    total_turns = 0
    for conversation in conversations:
        for message in normalize_messages(conversation):
            if message.get("role") not in {"user", "assistant"}:
                continue
            content = message.get("content", "")
            if not content:
                continue
            counts[detect_script(content)] += 1
            total_turns += 1

    percentages = {
        key: round(value / max(total_turns, 1) * 100, 2)
        for key, value in sorted(counts.items())
    }
    return {
        "total_turns": total_turns,
        "counts": dict(sorted(counts.items())),
        "percentages": percentages,
    }


def _build_sft_fallback_entries() -> list[dict[str, Any]]:
    fallback_entries: list[dict[str, Any]] = []

    for conversation in _build_augmented_intent_conversations(
        variants_per_row=CFG.sft_fallback_augmented_variants_per_row,
    ):
        fallback_entries.append(
            {
                "messages": conversation,
                "source": "commandes_augmented",
                "weight": _conversation_weight("commandes_augmented"),
            }
        )

    for conversation in _generate_slot_rich_bootstrap_conversations(limit=CFG.sft_fallback_slot_bootstrap_limit):
        fallback_entries.append(
            {
                "messages": conversation,
                "source": "slot_bootstrap",
                "weight": _conversation_weight("slot_bootstrap"),
            }
        )

    for conversation in _build_rag_grounded_conversations(
        max_delivery=CFG.sft_fallback_rag_delivery_limit,
        max_lens=CFG.sft_fallback_rag_lens_limit,
    ):
        fallback_entries.append(
            {
                "messages": conversation,
                "source": "rag_grounded",
                "weight": _conversation_weight("rag_grounded"),
            }
        )

    return fallback_entries


def _ensure_min_sft_unique_conversations(entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_unique = max(1, int(CFG.min_sft_unique_conversations))
    initial_unique = len(entries)
    if initial_unique >= target_unique:
        return entries, {
            "target_unique": target_unique,
            "initial_unique": initial_unique,
            "final_unique": initial_unique,
            "fallback_added_unique": 0,
            "fallback_triggered": False,
        }

    fallback_entries = _build_sft_fallback_entries()
    expanded_entries = _collapse_sft_conversation_entries(entries + fallback_entries)
    final_unique = len(expanded_entries)
    added_unique = max(0, final_unique - initial_unique)

    logger.warning(
        "SFT unique conversations below target (%d < %d). Fallback augmentation added %d unique conversations.",
        initial_unique,
        target_unique,
        added_unique,
    )

    return expanded_entries, {
        "target_unique": target_unique,
        "initial_unique": initial_unique,
        "final_unique": final_unique,
        "fallback_added_unique": added_unique,
        "fallback_triggered": True,
    }


def _extract_chat_samples(row: dict[str, Any], max_turns: int = 8) -> list[dict[str, str]]:
    payload = row.get("messages")
    if not isinstance(payload, list):
        return []

    cleaned = normalize_messages(payload)
    cleaned = [item for item in cleaned if item["role"] in {"system", "user", "assistant"}]

    if len(cleaned) < 2:
        return []

    samples: list[dict[str, str]] = []
    for idx, message in enumerate(cleaned):
        if message["role"] != "assistant":
            continue
        context = cleaned[max(0, idx - max_turns + 1) : idx + 1]
        if not any(item["role"] == "user" for item in context):
            continue
        formatted = _format_chat_messages(context)
        if len(formatted) < 30:
            continue
        samples.append({"text": formatted, "anchor": message["content"]})
    return samples


def _sample_prioritized_texts(chat_texts: list[str], plain_texts: list[str], max_texts: int) -> list[str]:
    combined = chat_texts + plain_texts
    if len(combined) <= max_texts:
        return combined

    random.seed(SEED)
    selected: list[str] = []

    min_chat = min(len(chat_texts), max(1, int(max_texts * 0.4)))
    if min_chat:
        selected.extend(random.sample(chat_texts, min_chat) if len(chat_texts) > min_chat else chat_texts)

    remaining = max_texts - len(selected)
    if remaining > 0:
        pool = [text for text in plain_texts if text not in selected]
        if len(pool) > remaining:
            selected.extend(random.sample(pool, remaining))
        else:
            selected.extend(pool)

    remaining = max_texts - len(selected)
    if remaining > 0:
        pool = [text for text in chat_texts if text not in selected]
        if len(pool) > remaining:
            selected.extend(random.sample(pool, remaining))
        else:
            selected.extend(pool)

    return selected[:max_texts]


def _dataset_supports_self_sup(spec: dict[str, Any]) -> bool:
    roles = spec.get("roles", [])
    if roles:
        normalized_roles = {str(role).strip().lower() for role in roles}
        return "self_sup" in normalized_roles

    dataset_type = str(spec.get("type", "")).strip().lower()
    if dataset_type in {"dpo", "preference", "rlhf"}:
        return False
    if dataset_type in {"sft", "dialogue", "conversation", "chat", "self_sup"}:
        return True

    if spec.get("enable_self_sup") is False:
        return False
    return True


def _dataset_supports_dpo(spec: dict[str, Any]) -> bool:
    roles = spec.get("roles", [])
    if roles:
        normalized_roles = {str(role).strip().lower() for role in roles}
        return bool(normalized_roles.intersection({"dpo", "preference", "rlhf"}))

    dataset_type = str(spec.get("type", "")).strip().lower()
    return dataset_type in {"dpo", "preference", "rlhf"}


def _dataset_supports_sft(spec: dict[str, Any]) -> bool:
    roles = spec.get("roles", [])
    if roles:
        normalized_roles = {str(role).strip().lower() for role in roles}
        return bool(normalized_roles.intersection({"sft", "conversation", "chat"}))

    dataset_type = str(spec.get("type", "")).strip().lower()
    return dataset_type in {"sft", "dialogue", "conversation", "chat"}


def prepare_self_supervised_data(
    raw_paths: dict[str, Path],
    output_dir: Path | None = None,
    max_texts: int | None = None,
) -> dict[str, Path]:
    output_dir = output_dir or PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    spec_map = dataset_spec_map()
    seen: set[str] = set()
    plain_texts: list[str] = []
    chat_texts: list[str] = []

    for name, path in raw_paths.items():
        spec = spec_map.get(name, {})
        if not _dataset_supports_self_sup(spec):
            logger.info("Skipping %s for self-sup: no self_sup role", name)
            continue

        strict_tunisian = bool(spec.get("strict_tunisian_filter", False))
        min_text_length = int(spec.get("min_text_length", 20))
        candidate_count = 0
        kept_plain = 0
        kept_chat = 0

        for row in _load_jsonl(path):
            for sample in _extract_chat_samples(row):
                candidate_count += 1
                if _contains_instructional_noise(sample["text"]):
                    continue
                if not _self_sup_language_ok(sample["anchor"], strict_tunisian=strict_tunisian):
                    continue
                if not is_clean_tounsi(sample["anchor"], max_fusha=3):
                    continue
                dedup_key = normalize_for_dedup(sample["text"])
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                chat_texts.append(sample["text"])
                kept_chat += 1

            candidate_texts = extract_text_candidates_generic(row, spec=spec)
            dpo_pair = _extract_dpo_pair(row)
            if dpo_pair:
                candidate_texts.extend([dpo_pair["chosen"], dpo_pair["rejected"]])

            for text in candidate_texts:
                candidate_count += 1
                text = normalize_text(text)
                if len(text) < min_text_length:
                    continue
                if len(text) > 400:
                    text = text[:400].rstrip()
                if "```" in text:
                    continue
                if _is_low_quality_text(text):
                    continue
                if not is_clean_tounsi(text, max_fusha=3):
                    continue
                if not _self_sup_language_ok(text, strict_tunisian=strict_tunisian):
                    continue
                dedup_key = normalize_for_dedup(text)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                plain_texts.append(text)
                kept_plain += 1

        logger.info(
            "Self-sup filter source=%s strict_tunisian=%s kept_plain=%d kept_chat=%d candidates=%d",
            name,
            strict_tunisian,
            kept_plain,
            kept_chat,
            candidate_count,
        )

    enrichment_all = RAW_DATA_DIR / "enrichment" / "all_clean_tounsi.jsonl"
    if enrichment_all.exists():
        for row in _load_jsonl(enrichment_all):
            for text in extract_text_candidates_generic(row):
                text = normalize_text(text)
                if len(text) < 20:
                    continue
                if _is_low_quality_text(text):
                    continue
                if not is_clean_tounsi(text, max_fusha=3):
                    continue
                dedup_key = normalize_for_dedup(text)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                plain_texts.append(text)

    for text in _load_config_self_sup_texts():
        if not _self_sup_language_ok(text, strict_tunisian=False):
            continue
        dedup_key = normalize_for_dedup(text)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        plain_texts.append(text)

    for text in _load_synthetic_self_sup_texts():
        if not _self_sup_language_ok(text, strict_tunisian=False):
            continue
        dedup_key = normalize_for_dedup(text)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        plain_texts.append(text)

    for text in _load_rag_grounded_self_sup_texts():
        if not _self_sup_language_ok(text, strict_tunisian=False):
            continue
        dedup_key = normalize_for_dedup(text)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        plain_texts.append(text)

    texts = chat_texts + plain_texts
    if max_texts and len(texts) > max_texts:
        texts = _sample_prioritized_texts(chat_texts, plain_texts, max_texts)

    random.seed(SEED)
    random.shuffle(texts)

    if len(texts) <= 1:
        train_texts = texts
        val_texts = []
    else:
        split_at = max(1, int(len(texts) * 0.97))
        split_at = min(split_at, len(texts) - 1)
        train_texts = texts[:split_at]
        val_texts = texts[split_at:]

    paths = {
        "train": output_dir / "self_sup_train.jsonl",
        "val": output_dir / "self_sup_val.jsonl",
    }
    for split_name, split_texts in [("train", train_texts), ("val", val_texts)]:
        with open(paths[split_name], "w", encoding="utf-8") as handle:
            for text in split_texts:
                handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        logger.info("Self-supervised %s: %d texts -> %s", split_name, len(split_texts), paths[split_name])

    logger.info(
        "Self-sup combined rows=%d (chat=%d plain=%d)",
        len(texts),
        len(chat_texts),
        len(plain_texts),
    )
    return paths


def prepare_sft_data(
    raw_paths: dict[str, Path],
    output_dir: Path | None = None,
    max_samples: int | None = None,
) -> dict[str, Path]:
    output_dir = output_dir or PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    conversation_entries = _collapse_sft_conversation_entries(_filtered_conversations(raw_paths))
    conversation_entries, fallback_summary = _ensure_min_sft_unique_conversations(conversation_entries)
    if max_samples and len(conversation_entries) > max_samples:
        random.seed(SEED)
        conversation_entries = random.sample(conversation_entries, max_samples)

    random.seed(SEED)
    random.shuffle(conversation_entries)

    n = len(conversation_entries)
    if n <= 2:
        train_entries = conversation_entries
        val = []
        test = []
    else:
        n_val = max(1, int(n * 0.05))
        n_test = max(1, int(n * 0.05))
        while n - n_val - n_test < 1 and (n_val > 1 or n_test > 1):
            if n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1

        def _entry_key(entry: dict[str, Any]) -> str:
            fingerprint = _conversation_fingerprint(entry.get("messages", []))
            return fingerprint or normalize_for_dedup(str(entry.get("messages", "")))

        eval_target = n_val + n_test
        domain_pool = [
            entry
            for entry in conversation_entries
            if conversation_domain_ok(entry.get("messages", []), source="validation")
        ]
        non_domain_pool = [
            entry
            for entry in conversation_entries
            if not conversation_domain_ok(entry.get("messages", []), source="validation")
        ]

        eval_entries = list(domain_pool[:eval_target])
        if len(eval_entries) < eval_target:
            eval_entries.extend(non_domain_pool[: eval_target - len(eval_entries)])

        eval_keys = {_entry_key(entry) for entry in eval_entries}
        train_entries = [entry for entry in conversation_entries if _entry_key(entry) not in eval_keys]

        if len(train_entries) < 1 and eval_entries:
            train_entries = [eval_entries.pop()]

        val_entries = eval_entries[:n_val]
        test_entries = eval_entries[n_val : n_val + n_test]
        val = [entry["messages"] for entry in val_entries]
        test = [entry["messages"] for entry in test_entries]

    if n > 2 and not val:
        val = test[:1]
        test = test[1:]

    train = _materialize_weighted_train_conversations(train_entries)
    source_distribution = dict(
        sorted(
            Counter(str(entry.get("source", "unknown")) for entry in conversation_entries).items(),
            key=lambda item: item[0],
        )
    )
    train_script_distribution = _script_distribution_from_conversations(train)
    val_script_distribution = _script_distribution_from_conversations(val)
    test_script_distribution = _script_distribution_from_conversations(test)
    prep_report = {
        "unique_conversations": len(conversation_entries),
        "train_unique_conversations": len(train_entries),
        "train_materialized_conversations": len(train),
        "val_unique_conversations": len(val),
        "test_unique_conversations": len(test),
        "max_train_repeats": CFG.max_sft_train_repeats,
        "train_repeat_factor": round(len(train) / max(len(train_entries), 1), 2),
        "min_unique_target": CFG.min_sft_unique_conversations,
        "source_distribution": source_distribution,
        "fallback_summary": fallback_summary,
        "script_distribution": {
            "train": train_script_distribution,
            "val": val_script_distribution,
            "test": test_script_distribution,
        },
    }
    (REPORTS_DIR / "sft_data_prep.json").write_text(
        json.dumps(prep_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    paths = {}
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = output_dir / f"sft_{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for conv in split_data:
                handle.write(json.dumps({"messages": conv}, ensure_ascii=False) + "\n")
        paths[split_name] = path
        logger.info("SFT %s: %d conversations -> %s", split_name, len(split_data), path)
    logger.info("SFT prep summary: %s", prep_report)
    return paths


def prepare_dpo_data(
    raw_paths: dict[str, Path],
    output_dir: Path | None = None,
) -> dict[str, Path]:
    output_dir = output_dir or PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[dict[str, str]] = []
    seen_pairs: set[str] = set()
    spec_map = dataset_spec_map()
    source_counts: dict[str, int] = {}

    for name, path in raw_paths.items():
        spec = spec_map.get(name, {})
        if not _dataset_supports_dpo(spec):
            continue

        source_cap = spec.get("max_dpo_pairs")
        source_cap_int = int(source_cap) if source_cap is not None else None
        loaded_for_source = 0

        for row in _load_jsonl(path):
            pair = _extract_dpo_pair(row)
            if not pair:
                continue
            if len(pair["chosen"].strip()) < 15 or len(pair["rejected"].strip()) < 15:
                continue
            if not _dpo_pair_ok(pair):
                continue

            dedup_key = normalize_for_dedup(
                f"{pair['prompt']} || {pair['chosen']} || {pair['rejected']}"
            )
            if dedup_key in seen_pairs:
                continue
            seen_pairs.add(dedup_key)

            pairs.append(pair)
            loaded_for_source += 1
            if source_cap_int is not None and loaded_for_source >= source_cap_int:
                break

        source_counts[name] = loaded_for_source
        logger.info(
            "DPO source=%s loaded_pairs=%d cap=%s",
            name,
            loaded_for_source,
            source_cap_int,
        )

    if APPROVED_FEEDBACK_DPO_PATH.exists():
        for row in _load_jsonl(APPROVED_FEEDBACK_DPO_PATH):
            pair = _extract_dpo_pair(row)
            if pair and _dpo_pair_ok(pair):
                dedup_key = normalize_for_dedup(
                    f"{pair['prompt']} || {pair['chosen']} || {pair['rejected']}"
                )
                if dedup_key in seen_pairs:
                    continue
                seen_pairs.add(dedup_key)
                pairs.append(pair)
                source_counts["approved_feedback"] = source_counts.get("approved_feedback", 0) + 1

    target_total_pairs = max(int(CFG.min_dpo_train_rows * 1.2), CFG.min_dpo_train_rows + 500)
    synthetic_added = 0
    if len(pairs) < target_total_pairs:
        synthetic_pairs = _build_synthetic_dpo_pairs(raw_paths, limit=target_total_pairs - len(pairs))
        for pair in synthetic_pairs:
            dedup_key = normalize_for_dedup(
                f"{pair['prompt']} || {pair['chosen']} || {pair['rejected']}"
            )
            if dedup_key in seen_pairs:
                continue
            seen_pairs.add(dedup_key)
            pairs.append(pair)
            synthetic_added += 1
        if synthetic_added:
            source_counts["synthetic_domain_dpo"] = synthetic_added
            logger.info("DPO synthetic_domain_dpo added_pairs=%d", synthetic_added)

    if not pairs:
        logger.warning("No DPO pairs found after filtering.")
        return {}

    random.seed(SEED)
    random.shuffle(pairs)
    if len(pairs) <= 2:
        train = pairs
        val = []
    else:
        n_val = max(1, int(len(pairs) * 0.1))
        if len(pairs) - n_val < 1:
            n_val = 1
        train = pairs[:-n_val]
        val = pairs[-n_val:]

    paths = {}
    for split_name, split_data in [("train", train), ("val", val)]:
        path = output_dir / f"dpo_{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for pair in split_data:
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
        paths[split_name] = path
        logger.info("DPO %s: %d pairs -> %s", split_name, len(split_data), path)

    (REPORTS_DIR / "dpo_data_prep.json").write_text(
        json.dumps(
            {
                "total_pairs": len(pairs),
                "train_pairs": len(train),
                "val_pairs": len(val),
                "source_counts": source_counts,
                "synthetic_added": synthetic_added,
                "target_total_pairs": target_total_pairs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    return _load_jsonl(path)


def _count_jsonl_rows(path: Path | None) -> int:
    if not path or not path.exists():
        return 0
    with open(path, encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _collect_processed_texts(path: Path | None, key: str) -> list[str]:
    if not path or not path.exists():
        return []
    rows = _iter_jsonl(path)
    texts: list[str] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
    return texts


def _collect_sft_messages(path: Path | None) -> list[list[dict[str, str]]]:
    if not path or not path.exists():
        return []
    rows = _iter_jsonl(path)
    conversations: list[list[dict[str, str]]] = []
    for row in rows:
        messages = row.get("messages")
        if isinstance(messages, list):
            conversations.append(normalize_messages(messages))
    return conversations


def _write_prepared_dataset_audit(outputs: dict[str, dict[str, Path]]) -> None:
    marker_patterns = {
        "num_client": re.compile(r"num[_ ]?client|code[_ ]?client|\bCLI\d+", re.IGNORECASE),
        "order_id": re.compile(r"\bCMD[-_ ]?\d+", re.IGNORECASE),
        "reference": re.compile(r"r[ée]f[ée]rence|code verre|catalog|catalogue", re.IGNORECASE),
        "delivery": re.compile(r"livraison|agence|secteur|creneau|créneau|planning", re.IGNORECASE),
        "optic_values": re.compile(r"\bOD\b|\bOG\b|sphere|sphère|cyl|cylindre|axe|addition|diam[eè]tre", re.IGNORECASE),
        "availability": re.compile(r"dispo|disponibilit|stock|fabrication|sur commande", re.IGNORECASE),
    }

    def pick_example(conversations: list[list[dict[str, str]]], pattern: re.Pattern[str]) -> dict[str, str] | None:
        for conversation in conversations:
            joined = " ".join(message.get("content", "") for message in conversation)
            if not pattern.search(joined):
                continue
            user_text = next((m.get("content", "") for m in conversation if m.get("role") == "user"), "")
            assistant_text = next((m.get("content", "") for m in conversation if m.get("role") == "assistant"), "")
            if user_text and assistant_text:
                return {"user": user_text, "assistant": assistant_text}
        return None

    self_sup_train = outputs.get("self_sup", {}).get("train")
    self_sup_val = outputs.get("self_sup", {}).get("val")
    sft_train = outputs.get("sft", {}).get("train")
    sft_val = outputs.get("sft", {}).get("val")
    sft_test = outputs.get("sft", {}).get("test")
    dpo_train = outputs.get("dpo", {}).get("train")
    dpo_val = outputs.get("dpo", {}).get("val")

    self_sup_texts = _collect_processed_texts(self_sup_train, "text")
    self_sup_val_texts = _collect_processed_texts(self_sup_val, "text")
    sft_conversations = _collect_sft_messages(sft_train)
    sft_val_conversations = _collect_sft_messages(sft_val)
    dpo_pairs = _iter_jsonl(dpo_train) if dpo_train and dpo_train.exists() else []

    commandes_rows = _load_jsonl(COMMANDES_INTENTS_PATH)
    intent_counts = Counter()
    slot_field_counts = Counter()
    for row in commandes_rows:
        intent = canonicalize_intent(row.get("intent"))
        if intent:
            intent_counts[intent] += 1
        slots = row.get("slots", {}) if isinstance(row.get("slots"), dict) else {}
        for key, value in slots.items():
            if value not in ("", None, [], {}):
                slot_field_counts[key] += 1

    sft_conversation_hits = Counter()
    sft_assistant_hits = Counter()
    for conversation in sft_conversations:
        joined = " ".join(message.get("content", "") for message in conversation)
        for label, pattern in marker_patterns.items():
            if pattern.search(joined):
                sft_conversation_hits[label] += 1
            if any(
                message.get("role") == "assistant" and pattern.search(message.get("content", ""))
                for message in conversation
            ):
                sft_assistant_hits[label] += 1

    dpo_prompt_hits = Counter()
    for row in dpo_pairs:
        prompt = str(row.get("prompt", ""))
        for label, pattern in marker_patterns.items():
            if pattern.search(prompt):
                dpo_prompt_hits[label] += 1

    report = {
        "commandes_intents_source": {
            "path": str(COMMANDES_INTENTS_PATH),
            "rows": len(commandes_rows),
            "intent_counts": dict(intent_counts),
            "slot_field_counts": dict(slot_field_counts),
        },
        "self_sup": {
            "train_rows": _count_jsonl_rows(self_sup_train),
            "val_rows": _count_jsonl_rows(self_sup_val),
            "train_quality": compute_quality_stats(self_sup_texts[:5000]),
            "val_quality": compute_quality_stats(self_sup_val_texts[:1000]),
        },
        "sft": {
            "train_rows": _count_jsonl_rows(sft_train),
            "val_rows": _count_jsonl_rows(sft_val),
            "test_rows": _count_jsonl_rows(sft_test),
            "train_marker_coverage": dict(sft_conversation_hits),
            "assistant_marker_coverage": dict(sft_assistant_hits),
            "val_rows_loaded_for_examples": len(sft_val_conversations),
            "examples": {
                label: pick_example(sft_conversations, pattern)
                for label, pattern in marker_patterns.items()
            },
        },
        "dpo": {
            "train_rows": _count_jsonl_rows(dpo_train),
            "val_rows": _count_jsonl_rows(dpo_val),
            "prompt_marker_coverage": dict(dpo_prompt_hits),
            "examples": dpo_pairs[:3],
        },
    }

    report_json_path = REPORTS_DIR / "prepared_dataset_audit.json"
    report_md_path = REPORTS_DIR / "prepared_dataset_audit.md"
    report_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown_lines = [
        "# Prepared Dataset Audit",
        "",
        "## Commandes Intents Source",
        f"- rows: {report['commandes_intents_source']['rows']}",
        f"- intents: {json.dumps(report['commandes_intents_source']['intent_counts'], ensure_ascii=False)}",
        f"- slot_fields: {json.dumps(report['commandes_intents_source']['slot_field_counts'], ensure_ascii=False)}",
        "",
        "## Self-Supervised",
        f"- train_rows: {report['self_sup']['train_rows']}",
        f"- val_rows: {report['self_sup']['val_rows']}",
        f"- train_quality: {json.dumps(report['self_sup']['train_quality'], ensure_ascii=False)}",
        "",
        "## SFT",
        f"- train_rows: {report['sft']['train_rows']}",
        f"- val_rows: {report['sft']['val_rows']}",
        f"- test_rows: {report['sft']['test_rows']}",
        f"- train_marker_coverage: {json.dumps(report['sft']['train_marker_coverage'], ensure_ascii=False)}",
        f"- assistant_marker_coverage: {json.dumps(report['sft']['assistant_marker_coverage'], ensure_ascii=False)}",
        "",
        "## DPO",
        f"- train_rows: {report['dpo']['train_rows']}",
        f"- val_rows: {report['dpo']['val_rows']}",
        f"- prompt_marker_coverage: {json.dumps(report['dpo']['prompt_marker_coverage'], ensure_ascii=False)}",
        "",
        "## SFT Examples",
    ]
    for label, example in report["sft"]["examples"].items():
        markdown_lines.append(f"- {label}: {json.dumps(example, ensure_ascii=False)}")
    markdown_lines.extend(["", "## DPO Examples"])
    for example in report["dpo"]["examples"]:
        markdown_lines.append(f"- {json.dumps(example, ensure_ascii=False)}")

    report_md_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    logger.info("Prepared dataset audit written to %s and %s", report_json_path, report_md_path)


def prepare_all_data(
    raw_paths: dict[str, Path],
    output_dir: Path | None = None,
    max_sft_samples: int | None = None,
    max_self_sup_texts: int | None = None,
) -> dict[str, dict[str, Path]]:
    outputs = {
        "self_sup": prepare_self_supervised_data(raw_paths, output_dir=output_dir, max_texts=max_self_sup_texts),
        "sft": prepare_sft_data(raw_paths, output_dir=output_dir, max_samples=max_sft_samples),
        "dpo": prepare_dpo_data(raw_paths, output_dir=output_dir),
    }
    _write_prepared_dataset_audit(outputs)
    return outputs
