"""
Recommendation chatbot engine driven by the official Scenario Reco guide.

This module is deterministic by design so recommendations stay stable and
traceable across sessions while still using RAG evidence snippets.
"""
from __future__ import annotations

import copy
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

from .config import DOMAIN_CFG, HISTORY_DIR, logger, resolve_project_path
from .rag import VectorRAGRetriever
from .storage import get_database_backend


RecommendationProfile = dict[str, Any]


_RECO_SESSIONS: dict[str, dict[str, Any]] = {}

_MEMORY_CFG = DOMAIN_CFG.get("memory", {})
_RECO_STATE_PATH = resolve_project_path(
    _MEMORY_CFG.get("recommendation_state_path", HISTORY_DIR / "recommendation_session_state.json")
)
_RECO_LOG_PATH = resolve_project_path(
    _MEMORY_CFG.get("recommendation_log_path", HISTORY_DIR / "recommendation_conversations.jsonl")
)
_RECO_HISTORY_LIMIT = int(_MEMORY_CFG.get("recommendation_history_limit", 20))
_RECO_STATE_LOADED = False

for _path in (_RECO_STATE_PATH, _RECO_LOG_PATH):
    _path.parent.mkdir(parents=True, exist_ok=True)


_KEY_ALIASES = {
    "add": "add_power",
    "addition": "add_power",
    "diff_od_og": "od_og_diff",
    "difference_od_og": "od_og_diff",
    "montage": "frame_type",
    "usage_env": "work_env",
    "besoin": "main_need",
    "sante_oculaire": "ocular_health",
    "gene_lumiere": "light_discomfort",
    "innovation": "innovation_sensitive",
    "adaptation": "adaptation_easy",
}


_YES_WORDS = {"oui", "yes", "y", "ok", "daccord", "d'accord", "bien sur", "bien-sûr", "exact"}
_NO_WORDS = {"non", "no", "n", "jamais", "pas"}


_OcularPrevenciaTriggers = {
    "pseudophaque",
    "aphakie",
    "cataracte",
    "glaucome",
    "dmla",
    "conjonctivite",
    "retinopathie diabetique",
    "retinopathie",
}


def _load_sessions_once() -> None:
    global _RECO_STATE_LOADED
    if _RECO_STATE_LOADED:
        return

    if _RECO_STATE_PATH.exists():
        try:
            payload = json.loads(_RECO_STATE_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not load recommendation session state: %s", exc)
            payload = {}

        if isinstance(payload, dict):
            for session_id, state in payload.items():
                if not isinstance(state, dict):
                    continue
                _RECO_SESSIONS[str(session_id)] = {
                    "profile": state.get("profile", {}) if isinstance(state.get("profile"), dict) else {},
                    "history": list(state.get("history", []))[-_RECO_HISTORY_LIMIT:],
                    "last_question_field": state.get("last_question_field"),
                    "updated_at": float(state.get("updated_at", time.time())),
                }

    _RECO_STATE_LOADED = True


def _save_sessions() -> None:
    payload: dict[str, dict[str, Any]] = {}
    for session_id, state in _RECO_SESSIONS.items():
        payload[session_id] = {
            "profile": state.get("profile", {}),
            "history": list(state.get("history", []))[-_RECO_HISTORY_LIMIT:],
            "last_question_field": state.get("last_question_field"),
            "updated_at": float(state.get("updated_at", time.time())),
        }
    _RECO_STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_recommendation_event(
    *,
    session_id: str,
    user_text: str,
    assistant_text: str,
    metadata: dict[str, Any],
) -> None:
    now = time.time()
    record = {
        "endpoint": "recommendation",
        "session_id": session_id,
        "timestamp": now,
        "user": user_text,
        "assistant": assistant_text,
        "metadata": metadata,
    }

    with open(_RECO_LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    try:
        get_database_backend().record_conversation(
            {
                "session_id": session_id,
                "timestamp": now,
                "model_variant": "recommendation",
                "user": user_text,
                "assistant": assistant_text,
                "metadata": {"endpoint": "recommendation", **metadata},
            }
        )
    except Exception as exc:
        logger.warning("Could not persist recommendation event in DB: %s", exc)


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def _normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = _norm(str(value))
    if not text:
        return None
    if text in _YES_WORDS:
        return True
    if text in _NO_WORDS:
        return False
    return None


def _normalize_choice(value: Any, choices: dict[str, str]) -> str | None:
    text = _norm(str(value))
    if not text:
        return None
    if text in choices:
        return choices[text]
    for key, mapped in choices.items():
        if key in text:
            return mapped
    return None


def _extract_profile_updates_from_message(message: str) -> RecommendationProfile:
    text = str(message or "").strip()
    normalized = _norm(text)
    updates: RecommendationProfile = {}

    age_match = re.search(r"(?:j[' ]?ai|age|age de|age est|j ai)?\s*(\d{1,2})\s*ans\b", normalized)
    if age_match:
        updates["age"] = int(age_match.group(1))

    if "perce" in normalized:
        updates["frame_type"] = "perce"
    elif "nylor" in normalized:
        updates["frame_type"] = "nylor"
    elif "metall" in normalized:
        updates["frame_type"] = "metallique"
    elif "plastique" in normalized:
        updates["frame_type"] = "plastique"

    if "correction" in normalized or "od" in normalized or "og" in normalized:
        if "inferieur" in normalized and "200" in normalized:
            updates["correction_total"] = 199
        elif "superieur" in normalized and ("625" in normalized or "600" in normalized):
            updates["correction_total"] = 650
        elif "entre" in normalized:
            range_match = re.search(r"entre\s*(\d{2,4})\s*(?:et|a)\s*(\d{2,4})", normalized)
            if range_match:
                low = float(range_match.group(1))
                high = float(range_match.group(2))
                updates["correction_total"] = (low + high) / 2.0
        else:
            total_match = re.search(r"correction(?: totale)?\s*(?:=|:)?\s*(\d{2,4}(?:[\.,]\d+)?)", normalized)
            if total_match:
                updates["correction_total"] = float(total_match.group(1).replace(",", "."))

    add_match = re.search(r"(?:\badd\b|addition)\s*(?:=|:)?\s*(\d+(?:[\.,]\d+)?)", normalized)
    if add_match:
        updates["add_power"] = float(add_match.group(1).replace(",", "."))

    diff_match = re.search(r"(?:difference|ecart).{0,20}(\d{2,4})", normalized)
    if diff_match and ("od" in normalized or "og" in normalized):
        updates["od_og_diff"] = float(diff_match.group(1))
    elif "difference" in normalized and "200" in normalized and "superieur" in normalized:
        updates["od_og_diff"] = 250.0

    if "transpar" in normalized:
        updates["main_need"] = "transparence"
    elif "bleue" in normalized or "lumiere bleue" in normalized:
        updates["main_need"] = "lumiere_bleue"
    elif "conduite" in normalized and ("soir" in normalized or "nuit" in normalized):
        updates["main_need"] = "conduite_soir"
    elif "rayure" in normalized:
        updates["main_need"] = "rayures"
    elif "nettoy" in normalized or "anti reflet" in normalized or "antireflet" in normalized:
        updates["main_need"] = "nettoyage"

    if "interieur" in normalized and "exterieur" in normalized:
        updates["work_env"] = "mixte"
    elif "exterieur" in normalized:
        updates["work_env"] = "exterieur"
    elif "interieur" in normalized:
        updates["work_env"] = "interieur"

    if "tres fort" in normalized or "tres sensible" in normalized or "forte" in normalized:
        updates["light_discomfort"] = "forte"
    elif "moyen" in normalized:
        updates["light_discomfort"] = "moyenne"
    elif "faible" in normalized or "absente" in normalized:
        updates["light_discomfort"] = "faible"

    if "innovation" in normalized or "technolog" in normalized:
        if any(token in normalized for token in _YES_WORDS):
            updates["innovation_sensitive"] = True

    if "adaptation facile" in normalized:
        if any(token in normalized for token in _NO_WORDS):
            updates["adaptation_easy"] = False
        else:
            updates["adaptation_easy"] = True

    if "ordinateur" in normalized:
        if "tres souvent" in normalized:
            updates["computer_usage"] = "high"
        elif "parfois" in normalized or "un peu" in normalized:
            updates["computer_usage"] = "medium"
        else:
            updates["computer_usage"] = "low"

    if "bouge" in normalized and "tete" in normalized:
        updates["head_eye_behavior"] = "head"
    elif "yeux" in normalized and "tete" in normalized:
        updates["head_eye_behavior"] = "mixed"
    elif "yeux" in normalized:
        updates["head_eye_behavior"] = "eyes"

    if "conduis" in normalized or "conduite" in normalized:
        if "nuit" in normalized and ("souvent" in normalized or "tres" in normalized):
            updates["night_driving"] = True

    if "paire" in normalized and "soleil" in normalized:
        if any(word in normalized for word in _NO_WORDS):
            updates["wants_sun_pair"] = False
        elif any(word in normalized for word in _YES_WORDS) or "oui" in normalized:
            updates["wants_sun_pair"] = True

    if "reflet" in normalized:
        if any(word in normalized for word in _NO_WORDS):
            updates["glare_exposure"] = False
        else:
            updates["glare_exposure"] = True

    if "plein soleil" in normalized or "forte lumiere" in normalized or "visualiser" in normalized:
        if any(word in normalized for word in _NO_WORDS):
            updates["sun_vision_difficulty"] = False
        elif "mal" in normalized or "diffic" in normalized or "oui" in normalized:
            updates["sun_vision_difficulty"] = True

    for pathology in _OcularPrevenciaTriggers:
        if pathology in normalized:
            updates["ocular_health"] = pathology
            break
    if "ras" in normalized and "ocul" in normalized:
        updates["ocular_health"] = "ras"

    return updates


def _normalize_updates(payload: RecommendationProfile | None) -> RecommendationProfile:
    payload = payload or {}
    normalized: RecommendationProfile = {}
    for raw_key, raw_value in payload.items():
        key = _KEY_ALIASES.get(str(raw_key), str(raw_key))
        if raw_value in (None, ""):
            continue

        if key in {"age"}:
            number = _to_float(raw_value)
            if number is not None:
                normalized[key] = int(number)
            continue

        if key in {"correction_total", "add_power", "od_og_diff"}:
            number = _to_float(raw_value)
            if number is not None:
                normalized[key] = number
            continue

        if key in {"innovation_sensitive", "adaptation_easy", "night_driving", "wants_sun_pair", "glare_exposure", "sun_vision_difficulty"}:
            boolean = _normalize_bool(raw_value)
            if boolean is not None:
                normalized[key] = boolean
            continue

        if key == "frame_type":
            mapped = _normalize_choice(raw_value, {
                "perce": "perce",
                "nylor": "nylor",
                "plastique": "plastique",
                "metallique": "metallique",
                "metal": "metallique",
            })
            if mapped:
                normalized[key] = mapped
            continue

        if key == "main_need":
            mapped = _normalize_choice(raw_value, {
                "transparence": "transparence",
                "lumiere bleue": "lumiere_bleue",
                "lumiere_bleue": "lumiere_bleue",
                "conduite soir": "conduite_soir",
                "conduite_soir": "conduite_soir",
                "rayures": "rayures",
                "nettoyage": "nettoyage",
            })
            if mapped:
                normalized[key] = mapped
            continue

        if key == "work_env":
            mapped = _normalize_choice(raw_value, {
                "interieur": "interieur",
                "exterieur": "exterieur",
                "mixte": "mixte",
                "interieur exterieur": "mixte",
            })
            if mapped:
                normalized[key] = mapped
            continue

        if key == "light_discomfort":
            mapped = _normalize_choice(raw_value, {
                "faible": "faible",
                "moyenne": "moyenne",
                "forte": "forte",
                "tres forte": "forte",
            })
            if mapped:
                normalized[key] = mapped
            continue

        if key == "head_eye_behavior":
            mapped = _normalize_choice(raw_value, {
                "head": "head",
                "eyes": "eyes",
                "mixed": "mixed",
                "tete": "head",
                "yeux": "eyes",
            })
            if mapped:
                normalized[key] = mapped
            continue

        if key == "computer_usage":
            mapped = _normalize_choice(raw_value, {
                "high": "high",
                "medium": "medium",
                "low": "low",
                "tres souvent": "high",
                "souvent": "high",
                "parfois": "medium",
                "rarement": "low",
            })
            if mapped:
                normalized[key] = mapped
            continue

        normalized[key] = str(raw_value)
    return normalized


def _required_fields(profile: RecommendationProfile) -> list[str]:
    required = ["age", "frame_type", "correction_total", "main_need", "work_env", "light_discomfort"]
    return [field for field in required if profile.get(field) in (None, "")]


def _recommend_family(age: int) -> tuple[str, str]:
    if age >= 42:
        return "Varilux", "Age >= 42 ans => famille Varilux."
    if age >= 31:
        return "Eyezen", "Age 31-41 ans => famille Eyezen."
    return "Simple foyer", "Age <= 30 ans => verres simple foyer."


def _recommend_index(frame_type: str | None, correction_total: float | None) -> tuple[str, str]:
    if frame_type == "perce":
        return "1.60", "Montage perce => indice minimum 1.60."

    if correction_total is None:
        return "1.56", "Indice par defaut en attente de correction detaillee."

    if correction_total < 225:
        return "1.50", "Correction totale < 225 => indice 1.50."
    if correction_total <= 400:
        return "1.56", "Correction totale 225-400 => indice 1.56."
    if correction_total <= 600:
        return "1.60", "Correction totale 425-600 => indice 1.60."
    return "1.67", "Correction totale > 600 => indice 1.67."


def _recommend_varilux_design(profile: RecommendationProfile, age: int) -> tuple[str, str]:
    add_power = _to_float(profile.get("add_power"))
    diff = _to_float(profile.get("od_og_diff"))
    innovation = bool(profile.get("innovation_sensitive")) if profile.get("innovation_sensitive") is not None else False
    adaptation_easy = bool(profile.get("adaptation_easy")) if profile.get("adaptation_easy") is not None else False
    head_eye = profile.get("head_eye_behavior")
    computer_usage = profile.get("computer_usage")
    night_driving = bool(profile.get("night_driving")) if profile.get("night_driving") is not None else False

    if innovation:
        return "Varilux XR", "Client sensible a l'innovation => Varilux XR prioritaire."
    if diff is not None and diff > 200:
        return "Varilux Physio 3.0", "Difference OD/OG > 200 => Varilux Physio 3.0 prioritaire."
    if add_power is not None and add_power > 2.50:
        return "Varilux X Design", "ADD > 2.50 => Varilux X Design."
    if add_power is not None and add_power >= 2.25:
        return "Varilux S Design", "ADD >= 2.25 => Varilux S Design."

    if computer_usage == "high":
        return "Varilux Comfort Max", "Usage ordinateur tres frequent => Varilux Comfort Max."
    if head_eye == "head":
        return "Varilux Liberty 3.0", "Client bouge surtout la tete => Varilux Liberty 3.0."
    if head_eye == "eyes":
        return "Varilux Physio 3.0", "Client utilise surtout les yeux => Varilux Physio 3.0."
    if night_driving or computer_usage == "medium" or head_eye == "mixed":
        return "Varilux Comfort 3.0", "Conduite de nuit / usage mixte => Varilux Comfort 3.0."
    if adaptation_easy and 42 <= age <= 50:
        return "Varilux X Design", "Recherche adaptation facile (42-50 ans) => Varilux X Design."

    return "Varilux Comfort 3.0", "Varilux par defaut: Comfort 3.0."


def _recommend_treatment(profile: RecommendationProfile) -> tuple[str, str]:
    main_need = profile.get("main_need")
    treatment_map = {
        "transparence": "Crizal Sapphire HR",
        "lumiere_bleue": "Crizal Prevencia",
        "conduite_soir": "Crizal Drive",
        "rayures": "Crizal Rock",
        "nettoyage": "Crizal Easy Pro",
    }
    treatment = treatment_map.get(str(main_need), "Crizal Sapphire HR")
    reason = f"Besoin principal ({main_need}) => {treatment}."

    ocular = _norm(str(profile.get("ocular_health", "")))
    if ocular and ocular != "ras":
        if any(trigger in ocular for trigger in _OcularPrevenciaTriggers):
            return "Crizal Prevencia", "Pathologie oculaire signalee => Crizal Prevencia prioritaire."

    return treatment, reason


def _recommend_color(profile: RecommendationProfile) -> tuple[str, str]:
    wants_sun_pair = profile.get("wants_sun_pair")
    glare_exposure = bool(profile.get("glare_exposure")) if profile.get("glare_exposure") is not None else False
    sun_diff = bool(profile.get("sun_vision_difficulty")) if profile.get("sun_vision_difficulty") is not None else False
    work_env = profile.get("work_env")
    light = profile.get("light_discomfort")

    if wants_sun_pair is True:
        if glare_exposure or sun_diff or light == "forte":
            return "Polarisant", "Paire soleil + reflets/difficulte en forte lumiere => Polarisant."
        return "Solaire", "Paire dediee soleil sans reflets majeurs => Solaire."

    if glare_exposure:
        return "Polarisant", "Reflets frequents (route/eau/soleil) => Polarisant."
    if light in {"forte", "moyenne"}:
        return "Transitions", "Gene lumineuse moyenne/forte => Transitions."
    if work_env in {"exterieur", "mixte"}:
        return "Transitions", "Usage exterieur ou mixte => Transitions."
    return "Blanc", "Usage interieur et gene lumineuse faible => Blanc."


def _build_recommendation(profile: RecommendationProfile) -> dict[str, Any]:
    age = int(_to_float(profile.get("age")) or 0)
    correction_total = _to_float(profile.get("correction_total"))
    frame_type = profile.get("frame_type")

    family, family_reason = _recommend_family(age)
    lens_type = family
    lens_reason = family_reason

    if family == "Varilux":
        lens_type, lens_reason = _recommend_varilux_design(profile, age)

    index_value, index_reason = _recommend_index(str(frame_type or ""), correction_total)
    treatment, treatment_reason = _recommend_treatment(profile)
    color, color_reason = _recommend_color(profile)

    filled = len([key for key, value in profile.items() if value not in (None, "")])
    confidence = round(min(0.99, 0.55 + min(10, filled) * 0.04), 2)

    rationale = [lens_reason, index_reason, treatment_reason, color_reason]
    return {
        "lens_type": lens_type,
        "index": index_value,
        "treatment": treatment,
        "color": color,
        "confidence": confidence,
        "rationale": rationale,
        "applied_rules": rationale,
    }


def _next_questions(profile: RecommendationProfile) -> list[dict[str, str]]:
    questions: list[dict[str, str]] = []

    prompts = {
        "age": "Quel age avez-vous ?",
        "frame_type": "Quel type de montage preferez-vous (Nylor, Plastique, Metallique, Perce) ?",
        "correction_total": "Quelle est la correction totale (ou plage) pour OD/OG ?",
        "add_power": "Quelle est la valeur d'addition (ADD) ?",
        "main_need": "Quel est votre besoin principal (transparence, lumiere bleue, conduite soir, rayures, nettoyage) ?",
        "ocular_health": "Etat oculaire actuel (RAS, glaucome, cataracte, etc.) ?",
        "work_env": "Vous travaillez surtout en interieur, exterieur, ou mixte ?",
        "light_discomfort": "Niveau de gene a la lumiere (faible, moyenne, forte) ?",
        "wants_sun_pair": "Souhaitez-vous une paire principalement pour le soleil ? (oui/non)",
        "glare_exposure": "Etes-vous souvent expose a des reflets genants ? (oui/non)",
        "sun_vision_difficulty": "Avez-vous du mal a voir en plein soleil ? (oui/non)",
        "head_eye_behavior": "Quand vous regardez sur les cotes, utilisez-vous plutot la tete ou les yeux ?",
        "innovation_sensitive": "Etes-vous sensible aux solutions innovantes ? (oui/non)",
        "night_driving": "Conduisez-vous souvent la nuit ? (oui/non)",
        "computer_usage": "Utilisez-vous l'ordinateur tres souvent, parfois, ou rarement ?",
    }

    for field in ["age", "frame_type", "correction_total", "main_need", "ocular_health", "work_env", "light_discomfort", "wants_sun_pair"]:
        if profile.get(field) in (None, ""):
            questions.append({"field": field, "question": prompts[field]})

    family = None
    if profile.get("age") not in (None, ""):
        family, _ = _recommend_family(int(_to_float(profile.get("age")) or 0))

    if family == "Varilux":
        for field in ["add_power", "head_eye_behavior", "innovation_sensitive", "night_driving", "computer_usage"]:
            if profile.get(field) in (None, ""):
                questions.append({"field": field, "question": prompts[field]})

    if profile.get("wants_sun_pair") is True:
        for field in ["glare_exposure", "sun_vision_difficulty"]:
            if profile.get(field) in (None, ""):
                questions.append({"field": field, "question": prompts[field]})

    return questions[:4]


def _render_response(profile: RecommendationProfile, recommendation: dict[str, Any] | None, next_questions: list[dict[str, str]]) -> str:
    if recommendation is None:
        q = "\n".join(f"- {item['question']}" for item in next_questions)
        return (
            "Parfait, je construis la recommandation selon le guide officiel. "
            "J'ai besoin de ces informations pour finaliser:\n"
            f"{q}"
        ).strip()

    summary = (
        "Recommandation pro (guide officiel)\n"
        f"- Type de verres: {recommendation['lens_type']}\n"
        f"- Indice: {recommendation['index']}\n"
        f"- Traitement: {recommendation['treatment']}\n"
        f"- Couleur: {recommendation['color']}\n"
        f"- Confiance: {recommendation['confidence']}"
    )
    reasons = "\n".join(f"- {item}" for item in recommendation.get("rationale", []))
    return f"{summary}\nJustification:\n{reasons}".strip()


def _build_rag_query(message: str, profile: RecommendationProfile) -> str:
    parts = [
        "scenario recommandation verres",
        str(message or "").strip(),
    ]
    for key in [
        "age",
        "frame_type",
        "correction_total",
        "add_power",
        "od_og_diff",
        "main_need",
        "ocular_health",
        "work_env",
        "light_discomfort",
        "wants_sun_pair",
        "glare_exposure",
        "sun_vision_difficulty",
        "head_eye_behavior",
        "innovation_sensitive",
        "night_driving",
        "computer_usage",
    ]:
        value = profile.get(key)
        if value not in (None, ""):
            parts.append(f"{key}={value}")
    return " | ".join(parts)


def process_recommendation_turn(
    *,
    session_id: str,
    message: str | None,
    incoming_profile: RecommendationProfile | None,
    retriever: VectorRAGRetriever,
    top_k: int = 4,
    reset: bool = False,
) -> dict[str, Any]:
    _load_sessions_once()

    if reset:
        _RECO_SESSIONS.pop(session_id, None)
        _save_sessions()

    state = _RECO_SESSIONS.setdefault(
        session_id,
        {
            "profile": {},
            "history": [],
            "last_question_field": None,
        },
    )

    base_profile = _normalize_updates(state.get("profile", {}))
    parsed = _extract_profile_updates_from_message(message or "") if message else {}
    incoming = _normalize_updates(incoming_profile)

    merged = dict(base_profile)
    merged.update(incoming)
    merged.update(parsed)

    last_question_field = state.get("last_question_field")
    if last_question_field and last_question_field not in merged and message:
        bool_answer = _normalize_bool(message)
        if bool_answer is not None:
            merged[last_question_field] = bool_answer

    state["profile"] = merged

    missing = _required_fields(merged)
    recommendation = None if missing else _build_recommendation(merged)
    next_questions = _next_questions(merged)
    state["last_question_field"] = next_questions[0]["field"] if next_questions else None

    query = _build_rag_query(message or "", merged)
    rag_results = retriever.search(query, top_k=max(1, min(int(top_k or 4), 8)))
    trimmed_rag = []
    for item in rag_results:
        trimmed_rag.append(
            {
                "doc_id": item.get("doc_id"),
                "score": item.get("score"),
                "source": item.get("source"),
                "text": str(item.get("text", ""))[:550],
            }
        )

    response = _render_response(merged, recommendation, next_questions)

    history_item = {
        "message": message,
        "profile": copy.deepcopy(merged),
        "recommendation": copy.deepcopy(recommendation),
    }
    state.setdefault("history", []).append(history_item)
    state["history"] = state["history"][-_RECO_HISTORY_LIMIT:]
    state["updated_at"] = time.time()

    _save_sessions()

    if message:
        _append_recommendation_event(
            session_id=session_id,
            user_text=str(message),
            assistant_text=response,
            metadata={
                "missing_fields": list(missing),
                "next_question_fields": [item["field"] for item in next_questions],
                "recommendation": copy.deepcopy(recommendation),
                "profile": copy.deepcopy(merged),
                "rag_sources": [item.get("source") for item in trimmed_rag if item.get("source")],
            },
        )

    return {
        "session_id": session_id,
        "response": response,
        "profile": merged,
        "extracted_updates": parsed,
        "incoming_updates": incoming,
        "missing_fields": missing,
        "next_questions": [item["question"] for item in next_questions],
        "next_question_fields": [item["field"] for item in next_questions],
        "recommendation": recommendation,
        "rag_results": trimmed_rag,
    }


def reset_recommendation_sessions(session_id: str | None = None) -> None:
    _load_sessions_once()

    if session_id:
        _RECO_SESSIONS.pop(session_id, None)
    else:
        _RECO_SESSIONS.clear()

    _save_sessions()


def recommendation_stats() -> dict[str, int]:
    _load_sessions_once()
    return {
        "active_sessions": len(_RECO_SESSIONS),
        "conversation_events": _count_jsonl_lines(_RECO_LOG_PATH),
    }
