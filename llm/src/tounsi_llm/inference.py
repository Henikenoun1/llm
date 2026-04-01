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
from .config import CFG, DOMAIN_CFG, FEW_SHOTS_CFG, logger
from .memory import ConversationMemoryStore
from .rag import VectorRAGRetriever
from .tools import ToolRegistry, get_tool_registry


PHONE_RE = re.compile(r"\+?216[\s\-]?\d{2}[\s\-]?\d{3}[\s\-]?\d{3}")
ORDER_RE = re.compile(r"ORD-[A-Z0-9]{5,}")
DATE_RE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")
TIME_RE = re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b")
CUSTOMER_ID_RE = re.compile(r"\b(?:CLI|CLT|CLIENT|CUST)[-_]?[A-Z0-9]{3,}\b", re.IGNORECASE)

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
        if char.isdigit() or ("a" <= char <= "z") or ("\u0600" <= char <= "\u06FF") or char.isspace():
            kept_chars.append(char)
        else:
            kept_chars.append(" ")
    return re.sub(r"\s+", " ", "".join(kept_chars)).strip()


def infer_intent(text: str) -> str:
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

    if has_keyword(intent_keywords.get("greeting", [])) and not has_keyword(business_keywords):
        return "greeting"

    ordered_intents = [
        "price_inquiry",
        "order_creation",
        "order_tracking",
        "appointment_booking",
        "store_info",
        "product_info",
        "thanks",
    ]
    for intent in ordered_intents:
        if has_keyword(intent_keywords.get(intent, [])):
            return intent
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

    if match := ORDER_RE.search(text):
        slots["order_id"] = match.group(0)
    if match := re.search(r"\b1\.(?:50|56|60|67)\b", text):
        slots["index"] = match.group(0)
    if match := PHONE_RE.search(text):
        slots["phone"] = match.group(0)
    if match := DATE_RE.search(text):
        slots["date"] = match.group(0)
    if match := TIME_RE.search(text):
        slots["time_slot"] = match.group(0)
    if match := CUSTOMER_ID_RE.search(text):
        slots["customer_id"] = match.group(0)

    for slot_name, pattern in DOMAIN_CFG.get("slot_patterns", {}).items():
        if slot_name in slots:
            continue
        try:
            compiled = re.compile(str(pattern), re.IGNORECASE)
        except re.error:
            logger.warning("Invalid slot pattern for %s: %s", slot_name, pattern)
            continue
        if match := compiled.search(text):
            slots[slot_name] = match.group(0)

    city = _extract_alias(text, DOMAIN_CFG.get("location_aliases", {}))
    if city:
        slots["city"] = city

    product = _extract_alias(text, DOMAIN_CFG.get("product_aliases", {}))
    if product:
        slots["product"] = product

    return slots


def _merge_slots(base: dict[str, Any] | None, updates: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(base or {})
    for key, value in (updates or {}).items():
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
    active_intent = str(session_state.get("active_intent") or "")
    open_form = bool(session_state.get("open_form"))
    known_slots = session_state.get("slots", {}) if isinstance(session_state.get("slots"), dict) else {}

    resolved_intent = intent
    carried_slots: dict[str, Any] = {}
    if active_intent and open_form and intent in {"unknown", active_intent}:
        resolved_intent = active_intent
        carried_slots = known_slots
    elif intent != "unknown" and intent == active_intent and open_form:
        carried_slots = known_slots
    elif intent == "unknown" and active_intent and len(text.split()) <= 8 and extracted_slots:
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
    required = DOMAIN_CFG.get("required_slots", {}).get(intent, [])
    missing = [slot for slot in required if not slots.get(slot)]
    tool_name = DOMAIN_CFG.get("intent_to_tool", {}).get(intent)
    if missing or not tool_name:
        return None, {}, missing

    tool_args = {slot: slots.get(slot) for slot in required}
    for optional in ["coating", "city", "phone", "date", "time_slot"]:
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

    def score(example: dict[str, Any]) -> float:
        example_user = str(example.get("user", ""))
        example_intent = str(example.get("intent", ""))
        base = 1.0 if example_intent == intent else 0.0
        a_tokens = set(_norm_for_matching(user_text).split())
        b_tokens = set(_norm_for_matching(example_user).split())
        overlap = len(a_tokens & b_tokens) / max(len(a_tokens), len(b_tokens), 1)
        return base + overlap

    ranked = sorted(FEW_SHOTS_CFG, key=score, reverse=True)
    selected = []
    for example in ranked[: CFG.few_shot_top_k]:
        if example.get("user") and example.get("assistant"):
            selected.append(
                {"role": "user", "content": str(example["user"])}
            )
            selected.append(
                {"role": "assistant", "content": str(example["assistant"])}
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


def _strip_leaked_english(text: str) -> str:
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

    text = normalize_text(user_text)
    raw_intent = infer_intent(text)
    extracted_slots = extract_slots(text)
    session_state = memory_store.get_session_state(session_id) if memory_store and session_id else {}
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
    auto_execute_tool = _should_execute_tool_for_mode(intent, missing_slots, runtime_mode)
    tool_result = tool_registry.execute(tool_name, tool_args) if tool_name and auto_execute_tool else None

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

    messages: list[dict[str, str]] = [
        {"role": "system", "content": DOMAIN_CFG.get("system_prompt", "")},
        {"role": "system", "content": DOMAIN_CFG.get("humanized_prompt", "")},
        {"role": "system", "content": _mode_instruction(runtime_mode)},
    ]
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
        response = _strip_leaked_english(response)
        response = _humanize(response)
        response = _truncate(response)

    if _is_garbage_response(response):
        response = _get_fallback_response("unclear")

    tool_status = str((tool_result or {}).get("status", "")).lower() if isinstance(tool_result, dict) else ""
    needs_human_review = bool(missing_slots) or bool(tool_name and not auto_execute_tool)
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
        "correction_applied": correction_applied,
    }
