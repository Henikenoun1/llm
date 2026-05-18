#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "tounsi_raw"
ENRICH_DIR = RAW_DIR / "enrichment"
PROCESSED_DIR = ROOT / "data" / "processed"
HISTORY_DIR = ROOT / "data" / "history"
CONFIGS_DIR = ROOT / "configs"
REPORTS_DIR = ROOT / "reports"
MANIFESTS_DIR = ROOT / "artifacts" / "manifests"

TDC_DATASET_ID = "Syrinesmati/tunisian-dialect-corpus"
TDC_RAW_PATH = ENRICH_DIR / "tunisian_dialect_corpus.jsonl"
CUSTOM_SFT_PATH = PROCESSED_DIR / "custom_tounsi_conversations.jsonl"
FALLBACK_DPO_PATH = HISTORY_DIR / "feedback_dpo.jsonl"
PIPELINE_MANIFEST_PATH = MANIFESTS_DIR / "axolotl_pipeline_state.json"

SELF_SUP_TRAIN = PROCESSED_DIR / "self_sup_train.jsonl"
SELF_SUP_VAL = PROCESSED_DIR / "self_sup_val.jsonl"
SFT_TRAIN = PROCESSED_DIR / "sft_train.jsonl"
SFT_VAL = PROCESSED_DIR / "sft_val.jsonl"
SFT_TEST = PROCESSED_DIR / "sft_test.jsonl"
INTENT_SLOT_TRAIN = PROCESSED_DIR / "intent_slot_train.jsonl"
INTENT_SLOT_VAL = PROCESSED_DIR / "intent_slot_val.jsonl"
DPO_TRAIN = PROCESSED_DIR / "dpo_train.jsonl"
DPO_VAL = PROCESSED_DIR / "dpo_val.jsonl"
ORPO_TRAIN = PROCESSED_DIR / "orpo_train.jsonl"
ORPO_VAL = PROCESSED_DIR / "orpo_val.jsonl"
INTENT_SLOT_REPORT_PATH = REPORTS_DIR / "intent_slot_data_prep.json"
COMMANDES_INTENTS_PATH = ROOT / "data" / "config" / "commandes_intents.jsonl"

SSF_OUTPUT_DIR = Path("outputs") / "1_ssf"
SFT_OUTPUT_DIR = Path("outputs") / "2_sft"
INTENT_SLOT_OUTPUT_DIR = Path("outputs") / "3_intent_slot"
ORPO_OUTPUT_DIR = Path("outputs") / "4_orpo"
SSF_MERGED_DIR = SSF_OUTPUT_DIR / "merged"
SFT_MERGED_DIR = SFT_OUTPUT_DIR / "merged"
INTENT_SLOT_MERGED_DIR = INTENT_SLOT_OUTPUT_DIR / "merged"

SEED = 42

PHASE1_TEMPLATE_NOTE = (
    "# NOTE: Unsloth was requested, but Axolotl's documented Unsloth support does not "
    "cover the detected Qwen checkpoint lineage.\n"
    "# The runtime config therefore stays on the T4-safe QLoRA path so launch remains reliable.\n"
)


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _rel(value: str | Path | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        try:
            return path.relative_to(ROOT).as_posix()
        except ValueError:
            return path.as_posix()
    return path.as_posix()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _fingerprint_messages(messages: list[dict[str, str]]) -> str:
    return json.dumps(messages, ensure_ascii=False, sort_keys=True)


def _detect_latest_checkpoint() -> str | None:
    candidates: list[Path] = []
    for root_name in ("artifacts", "outputs"):
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_dir():
                continue
            if (path / "trainer_state.json").exists():
                candidates.append(path)
                continue
            if path.name.startswith("checkpoint-") and (path / "adapter_model.safetensors").exists():
                candidates.append(path)

    if not candidates:
        return None
    latest = max(candidates, key=lambda item: item.stat().st_mtime)
    return _rel(latest)


def _detect_base_model(latest_checkpoint: str | None) -> str:
    if latest_checkpoint:
        adapter_config = ROOT / latest_checkpoint / "adapter_config.json"
        if adapter_config.exists():
            data = json.loads(adapter_config.read_text(encoding="utf-8"))
            model_name = str(data.get("base_model_name_or_path") or "").strip()
            if model_name:
                return model_name

    sys.path.insert(0, str(ROOT))
    from src.tounsi_llm.config import CFG  # pylint: disable=import-outside-toplevel

    return str(CFG.base_model)


def _detect_phase1_lora_settings(latest_checkpoint: str | None) -> dict[str, float | int]:
    settings: dict[str, float | int] = {
        "lora_r": 32,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
    }
    if not latest_checkpoint:
        return settings

    adapter_config = ROOT / latest_checkpoint / "adapter_config.json"
    if not adapter_config.exists():
        return settings

    try:
        data = json.loads(adapter_config.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return settings

    lora_r = data.get("r")
    if isinstance(lora_r, (int, float)) and lora_r > 0:
        settings["lora_r"] = int(lora_r)

    lora_alpha = data.get("lora_alpha")
    if isinstance(lora_alpha, (int, float)) and lora_alpha > 0:
        settings["lora_alpha"] = int(lora_alpha) if float(lora_alpha).is_integer() else float(lora_alpha)

    lora_dropout = data.get("lora_dropout")
    if isinstance(lora_dropout, (int, float)) and lora_dropout >= 0:
        settings["lora_dropout"] = float(lora_dropout)

    return settings


def _extract_tdc_text(row: dict[str, Any]) -> str:
    for key in ("text", "Tweet", "tweet", "sentence", "content", "utterance", "comment"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_label(value: Any) -> str:
    lowered = str(value).strip().lower()
    if lowered in {"1", "positive", "pos", "joy", "happy"}:
        return "positive"
    if lowered in {"0", "negative", "neg", "-1", "anger", "sad"}:
        return "negative"
    if lowered in {"2", "neutral"}:
        return "neutral"
    return "neutral"


def _download_tdc() -> dict[str, Any]:
    result = {"dataset": TDC_DATASET_ID, "downloaded_rows": 0, "path": _rel(TDC_RAW_PATH), "used_cache": False}
    ENRICH_DIR.mkdir(parents=True, exist_ok=True)

    try:
        dataset = load_dataset(TDC_DATASET_ID, split="train")
        rows = [dict(row) for row in dataset]
        _write_jsonl(TDC_RAW_PATH, rows)
        result["downloaded_rows"] = len(rows)
        return result
    except Exception as exc:  # pragma: no cover - network dependent
        if TDC_RAW_PATH.exists():
            result["used_cache"] = True
            result["warning"] = f"download_failed_using_cache: {exc}"
            result["downloaded_rows"] = _jsonl_count(TDC_RAW_PATH)
            return result
        raise RuntimeError(f"Unable to download {TDC_DATASET_ID}: {exc}") from exc


def _tdc_reply_for_label(label: str, index: int) -> str:
    positive = [
        "يعطيك الصحة، كلامك يبين إلي الجو باهي ومودك مريقل.",
        "واضح من كلامك إلي عندك إحساس إيجابي وبرشة ارتياح.",
        "مفهوم، الرسالة متاعك فيها نفس باهي ومشاعر مزيانة.",
    ]
    negative = [
        "نفهمك، من لهجتك باين إلي الحكاية مضايقتك ومقلقتك شوية.",
        "واضح إنك موش مرتاح من الموضوع وهذا ظاهر في كلامك.",
        "مفهوم، الرسالة تعكس تعب ولا انزعاج ونجم نلقط هذا من الصياغة.",
    ]
    neutral = [
        "وصلت، كلامك يوصف الوضع بطريقة مباشرة أكثر منو عاطفية.",
        "واضح، الرسالة محايدة وتركز على المعلومة أكثر من الإحساس.",
        "مفهوم، الصياغة عندك وصفية وعادية من غير شحنة كبيرة.",
    ]
    bank = {"positive": positive, "negative": negative, "neutral": neutral}[label]
    return bank[index % len(bank)]


def _augment_custom_sft_from_tdc(limit: int = 3000) -> dict[str, Any]:
    existing_rows = _read_jsonl(CUSTOM_SFT_PATH)
    seen = {_fingerprint_messages(row.get("messages", [])) for row in existing_rows if row.get("messages")}
    generated = 0

    for index, row in enumerate(_read_jsonl(TDC_RAW_PATH)):
        text = _extract_tdc_text(row)
        if len(text) < 4:
            continue
        label = _normalize_label(row.get("label") or row.get("sentiment"))
        messages = [
            {
                "role": "system",
                "content": "إنت assistant تحكي بالتونسي بطريقة طبيعية ومختصرة.",
            },
            {"role": "user", "content": text},
            {"role": "assistant", "content": _tdc_reply_for_label(label, index)},
        ]
        fingerprint = _fingerprint_messages(messages)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        existing_rows.append(
            {
                "messages": messages,
                "metadata": {
                    "source": "tdc_hf",
                    "dataset": TDC_DATASET_ID,
                    "label": label,
                },
            }
        )
        generated += 1
        if generated >= limit:
            break

    _write_jsonl(CUSTOM_SFT_PATH, existing_rows)
    return {
        "custom_sft_path": _rel(CUSTOM_SFT_PATH),
        "generated_rows": generated,
        "total_rows": len(existing_rows),
    }


def _current_dpo_example_count() -> int:
    count = 0
    for path in (
        RAW_DIR / "dpo_tunisian_derja.jsonl",
        RAW_DIR / "dpo_mt_en_tn_msa_v2.jsonl",
        FALLBACK_DPO_PATH,
    ):
        count += _jsonl_count(path)
    return count


def _build_rule_based_dpo_pairs(target_rows: int = 500) -> list[dict[str, str]]:
    prompts = [
        "aslema, chnowa tal9a 3al statut mta3 el commande?",
        "bonjour, nheb n3ref idha fama disponibilité tawa.",
        "slm, tnajem t9olli wa9tech livraison taqribiya?",
        "aslema, chnowa يلزم باش نكمّل create_order?",
        "bonjour, nheb confirmation lel référence hedhi.",
        "slm, famech stock wala fabrication?",
        "aslema, nheb recap sghir b tounsi.",
        "bonjour, chnowa أول معلومة لازم نعطيهالك؟",
        "slm, tnajem tجاوبني بنفس الأسلوب التونسي؟",
        "aslema, فهمني الحكاية بكلام بسيط.",
    ]
    chosen_templates = [
        "عسلامة، نثبتلك المعطيات أول وبعدها نجاوبك بالتونسي ومن غير ما نخمن.",
        "أكيد، نعطيك الجواب الواضح أما كان بالمعلومة المؤكدة برك.",
        "حاضر، نفسرلك بالتونسي وبطريقة قصيرة وواضحة.",
        "تو نعاونك، أما نحتاج المعلومة الأساسية باش نثبت الوضعية.",
        "وصلت، نجاوبك بدارجة تونسية وبأسلوب مهني.",
    ]
    rejected_templates = [
        "Bonjour, je vais vous répondre en français standard après vérification.",
        "سأقوم بتزويدك بإجابة واضحة باللغة العربية الفصحى بعد التثبت.",
        "Je vais vous donner une réponse professionnelle en français.",
        "سأشرح لك الأمر بطريقة رسمية باللغة العربية الفصحى.",
        "Nous allons confirmer les informations puis répondre en français.",
    ]

    rows: list[dict[str, str]] = []
    random.seed(SEED)
    for idx in range(target_rows):
        prompt = prompts[idx % len(prompts)]
        chosen = chosen_templates[idx % len(chosen_templates)]
        rejected = rejected_templates[idx % len(rejected_templates)]
        rows.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "source": "rule_based_bootstrap",
            }
        )
    return rows


def _ensure_fallback_dpo(min_examples: int = 1000, bootstrap_examples: int = 500) -> dict[str, Any]:
    current_count = _current_dpo_example_count()
    result = {"current_examples": current_count, "generated_examples": 0, "path": _rel(FALLBACK_DPO_PATH)}
    if current_count >= min_examples:
        return result

    existing_rows = _read_jsonl(FALLBACK_DPO_PATH)
    rows = existing_rows + _build_rule_based_dpo_pairs(target_rows=bootstrap_examples)
    _write_jsonl(FALLBACK_DPO_PATH, rows)
    result["generated_examples"] = bootstrap_examples
    result["current_examples"] = current_count + bootstrap_examples
    return result


def _run_existing_prepare() -> dict[str, Any]:
    cmd = [sys.executable, "scripts/train.py", "--stage", "prepare"]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return {
        "self_sup_train_rows": _jsonl_count(SELF_SUP_TRAIN),
        "self_sup_val_rows": _jsonl_count(SELF_SUP_VAL),
        "sft_train_rows": _jsonl_count(SFT_TRAIN),
        "sft_val_rows": _jsonl_count(SFT_VAL),
        "dpo_train_rows": _jsonl_count(DPO_TRAIN),
        "dpo_val_rows": _jsonl_count(DPO_VAL),
    }


def _build_orpo_split(source_path: Path, destination_path: Path) -> int:
    rows: list[dict[str, Any]] = []
    for row in _read_jsonl(source_path):
        prompt = str(row.get("prompt") or "").strip()
        chosen = str(row.get("chosen") or "").strip()
        rejected = str(row.get("rejected") or "").strip()
        if not (prompt and chosen and rejected):
            continue
        rows.append(
            {
                "prompt": prompt,
                "chosen": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": chosen},
                ],
                "rejected": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": rejected},
                ],
            }
        )
    _write_jsonl(destination_path, rows)
    return len(rows)


def _build_orpo_dataset() -> dict[str, Any]:
    train_rows = _build_orpo_split(DPO_TRAIN, ORPO_TRAIN)
    val_rows = _build_orpo_split(DPO_VAL, ORPO_VAL)
    return {
        "orpo_train_rows": train_rows,
        "orpo_val_rows": val_rows,
        "orpo_train_path": _rel(ORPO_TRAIN),
        "orpo_val_path": _rel(ORPO_VAL),
    }


def _build_intent_slot_dataset(
    *,
    max_sft_windows: int = 2200,
    max_delivery_entries: int = 220,
    max_lens_entries: int = 900,
) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))

    from src.tounsi_llm.config import DOMAIN_CFG, FEW_SHOTS_CFG  # pylint: disable=import-outside-toplevel
    from src.tounsi_llm.domain_utils import (  # pylint: disable=import-outside-toplevel
        canonicalize_intent,
        canonicalize_slots,
        normalize_messages,
    )
    from src.tounsi_llm.inference import (  # pylint: disable=import-outside-toplevel
        extract_slots,
        infer_intent,
        route_to_tool,
    )
    from src.tounsi_llm.rag_assets import (  # pylint: disable=import-outside-toplevel
        load_delivery_rag_entries,
        load_lens_rag_entries,
        normalize_lookup,
    )

    required_slots_map = {
        canonicalize_intent(intent): list(values)
        for intent, values in DOMAIN_CFG.get("required_slots", {}).items()
        if isinstance(values, list)
    }
    intent_to_tool = {
        canonicalize_intent(intent): str(tool_name)
        for intent, tool_name in DOMAIN_CFG.get("intent_to_tool", {}).items()
        if str(tool_name).strip()
    }
    allowed_intents = sorted(set(required_slots_map) | set(intent_to_tool) | {"get_num_client"})
    canonical_slots = [
        "num_client",
        "order_id",
        "reference",
        "lens_code",
        "product",
        "material",
        "treatment",
        "diameter",
        "color",
        "index",
        "quantity",
        "priority",
        "agence",
        "secteur",
        "delivery_slot",
        "phone",
        "date",
        "time_slot",
        "od_sphere",
        "od_cyl",
        "od_axis",
        "og_sphere",
        "og_cyl",
        "og_axis",
        "addition",
    ]

    required_lines = []
    for intent in allowed_intents:
        required = required_slots_map.get(intent, ["num_client"] if intent == "get_num_client" else [])
        tool_name = intent_to_tool.get(intent, "none")
        required_lines.append(
            f"- {intent}: required_slots={', '.join(required) if required else 'none'}; tool={tool_name}"
        )
    system_prompt = "\n".join(
        [
            "Tu es un extracteur d'intent et de slots pour SIVO Essilor.",
            "Tu reponds uniquement en JSON valide, sans prose.",
            'Schema de sortie: {"intent":"...", "slots":{}, "required_slots":[], "missing_slots":[], "tool_name":null, "can_execute":false, "tool_args":{}}',
            f"Intents autorises: {', '.join(allowed_intents)}",
            f"Slots canoniques: {', '.join(canonical_slots)}",
            "Rappels:",
            "- ne jamais inventer un slot absent",
            "- conserver uniquement les slots detectes dans la conversation",
            "- missing_slots doit suivre required_slots pour l'intent detecte",
            "- tool_name est le tool metier associe a l'intent, ou null si aucun",
            "- can_execute=true seulement si tous les required_slots sont presents",
            "Required slots et tool mapping:",
            *required_lines,
        ]
    )

    delivery_entries = load_delivery_rag_entries()
    lens_entries = load_lens_rag_entries()
    print(
        f"[intent-slot] bootstrap sources: delivery_rag={len(delivery_entries)} lens_rag={len(lens_entries)}",
        flush=True,
    )
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()

    def build_payload(intent: str, slots: dict[str, Any]) -> dict[str, Any] | None:
        intent = canonicalize_intent(intent)
        slots = canonicalize_slots(slots)
        if intent in {"unknown", "greeting", "thanks"}:
            return None

        if intent == "get_num_client":
            required = ["num_client"]
            missing = [] if slots.get("num_client") else ["num_client"]
            tool_name = None
            tool_args: dict[str, Any] = {}
        else:
            required = list(required_slots_map.get(intent, []))
            tool_name = intent_to_tool.get(intent)
            _, executable_args, missing = route_to_tool(intent, slots)
            tool_args = executable_args if tool_name and not missing else {}

        return {
            "intent": intent,
            "slots": slots,
            "required_slots": required,
            "missing_slots": missing,
            "tool_name": tool_name,
            "can_execute": bool(tool_name and not missing),
            "tool_args": tool_args,
        }

    def rag_context(intent: str, slots: dict[str, Any], conversation_text: str) -> str:
        snippets: list[str] = []
        normalized_text = normalize_lookup(conversation_text)
        code = str(slots.get("lens_code") or slots.get("reference") or "").strip().upper()
        product = normalize_lookup(slots.get("product"))
        agence = normalize_lookup(slots.get("agence"))
        secteur = normalize_lookup(slots.get("secteur"))

        if intent in {"reference_confirmation", "availability_inquiry", "create_order"} or code or product:
            for entry in lens_entries:
                entry_code = str(entry.get("code", "")).strip().upper()
                entry_name = str(entry.get("name", "")).strip()
                name_norm = normalize_lookup(entry_name)
                if code and code == entry_code:
                    snippets.append(
                        "catalogue_verre: "
                        f"code={entry_code}; nom={entry_name}; matiere={entry.get('material') or 'n/a'}; "
                        f"geometrie={entry.get('geometry') or 'n/a'}; diametre={entry.get('diameter') or 'n/a'}"
                    )
                    break
                if product and product in name_norm:
                    snippets.append(
                        "catalogue_verre: "
                        f"code={entry_code}; nom={entry_name}; matiere={entry.get('material') or 'n/a'}; "
                        f"geometrie={entry.get('geometry') or 'n/a'}; diametre={entry.get('diameter') or 'n/a'}"
                    )
                    break
                if code and code and code.lower() in str(entry.get("text", "")).lower():
                    snippets.append(
                        "catalogue_verre: "
                        f"code={entry_code}; nom={entry_name}; matiere={entry.get('material') or 'n/a'}; "
                        f"geometrie={entry.get('geometry') or 'n/a'}; diametre={entry.get('diameter') or 'n/a'}"
                    )
                    break

        if intent in {"delivery_schedule", "order_tracking"} or agence or secteur:
            for entry in delivery_entries:
                entry_agence = normalize_lookup(entry.get("agence"))
                entry_secteur = normalize_lookup(entry.get("secteur"))
                if agence and agence not in entry_agence:
                    continue
                if secteur and secteur not in entry_secteur:
                    continue
                if not agence and not secteur and not (
                    entry_agence and entry_agence in normalized_text or entry_secteur and entry_secteur in normalized_text
                ):
                    continue
                snippets.append(
                    "planning_livraison: "
                    f"agence={entry.get('agence')}; secteur={entry.get('secteur')}; "
                    f"premier_creneau={entry.get('premier_creneau') or 'n/a'}; "
                    f"creneaux={', '.join(entry.get('tous_creneaux', [])[:3]) or 'n/a'}"
                )
                break

        return "\n".join(snippets[:2])

    def render_prompt(conversation: list[dict[str, str]], context: str) -> str:
        history_lines = []
        for message in conversation[-4:]:
            role = "Client" if message.get("role") == "user" else "Agent"
            content = str(message.get("content") or "").strip()
            if content:
                history_lines.append(f"{role}: {content}")
        parts = [
            "Analyse l'echange call-center suivant et extrais l'intent metier + les slots canoniques.",
            "Reponds en JSON uniquement.",
        ]
        if context:
            parts.append(f"[contexte_rag]\n{context}")
        parts.append(f"[conversation]\n{'\n'.join(history_lines)}")
        return "\n\n".join(parts)

    def add_example(
        conversation: list[dict[str, str]],
        intent: str,
        slots: dict[str, Any],
        source: str,
    ) -> None:
        normalized_conversation = [
            message
            for message in normalize_messages(conversation)
            if message.get("role") in {"user", "assistant"} and str(message.get("content") or "").strip()
        ]
        if not normalized_conversation:
            return
        latest_user = next(
            (message.get("content", "") for message in reversed(normalized_conversation) if message.get("role") == "user"),
            "",
        )
        if not latest_user:
            return

        payload = build_payload(intent, slots)
        if not payload:
            return

        context = rag_context(payload["intent"], payload["slots"], "\n".join(m["content"] for m in normalized_conversation))
        prompt = render_prompt(normalized_conversation, context)
        assistant_content = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_content},
        ]
        fingerprint = _fingerprint_messages(messages)
        if fingerprint in seen:
            return
        seen.add(fingerprint)
        rows.append({"messages": messages})
        source_counts[source] += 1
        intent_counts[payload["intent"]] += 1

    def masked_variants(intent: str, slots: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        variants: list[tuple[str, dict[str, Any]]] = []
        num_client = str(slots.get("num_client") or "").strip()
        order_id = str(slots.get("order_id") or "").strip()
        product = str(slots.get("product") or "").strip()
        material = str(slots.get("material") or "").strip()
        diameter = str(slots.get("diameter") or "").strip()
        reference = str(slots.get("reference") or slots.get("lens_code") or "").strip()
        agence = str(slots.get("agence") or "").strip()
        secteur = str(slots.get("secteur") or "").strip()

        if intent == "order_tracking":
            if order_id:
                variants.append((f"aslema, nheb netabba3 commande {order_id}", {"order_id": order_id}))
            if num_client:
                variants.append((f"bonjour {num_client}, nheb suivi commande", {"num_client": num_client}))
        elif intent in {"create_order", "order_creation"}:
            if num_client:
                variants.append((f"bonjour {num_client}, nheb na3di commande jdida", {"num_client": num_client}))
            if product:
                partial_slots = {"product": product}
                if material:
                    partial_slots["material"] = material
                if diameter:
                    partial_slots["diameter"] = diameter
                variants.append((f"nheb na3di commande {product}", partial_slots))
        elif intent == "reference_confirmation" and reference:
            variants.append((f"tnajem tconfirmili code verre {reference} ?", {"reference": reference, "lens_code": reference}))
        elif intent == "availability_inquiry" and reference:
            variants.append((f"aslema, dispo {reference} tawa ?", {"reference": reference, "lens_code": reference}))
        elif intent == "delivery_schedule" and agence:
            prompt = f"planning livraison agence {agence}"
            partial_slots: dict[str, Any] = {"agence": agence}
            if secteur:
                prompt += f" secteur {secteur}"
                partial_slots["secteur"] = secteur
            variants.append((prompt, partial_slots))
        return variants

    for row in _read_jsonl(COMMANDES_INTENTS_PATH):
        user_text = str(row.get("opticien") or row.get("user") or row.get("client") or "").strip()
        intent = canonicalize_intent(row.get("intent"))
        slots = canonicalize_slots(row.get("slots") if isinstance(row.get("slots"), dict) else {})
        if user_text:
            add_example([{"role": "user", "content": user_text}], intent, slots, "commandes_intents")
        for prompt_text, partial_slots in masked_variants(intent, slots):
            add_example([{"role": "user", "content": prompt_text}], intent, partial_slots, "commandes_masked")
    print(f"[intent-slot] loaded commandes sources rows={len(rows)}", flush=True)

    for row in FEW_SHOTS_CFG:
        user_text = str(row.get("opticien") or row.get("user") or row.get("client") or "").strip()
        intent = canonicalize_intent(row.get("intent"))
        if not user_text or intent in {"unknown", "greeting", "thanks"}:
            continue
        slots = canonicalize_slots(extract_slots(user_text))
        add_example([{"role": "user", "content": user_text}], intent, slots, "few_shots")
    print(f"[intent-slot] after few_shots rows={len(rows)}", flush=True)

    sft_candidates: list[tuple[list[dict[str, str]], str, dict[str, Any]]] = []
    sft_candidate_cap = max_sft_windows * 3
    stop_collecting_sft = False
    for path in (SFT_TRAIN,):
        for row in _read_jsonl(path):
            conversation = normalize_messages(row.get("messages", []))
            dialogue = [message for message in conversation if message.get("role") in {"user", "assistant"}]
            for index, message in enumerate(dialogue):
                if message.get("role") != "user":
                    continue
                window = dialogue[max(0, index - 3) : index + 1]
                user_bundle = "\n".join(item.get("content", "") for item in window if item.get("role") == "user").strip()
                if len(user_bundle) < 6:
                    continue
                slots = canonicalize_slots(extract_slots(user_bundle))
                intent = canonicalize_intent(infer_intent(user_bundle, extracted_slots=slots))
                if intent in {"unknown", "greeting", "thanks"}:
                    continue
                if intent == "get_num_client" and not slots and "client" not in user_bundle.lower() and "commande" not in user_bundle.lower():
                    continue
                if intent == "order_tracking" and not (slots.get("num_client") or slots.get("order_id")):
                    continue
                if intent in {"reference_confirmation", "availability_inquiry"} and not (
                    slots.get("reference") or slots.get("lens_code")
                ):
                    continue
                if intent == "delivery_schedule" and not (slots.get("agence") or slots.get("secteur")):
                    continue
                if intent in {"create_order", "price_inquiry", "product_info"}:
                    continue
                if intent == "appointment_booking" and not (
                    slots.get("city") or slots.get("date") or slots.get("time_slot") or slots.get("phone")
                ):
                    continue
                if intent == "store_info" and not slots.get("city"):
                    continue
                sft_candidates.append((window, intent, slots))
                if len(sft_candidates) >= sft_candidate_cap:
                    stop_collecting_sft = True
                    break
            if stop_collecting_sft:
                break
        if stop_collecting_sft:
            break

    random.seed(SEED)
    random.shuffle(sft_candidates)
    for window, intent, slots in sft_candidates[:max_sft_windows]:
        add_example(window, intent, slots, "sft_dialogue")
    print(
        f"[intent-slot] after sft windows rows={len(rows)} sampled_windows={min(len(sft_candidates), max_sft_windows)}",
        flush=True,
    )

    for index, entry in enumerate(delivery_entries[:max_delivery_entries]):
        agence = str(entry.get("agence") or "").strip()
        secteur = str(entry.get("secteur") or "").strip()
        if not (agence and secteur):
            continue
        num_client = str(6100 + index)
        order_id = f"CMD26{index:08d}"
        prompts = [
            (
                f"planning livraison agence {agence} secteur {secteur}, prochain creneau ?",
                "delivery_schedule",
                {"agence": agence, "secteur": secteur},
            ),
            (
                f"bonjour {num_client} nheb netabba3 {order_id}, surtout livraison agence {agence} secteur {secteur}",
                "order_tracking",
                {"num_client": num_client, "order_id": order_id, "agence": agence, "secteur": secteur},
            ),
            (
                f"aslema, agence {agence} secteur {secteur}, fama passage ba3d 12:00 ?",
                "delivery_schedule",
                {"agence": agence, "secteur": secteur, "time_slot": "12:00"},
            ),
        ]
        for prompt_text, intent, slots in prompts:
            add_example([{"role": "user", "content": prompt_text}], intent, slots, "delivery_rag")
    print(f"[intent-slot] after delivery rag rows={len(rows)}", flush=True)

    for index, entry in enumerate(lens_entries[:max_lens_entries]):
        code = str(entry.get("code") or "").strip().upper()
        name = str(entry.get("name") or "").strip()
        material = str(entry.get("material") or "").strip()
        diameter = str(entry.get("diameter") or "").strip()
        if not (code and name):
            continue
        num_client = str(7200 + index)
        availability_slots: dict[str, Any] = {
            "reference": code,
            "lens_code": code,
            "product": name,
        }
        create_slots: dict[str, Any] = {
            "num_client": num_client,
            "product": name,
        }
        if material:
            availability_slots["material"] = material
            create_slots["material"] = material
        if diameter:
            availability_slots["diameter"] = diameter
            create_slots["diameter"] = diameter

        prompts = [
            (
                f"tnajem tconfirmili code verre {code} ?",
                "reference_confirmation",
                {"reference": code, "lens_code": code},
            ),
            (
                f"aslema, dispo {code} {name}{f' diametre {diameter}' if diameter else ''} ?",
                "availability_inquiry",
                availability_slots,
            ),
            (
                f"bonjour {num_client} nheb na3di commande {name}{f' {material}' if material else ''}{f' diametre {diameter}' if diameter else ''}",
                "create_order",
                create_slots,
            ),
        ]
        for prompt_text, intent, slots in prompts:
            add_example([{"role": "user", "content": prompt_text}], intent, slots, "lens_rag")
    print(f"[intent-slot] after lens rag rows={len(rows)}", flush=True)

    random.seed(SEED)
    random.shuffle(rows)
    if len(rows) <= 1:
        train_rows = rows
        val_rows = []
    else:
        split_at = max(1, int(len(rows) * 0.95))
        split_at = min(split_at, len(rows) - 1)
        train_rows = rows[:split_at]
        val_rows = rows[split_at:]

    _write_jsonl(INTENT_SLOT_TRAIN, train_rows)
    _write_jsonl(INTENT_SLOT_VAL, val_rows)
    print(
        f"[intent-slot] wrote train={len(train_rows)} val={len(val_rows)} -> {INTENT_SLOT_TRAIN.as_posix()}",
        flush=True,
    )

    report = {
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "total_rows": len(rows),
        "source_counts": dict(sorted(source_counts.items())),
        "intent_counts": dict(sorted(intent_counts.items())),
        "train_path": _rel(INTENT_SLOT_TRAIN),
        "val_path": _rel(INTENT_SLOT_VAL),
    }
    INTENT_SLOT_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _render_phase1_yaml(
    base_model: str,
    latest_checkpoint: str | None,
    phase1_lora_settings: dict[str, float | int],
) -> str:
    resume_line = f"resume_from_checkpoint: {latest_checkpoint}\n" if latest_checkpoint else ""
    return (
        PHASE1_TEMPLATE_NOTE
        + "base_model: "
        + base_model
        + "\n"
        + "trust_remote_code: true\n"
        + "strict: false\n"
        + "seed: 42\n"
        + "output_dir: outputs/1_ssf\n"
        + "dataset_prepared_path: artifacts/cache/axolotl/1_ssf_prepared\n"
        + "adapter: qlora\n"
        + "load_in_4bit: true\n"
        + f"lora_r: {phase1_lora_settings['lora_r']}\n"
        + f"lora_alpha: {phase1_lora_settings['lora_alpha']}\n"
        + f"lora_dropout: {phase1_lora_settings['lora_dropout']}\n"
        + "lora_target_linear: true\n"
        + "sequence_len: 1024\n"
        + "sample_packing: true\n"
        + "pad_to_sequence_len: true\n"
        + "micro_batch_size: 1\n"
        + "gradient_accumulation_steps: 16\n"
        + "eval_batch_size: 1\n"
        + "num_epochs: 3\n"
        + "learning_rate: 5e-5\n"
        + "optimizer: paged_adamw_32bit\n"
        + "lr_scheduler: cosine\n"
        + "warmup_ratio: 0.03\n"
        + "max_grad_norm: 0.3\n"
        + "weight_decay: 0.0\n"
        + "gradient_checkpointing: true\n"
        + "fp16: true\n"
        + "bf16: false\n"
        + "tf32: false\n"
        + "sdp_attention: true\n"
        + "use_tensorboard: true\n"
        + "logging_steps: 10\n"
        + "eval_strategy: \"no\"\n"
        + "save_strategy: epoch\n"
        + "save_total_limit: 2\n"
        + resume_line
        + "datasets:\n"
        + "  - path: data/processed/self_sup_train.jsonl\n"
        + "    split: train\n"
        + "    type: completion\n"
        + "test_datasets:\n"
        + "  - path: data/processed/self_sup_val.jsonl\n"
        + "    split: train\n"
        + "    type: completion\n"
    )


def _render_phase2_yaml(base_model: str) -> str:
    return (
        PHASE1_TEMPLATE_NOTE
        + "base_model: "
        + base_model
        + "\n"
        + "trust_remote_code: true\n"
        + "strict: false\n"
        + "seed: 42\n"
        + "output_dir: outputs/2_sft\n"
        + "dataset_prepared_path: artifacts/cache/axolotl/2_sft_prepared\n"
        + "adapter: qlora\n"
        + "load_in_4bit: true\n"
        + "lora_r: 32\n"
        + "lora_alpha: 32\n"
        + "lora_dropout: 0.05\n"
        + "lora_target_linear: true\n"
        + "sequence_len: 1536\n"
        + "sample_packing: true\n"
        + "pad_to_sequence_len: true\n"
        + "micro_batch_size: 1\n"
        + "gradient_accumulation_steps: 16\n"
        + "eval_batch_size: 1\n"
        + "num_epochs: 5\n"
        + "learning_rate: 1.2e-4\n"
        + "optimizer: paged_adamw_32bit\n"
        + "lr_scheduler: cosine\n"
        + "warmup_ratio: 0.03\n"
        + "max_grad_norm: 0.3\n"
        + "weight_decay: 0.0\n"
        + "gradient_checkpointing: true\n"
        + "fp16: true\n"
        + "bf16: false\n"
        + "tf32: false\n"
        + "sdp_attention: true\n"
        + "use_tensorboard: true\n"
        + "logging_steps: 10\n"
        + "eval_strategy: epoch\n"
        + "save_strategy: epoch\n"
        + "save_total_limit: 2\n"
        + "chat_template: tokenizer_default\n"
        + "datasets:\n"
        + "  - path: data/processed/sft_train.jsonl\n"
        + "    split: train\n"
        + "    field_messages: messages\n"
        + "    type: chat_template\n"
        + "test_datasets:\n"
        + "  - path: data/processed/sft_val.jsonl\n"
        + "    split: train\n"
        + "    field_messages: messages\n"
        + "    type: chat_template\n"
    )


def _render_phase3_yaml(base_model: str) -> str:
    return (
        PHASE1_TEMPLATE_NOTE
        + "base_model: "
        + base_model
        + "\n"
        + "trust_remote_code: true\n"
        + "strict: false\n"
        + "seed: 42\n"
        + "output_dir: outputs/3_intent_slot\n"
        + "dataset_prepared_path: artifacts/cache/axolotl/3_intent_slot_prepared\n"
        + "adapter: qlora\n"
        + "load_in_4bit: true\n"
        + "lora_r: 32\n"
        + "lora_alpha: 32\n"
        + "lora_dropout: 0.05\n"
        + "lora_target_linear: true\n"
        + "sequence_len: 1024\n"
        + "sample_packing: true\n"
        + "pad_to_sequence_len: true\n"
        + "micro_batch_size: 1\n"
        + "gradient_accumulation_steps: 16\n"
        + "eval_batch_size: 1\n"
        + "num_epochs: 3\n"
        + "learning_rate: 8e-5\n"
        + "optimizer: paged_adamw_32bit\n"
        + "lr_scheduler: cosine\n"
        + "warmup_ratio: 0.03\n"
        + "max_grad_norm: 0.3\n"
        + "weight_decay: 0.0\n"
        + "gradient_checkpointing: true\n"
        + "fp16: true\n"
        + "bf16: false\n"
        + "tf32: false\n"
        + "sdp_attention: true\n"
        + "use_tensorboard: true\n"
        + "logging_steps: 10\n"
        + "eval_strategy: epoch\n"
        + "save_strategy: epoch\n"
        + "save_total_limit: 2\n"
        + "chat_template: tokenizer_default\n"
        + "datasets:\n"
        + "  - path: data/processed/intent_slot_train.jsonl\n"
        + "    split: train\n"
        + "    field_messages: messages\n"
        + "    type: chat_template\n"
        + "test_datasets:\n"
        + "  - path: data/processed/intent_slot_val.jsonl\n"
        + "    split: train\n"
        + "    field_messages: messages\n"
        + "    type: chat_template\n"
    )


def _render_phase4_yaml(base_model: str) -> str:
    return (
        PHASE1_TEMPLATE_NOTE
        + "base_model: "
        + base_model
        + "\n"
        + "trust_remote_code: true\n"
        + "strict: false\n"
        + "seed: 42\n"
        + "output_dir: outputs/4_orpo\n"
        + "dataset_prepared_path: artifacts/cache/axolotl/4_orpo_prepared\n"
        + "adapter: qlora\n"
        + "load_in_4bit: true\n"
        + "lora_r: 32\n"
        + "lora_alpha: 32\n"
        + "lora_dropout: 0.05\n"
        + "lora_target_linear: true\n"
        + "sequence_len: 1024\n"
        + "micro_batch_size: 1\n"
        + "gradient_accumulation_steps: 16\n"
        + "eval_batch_size: 1\n"
        + "num_epochs: 2\n"
        + "learning_rate: 3e-6\n"
        + "optimizer: paged_adamw_32bit\n"
        + "lr_scheduler: cosine\n"
        + "warmup_steps: 20\n"
        + "max_grad_norm: 0.3\n"
        + "gradient_checkpointing: true\n"
        + "fp16: true\n"
        + "bf16: false\n"
        + "tf32: false\n"
        + "sdp_attention: true\n"
        + "use_tensorboard: true\n"
        + "logging_steps: 10\n"
        + "eval_strategy: epoch\n"
        + "save_strategy: epoch\n"
        + "save_total_limit: 2\n"
        + "chat_template: tokenizer_default\n"
        + "rl: orpo\n"
        + "orpo_alpha: 0.1\n"
        + "remove_unused_columns: false\n"
        + "datasets:\n"
        + "  - path: data/processed/orpo_train.jsonl\n"
        + "    ds_type: json\n"
        + "    data_files:\n"
        + "      - data/processed/orpo_train.jsonl\n"
        + "    split: train\n"
        + "    type: chat_template.argilla\n"
    )


def _write_configs(base_model: str, latest_checkpoint: str | None, mode: str) -> dict[str, Any]:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    phase2_base = base_model
    phase3_base = base_model
    phase4_base = base_model

    if mode in {"after-ssf", "after-sft", "after-intent-slot"}:
        phase2_base = SSF_MERGED_DIR.as_posix()
        phase3_base = SSF_MERGED_DIR.as_posix()
        phase4_base = SSF_MERGED_DIR.as_posix()
    if mode in {"after-sft", "after-intent-slot"}:
        phase3_base = SFT_MERGED_DIR.as_posix()
        phase4_base = SFT_MERGED_DIR.as_posix()
    if mode == "after-intent-slot":
        phase4_base = INTENT_SLOT_MERGED_DIR.as_posix()

    phase1_path = CONFIGS_DIR / "1_ssf.yaml"
    phase2_path = CONFIGS_DIR / "2_sft.yaml"
    phase3_path = CONFIGS_DIR / "3_intent_slot.yaml"
    phase4_path = CONFIGS_DIR / "4_orpo.yaml"
    legacy_phase3_orpo = CONFIGS_DIR / "3_orpo.yaml"
    if legacy_phase3_orpo.exists():
        legacy_phase3_orpo.unlink()

    phase1_lora_settings = _detect_phase1_lora_settings(latest_checkpoint)

    phase1_path.write_text(
        _render_phase1_yaml(base_model, latest_checkpoint, phase1_lora_settings),
        encoding="utf-8",
    )
    phase2_path.write_text(_render_phase2_yaml(phase2_base), encoding="utf-8")
    phase3_path.write_text(_render_phase3_yaml(phase3_base), encoding="utf-8")
    phase4_path.write_text(_render_phase4_yaml(phase4_base), encoding="utf-8")

    return {
        "phase1_config": _rel(phase1_path),
        "phase2_config": _rel(phase2_path),
        "phase3_config": _rel(phase3_path),
        "phase4_config": _rel(phase4_path),
        "phase2_base_model": phase2_base,
        "phase3_base_model": phase3_base,
        "phase4_base_model": phase4_base,
        "phase1_lora_settings": phase1_lora_settings,
    }


def _write_manifest(payload: dict[str, Any]) -> None:
    PIPELINE_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PIPELINE_MANIFEST_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def validate_configs() -> dict[str, Any]:
    try:
        from axolotl.cli.config import load_cfg  # type: ignore[attr-defined]  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:
        from axolotl.cli import load_cfg  # pylint: disable=import-outside-toplevel

    statuses: list[dict[str, str]] = []
    for config_path in (
        CONFIGS_DIR / "1_ssf.yaml",
        CONFIGS_DIR / "2_sft.yaml",
        CONFIGS_DIR / "3_intent_slot.yaml",
        CONFIGS_DIR / "4_orpo.yaml",
    ):
        cfg = load_cfg(config_path.as_posix())
        statuses.append(
            {
                "config": _rel(config_path) or config_path.as_posix(),
                "status": "ok",
                "base_model": str(cfg.base_model),
                "output_dir": str(cfg.output_dir),
            }
        )
    return {"configs": statuses}


def bootstrap() -> dict[str, Any]:
    latest_checkpoint = _detect_latest_checkpoint()
    base_model = _detect_base_model(latest_checkpoint)
    tdc_download = _download_tdc()
    tdc_sft = _augment_custom_sft_from_tdc()
    dpo_bootstrap = _ensure_fallback_dpo()
    prepared = _run_existing_prepare()
    intent_slot = _build_intent_slot_dataset()
    orpo = _build_orpo_dataset()
    configs = _write_configs(base_model, latest_checkpoint, mode="bootstrap")
    validation = validate_configs()
    payload = {
        "mode": "bootstrap",
        "latest_checkpoint": latest_checkpoint,
        "base_model": base_model,
        "tdc_download": tdc_download,
        "tdc_sft": tdc_sft,
        "dpo_bootstrap": dpo_bootstrap,
        "prepared": prepared,
        "intent_slot": intent_slot,
        "orpo": orpo,
        "configs": configs,
        "validation": validation,
    }
    _write_manifest(payload)
    return payload


def refresh_runtime(mode: str) -> dict[str, Any]:
    if mode not in {"after-ssf", "after-sft", "after-intent-slot"}:
        raise ValueError(f"Unsupported refresh mode: {mode}")
    latest_checkpoint = _detect_latest_checkpoint()
    base_model = _detect_base_model(latest_checkpoint)
    configs = _write_configs(base_model, latest_checkpoint, mode=mode)
    validation = validate_configs()
    payload = {
        "mode": mode,
        "latest_checkpoint": latest_checkpoint,
        "base_model": base_model,
        "configs": configs,
        "validation": validation,
    }
    _write_manifest(payload)
    return payload


def render_bootstrap() -> dict[str, Any]:
    latest_checkpoint = _detect_latest_checkpoint()
    base_model = _detect_base_model(latest_checkpoint)
    intent_slot = _build_intent_slot_dataset()
    orpo = _build_orpo_dataset()
    configs = _write_configs(base_model, latest_checkpoint, mode="bootstrap")
    validation = validate_configs()
    payload = {
        "mode": "render-bootstrap",
        "latest_checkpoint": latest_checkpoint,
        "base_model": base_model,
        "intent_slot": intent_slot,
        "orpo": orpo,
        "configs": configs,
        "validation": validation,
    }
    _write_manifest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare data and Axolotl configs for the Tunisian pipeline.")
    parser.add_argument(
        "--mode",
        choices=["bootstrap", "after-ssf", "after-sft", "after-intent-slot", "render-bootstrap", "validate"],
        default="bootstrap",
    )
    args = parser.parse_args()

    if args.mode == "bootstrap":
        payload = bootstrap()
    elif args.mode == "validate":
        payload = validate_configs()
    elif args.mode == "render-bootstrap":
        payload = render_bootstrap()
    else:
        payload = refresh_runtime(args.mode)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
    # Some library background threads can occasionally block interpreter shutdown.
    # Allow forcing a hard exit for CLI reliability in pipeline orchestration.
    force_exit = os.environ.get("TOUNSI_PREPARE_FORCE_OS_EXIT", "1").strip().lower()
    if force_exit not in {"0", "false", "no"}:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
