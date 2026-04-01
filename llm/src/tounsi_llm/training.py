"""
Training pipeline:
  1. self-supervised language adaptation
  2. supervised dialogue fine-tuning
  3. DPO alignment
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import datasets as hfds
import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from .config import (
    ADAPTERS_DIR,
    CFG,
    CHECKPOINTS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    SEED,
    logger,
)
from .data_prep import SYSTEM_PROMPT, _KNOWN_SYSTEM_PROMPTS


def load_base_model() -> tuple[Any, Any]:
    logger.info("Loading base model: %s", CFG.base_model)

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
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def _attach_trainable_adapter(model, adapter_dir: Path | None = None):
    if CFG.use_4bit:
        model = prepare_model_for_kbit_training(model)

    if adapter_dir and (adapter_dir / "adapter_config.json").exists():
        logger.info("Continuing training from adapter %s", adapter_dir)
        return PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)

    lora_cfg = LoraConfig(
        r=CFG.lora_rank,
        lora_alpha=CFG.lora_alpha,
        lora_dropout=CFG.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=CFG.lora_target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def _chat_messages_to_text(messages: list[dict[str, Any]]) -> str:
    existing_system = None
    cleaned_messages: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", message.get("text", ""))
        if not content:
            continue
        if role == "system":
            existing_system = content
            continue
        cleaned_messages.append({"role": role, "content": content})

    system_prompt = existing_system if existing_system in _KNOWN_SYSTEM_PROMPTS else SYSTEM_PROMPT
    full_messages = [{"role": "system", "content": system_prompt}] + cleaned_messages

    parts = []
    for message in full_messages:
        parts.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _load_text_dataset(path: Path, kind: str) -> hfds.Dataset:
    rows: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if kind == "chat":
                messages = record.get("messages", [])
                if messages:
                    rows.append({"text": _chat_messages_to_text(messages)})
            else:
                text = record.get("text")
                if text:
                    rows.append({"text": text})
    return hfds.Dataset.from_list(rows)


def _run_sft_stage(
    *,
    stage_name: str,
    train_path: Path,
    val_path: Path | None,
    output_dir: Path,
    checkpoint_dir: Path,
    learning_rate: float,
    epochs: int,
    adapter_init_dir: Path | None,
    data_kind: str,
) -> dict[str, Any]:
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    model, tokenizer = load_base_model()
    model = _attach_trainable_adapter(model, adapter_dir=adapter_init_dir)

    train_ds = _load_text_dataset(train_path, kind=data_kind)
    val_ds = _load_text_dataset(val_path, kind=data_kind) if val_path and val_path.exists() else None

    args = SFTConfig(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=CFG.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=CFG.grad_accum_steps,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        weight_decay=CFG.weight_decay,
        warmup_ratio=CFG.warmup_ratio,
        logging_steps=10,
        save_steps=200,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=200 if val_ds else None,
        fp16=CFG.fp16,
        bf16=CFG.bf16,
        report_to=[],
        seed=SEED,
        max_seq_length=CFG.max_seq_len,
        dataset_text_field="text",
        packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=args,
    )

    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = dict(getattr(result, "metrics", {}) or {})
    metrics.update(
        {
            "stage": stage_name,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "training_time_min": round(elapsed / 60, 1),
            "train_examples": len(train_ds),
            "eval_examples": len(val_ds) if val_ds else 0,
            "adapter_init_dir": str(adapter_init_dir) if adapter_init_dir else None,
            "output_dir": str(output_dir),
        }
    )
    metrics_path = REPORTS_DIR / f"{stage_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("%s complete -> %s", stage_name, output_dir)
    return metrics


def train_self_supervised(
    train_path: Path | None = None,
    val_path: Path | None = None,
    output_dir: Path | None = None,
    epochs: int | None = None,
) -> dict[str, Any]:
    return _run_sft_stage(
        stage_name="self_sup",
        train_path=train_path or PROCESSED_DATA_DIR / "self_sup_train.jsonl",
        val_path=val_path or PROCESSED_DATA_DIR / "self_sup_val.jsonl",
        output_dir=output_dir or CFG.adapter_dir_for_stage("self_sup"),
        checkpoint_dir=CHECKPOINTS_DIR / "self_sup",
        learning_rate=CFG.self_sup_learning_rate,
        epochs=epochs or CFG.epochs_self_sup,
        adapter_init_dir=None,
        data_kind="text",
    )


def train_sft(
    train_path: Path | None = None,
    val_path: Path | None = None,
    output_dir: Path | None = None,
    epochs: int | None = None,
    adapter_init_dir: Path | None = None,
) -> dict[str, Any]:
    initial_adapter = adapter_init_dir
    if initial_adapter is None:
        initial_adapter = CFG.resolve_adapter_dir("self_sup")

    return _run_sft_stage(
        stage_name="sft",
        train_path=train_path or PROCESSED_DATA_DIR / "sft_train.jsonl",
        val_path=val_path or PROCESSED_DATA_DIR / "sft_val.jsonl",
        output_dir=output_dir or CFG.adapter_dir_for_stage("sft"),
        checkpoint_dir=CHECKPOINTS_DIR / "sft",
        learning_rate=CFG.sft_learning_rate,
        epochs=epochs or CFG.epochs_sft,
        adapter_init_dir=initial_adapter,
        data_kind="chat",
    )


def train_dpo(
    train_path: Path | None = None,
    val_path: Path | None = None,
    sft_adapter_dir: Path | None = None,
    output_dir: Path | None = None,
    epochs: int | None = None,
    beta: float = 0.1,
) -> dict[str, Any]:
    from trl import DPOConfig, DPOTrainer

    train_path = train_path or PROCESSED_DATA_DIR / "dpo_train.jsonl"
    val_path = val_path or PROCESSED_DATA_DIR / "dpo_val.jsonl"
    sft_adapter_dir = sft_adapter_dir or CFG.resolve_adapter_dir("sft")
    output_dir = output_dir or CFG.adapter_dir_for_stage("dpo")

    if not train_path.exists():
        raise FileNotFoundError(f"DPO training data not found: {train_path}")

    model, tokenizer = load_base_model()
    model = _attach_trainable_adapter(model, adapter_dir=sft_adapter_dir)

    def _load_pairs(path: Path) -> hfds.Dataset:
        rows: list[dict[str, str]] = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("prompt") and row.get("chosen") and row.get("rejected"):
                    rows.append(
                        {
                            "prompt": row["prompt"],
                            "chosen": row["chosen"],
                            "rejected": row["rejected"],
                        }
                    )
        return hfds.Dataset.from_list(rows)

    train_ds = _load_pairs(train_path)
    val_ds = _load_pairs(val_path) if val_path.exists() else None

    dpo_cfg = DPOConfig(
        output_dir=str(CHECKPOINTS_DIR / "dpo"),
        per_device_train_batch_size=CFG.batch_size,
        gradient_accumulation_steps=CFG.grad_accum_steps,
        learning_rate=CFG.dpo_learning_rate,
        num_train_epochs=epochs or CFG.epochs_dpo,
        logging_steps=10,
        save_steps=100,
        fp16=CFG.fp16,
        bf16=CFG.bf16,
        beta=beta,
        report_to=[],
        seed=SEED,
        max_length=CFG.max_seq_len,
        max_prompt_length=min(1024, CFG.max_seq_len // 2),
        gradient_checkpointing=True,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = dict(getattr(result, "metrics", {}) or {})
    metrics.update(
        {
            "stage": "dpo",
            "epochs": epochs or CFG.epochs_dpo,
            "learning_rate": CFG.dpo_learning_rate,
            "beta": beta,
            "training_time_min": round(elapsed / 60, 1),
            "train_examples": len(train_ds),
            "eval_examples": len(val_ds) if val_ds else 0,
            "adapter_init_dir": str(sft_adapter_dir) if sft_adapter_dir else None,
            "output_dir": str(output_dir),
        }
    )
    metrics_path = REPORTS_DIR / "dpo_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("dpo complete -> %s", output_dir)
    return metrics


def promote_adapter(source_variant: str = "dpo", destination_stage: str = "production") -> dict[str, Any]:
    source_dir = CFG.resolve_adapter_dir(source_variant)
    if source_dir is None:
        raise FileNotFoundError(f"No adapter found for variant '{source_variant}'")

    destination = CFG.adapter_dir_for_stage(destination_stage)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source_dir, destination)

    result = {"source": str(source_dir), "destination": str(destination)}
    (ADAPTERS_DIR / "active_production.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    logger.info("Promoted adapter %s -> %s", source_dir, destination)
    return result

