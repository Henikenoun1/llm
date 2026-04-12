"""
Training pipeline:
  1. self-supervised language adaptation
  2. supervised dialogue fine-tuning
  3. DPO alignment
"""
from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

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
from .domain_utils import normalize_messages


def load_base_model() -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        raise RuntimeError(
            "Missing training dependencies. Install required packages from requirements.txt before running training stages."
        ) from exc

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
    try:
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        raise RuntimeError(
            "Missing PEFT dependencies. Install required packages from requirements.txt before running training stages."
        ) from exc

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


def _chat_messages_to_text(messages: list[dict[str, Any]], tokenizer: Any) -> str:
    existing_system = None
    cleaned_messages: list[dict[str, str]] = []
    for message in normalize_messages(messages):
        role = message["role"]
        content = message["content"]
        if role == "system":
            existing_system = content
            continue
        cleaned_messages.append({"role": role, "content": content})

    system_prompt = existing_system if existing_system in _KNOWN_SYSTEM_PROMPTS else SYSTEM_PROMPT
    full_messages = [{"role": "system", "content": system_prompt}] + cleaned_messages
    return tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def _load_text_dataset(path: Path, kind: str, tokenizer: Any | None = None) -> Any:
    try:
        import datasets as hfds
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        raise RuntimeError(
            "Missing datasets dependency. Install required packages from requirements.txt before running training stages."
        ) from exc

    rows: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if kind == "chat":
                messages = record.get("messages", [])
                if messages and tokenizer is not None:
                    rows.append({"text": _chat_messages_to_text(messages, tokenizer=tokenizer)})
            else:
                text = record.get("text")
                if text:
                    rows.append({"text": text})
    return hfds.Dataset.from_list(rows)


def _write_log_history(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in history:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _safe_perplexity(loss: Any) -> float | None:
    try:
        value = float(loss)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value > 20:
        return None
    return round(math.exp(value), 4)


def _summarize_training_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    train_rows = [row for row in history if "loss" in row and "eval_loss" not in row]
    eval_rows = [row for row in history if "eval_loss" in row]

    final_train_loss = float(train_rows[-1]["loss"]) if train_rows else None
    final_eval_loss = float(eval_rows[-1]["eval_loss"]) if eval_rows else None
    best_eval_loss = min((float(row["eval_loss"]) for row in eval_rows), default=None)

    learning_rates = [
        float(row["learning_rate"])
        for row in history
        if row.get("learning_rate") is not None
    ]

    return {
        "logged_train_points": len(train_rows),
        "logged_eval_points": len(eval_rows),
        "final_train_loss": final_train_loss,
        "final_eval_loss": final_eval_loss,
        "best_eval_loss": best_eval_loss,
        "best_eval_perplexity": _safe_perplexity(best_eval_loss),
        "final_eval_perplexity": _safe_perplexity(final_eval_loss),
        "min_learning_rate": min(learning_rates) if learning_rates else None,
        "max_learning_rate": max(learning_rates) if learning_rates else None,
    }


def _default_stage_max_steps(stage_name: str) -> int | None:
    defaults = {
        "self_sup": CFG.self_sup_max_steps,
        "sft": CFG.sft_max_steps,
        "dpo": CFG.dpo_max_steps,
    }
    return defaults.get(stage_name)


def _default_stage_max_seq_len(stage_name: str) -> int:
    defaults = {
        "self_sup": CFG.self_sup_max_seq_len,
        "sft": CFG.sft_max_seq_len,
        "dpo": CFG.dpo_max_seq_len,
    }
    return defaults.get(stage_name, CFG.max_seq_len)


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
    max_steps: int | None = None,
    max_seq_length: int | None = None,
) -> dict[str, Any]:
    try:
        import torch
        from transformers import EarlyStoppingCallback
        from trl import SFTConfig, SFTTrainer
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        raise RuntimeError(
            "Missing training dependencies. Install required packages from requirements.txt before running training stages."
        ) from exc

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    model, tokenizer = load_base_model()
    model = _attach_trainable_adapter(model, adapter_dir=adapter_init_dir)

    train_ds = _load_text_dataset(train_path, kind=data_kind, tokenizer=tokenizer)
    val_ds = (
        _load_text_dataset(val_path, kind=data_kind, tokenizer=tokenizer)
        if val_path and val_path.exists()
        else None
    )
    effective_seq_len = max_seq_length or _default_stage_max_seq_len(stage_name)
    effective_max_steps = max_steps if max_steps is not None else _default_stage_max_steps(stage_name)

    args = SFTConfig(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=CFG.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=CFG.grad_accum_steps,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        weight_decay=CFG.weight_decay,
        logging_strategy="steps",
        warmup_ratio=CFG.warmup_ratio,
        logging_steps=5,
        logging_first_step=True,
        save_strategy="epoch",
        eval_strategy="epoch" if val_ds else "no",
        fp16=CFG.fp16,
        bf16=CFG.bf16,
        report_to=[],
        seed=SEED,
        max_seq_length=effective_seq_len,
        dataset_text_field="text",
        packing=False,
        max_steps=effective_max_steps if effective_max_steps is not None else -1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        load_best_model_at_end=bool(val_ds),
        metric_for_best_model="eval_loss" if val_ds else None,
        greater_is_better=False if val_ds else None,
        save_total_limit=CFG.save_total_limit,
        optim=CFG.optim,
        lr_scheduler_type=CFG.lr_scheduler_type,
        max_grad_norm=CFG.max_grad_norm,
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=CFG.early_stopping_patience)] if val_ds else []

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=args,
        callbacks=callbacks,
    )

    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start
    log_history = list(getattr(trainer.state, "log_history", []) or [])
    diagnostics = _summarize_training_history(log_history)

    for row in log_history:
        if row.get("eval_loss") is None:
            continue
        logger.info(
            "%s eval epoch=%s step=%s eval_loss=%.4f eval_ppl=%s",
            stage_name,
            row.get("epoch"),
            row.get("step"),
            float(row["eval_loss"]),
            _safe_perplexity(row.get("eval_loss")),
        )

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
            "max_steps": effective_max_steps,
            "max_seq_length": effective_seq_len,
            "continued_from_adapter": bool(adapter_init_dir and (adapter_init_dir / "adapter_config.json").exists()),
            "adapter_init_dir": str(adapter_init_dir) if adapter_init_dir else None,
            "output_dir": str(output_dir),
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "diagnostics": diagnostics,
        }
    )
    history_path = REPORTS_DIR / f"{stage_name}_train_log.jsonl"
    _write_log_history(history_path, log_history)
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
    max_steps: int | None = None,
    max_seq_length: int | None = None,
    continue_from_adapter: bool | None = None,
) -> dict[str, Any]:
    if continue_from_adapter is None:
        continue_from_adapter = CFG.self_sup_continue_from_adapter
    initial_adapter = CFG.resolve_adapter_dir("self_sup") if continue_from_adapter else None

    return _run_sft_stage(
        stage_name="self_sup",
        train_path=train_path or PROCESSED_DATA_DIR / "self_sup_train.jsonl",
        val_path=val_path or PROCESSED_DATA_DIR / "self_sup_val.jsonl",
        output_dir=output_dir or CFG.adapter_dir_for_stage("self_sup"),
        checkpoint_dir=CHECKPOINTS_DIR / "self_sup",
        learning_rate=CFG.self_sup_learning_rate,
        epochs=epochs or CFG.epochs_self_sup,
        adapter_init_dir=initial_adapter,
        data_kind="text",
        max_steps=max_steps,
        max_seq_length=max_seq_length or CFG.self_sup_max_seq_len,
    )


def train_sft(
    train_path: Path | None = None,
    val_path: Path | None = None,
    output_dir: Path | None = None,
    epochs: int | None = None,
    adapter_init_dir: Path | None = None,
    max_steps: int | None = None,
    max_seq_length: int | None = None,
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
        max_steps=max_steps,
        max_seq_length=max_seq_length or CFG.sft_max_seq_len,
    )


def train_dpo(
    train_path: Path | None = None,
    val_path: Path | None = None,
    sft_adapter_dir: Path | None = None,
    output_dir: Path | None = None,
    epochs: int | None = None,
    beta: float = 0.1,
    max_steps: int | None = None,
) -> dict[str, Any]:
    try:
        import datasets as hfds
        import torch
        from transformers import EarlyStoppingCallback
        from trl import DPOConfig, DPOTrainer
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        raise RuntimeError(
            "Missing training dependencies. Install required packages from requirements.txt before running training stages."
        ) from exc

    train_path = train_path or PROCESSED_DATA_DIR / "dpo_train.jsonl"
    val_path = val_path or PROCESSED_DATA_DIR / "dpo_val.jsonl"
    sft_adapter_dir = sft_adapter_dir or CFG.resolve_adapter_dir("sft")
    output_dir = output_dir or CFG.adapter_dir_for_stage("dpo")

    if not train_path.exists():
        raise FileNotFoundError(f"DPO training data not found: {train_path}")

    model, tokenizer = load_base_model()
    model = _attach_trainable_adapter(model, adapter_dir=sft_adapter_dir)

    def _load_pairs(path: Path) -> Any:
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
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=CFG.grad_accum_steps,
        learning_rate=CFG.dpo_learning_rate,
        num_train_epochs=epochs or CFG.epochs_dpo,
        logging_strategy="steps",
        logging_steps=5,
        logging_first_step=True,
        save_strategy="epoch",
        eval_strategy="epoch" if val_ds else "no",
        fp16=CFG.fp16,
        bf16=CFG.bf16,
        beta=beta,
        report_to=[],
        seed=SEED,
        max_length=CFG.dpo_max_seq_len,
        max_prompt_length=min(768, CFG.dpo_max_seq_len // 2),
        gradient_checkpointing=True,
        load_best_model_at_end=bool(val_ds),
        metric_for_best_model="eval_loss" if val_ds else None,
        greater_is_better=False if val_ds else None,
        save_total_limit=CFG.save_total_limit,
        optim=CFG.optim,
        lr_scheduler_type=CFG.lr_scheduler_type,
        max_grad_norm=CFG.max_grad_norm,
        max_steps=max_steps if max_steps is not None else CFG.dpo_max_steps,
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=CFG.early_stopping_patience)] if val_ds else []
    trainer = DPOTrainer(
        model=model,
        args=dpo_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start
    log_history = list(getattr(trainer.state, "log_history", []) or [])
    diagnostics = _summarize_training_history(log_history)

    for row in log_history:
        if row.get("eval_loss") is None:
            continue
        logger.info(
            "dpo eval epoch=%s step=%s eval_loss=%.4f eval_ppl=%s",
            row.get("epoch"),
            row.get("step"),
            float(row["eval_loss"]),
            _safe_perplexity(row.get("eval_loss")),
        )

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
            "max_steps": max_steps if max_steps is not None else CFG.dpo_max_steps,
            "training_time_min": round(elapsed / 60, 1),
            "train_examples": len(train_ds),
            "eval_examples": len(val_ds) if val_ds else 0,
            "adapter_init_dir": str(sft_adapter_dir) if sft_adapter_dir else None,
            "output_dir": str(output_dir),
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "diagnostics": diagnostics,
        }
    )
    history_path = REPORTS_DIR / "dpo_train_log.jsonl"
    _write_log_history(history_path, log_history)
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

