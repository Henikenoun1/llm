#!/usr/bin/env python3
"""
Pipeline entry point.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tounsi_llm.config import (
    ADAPTERS_DIR,
    CACHE_DIR,
    CFG,
    CHECKPOINTS_DIR,
    DOMAIN_CFG,
    HISTORY_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    RUNTIME,
    logger,
)


def _clear_directory(path: Path, *, preserve_names: set[str] | None = None) -> None:
    preserve_names = preserve_names or set()
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return

    for child in path.iterdir():
        if child.name in preserve_names:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _reset_generated_processed_files() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for pattern in ["self_sup_*.jsonl", "sft_*.jsonl", "dpo_*.jsonl"]:
        for path in PROCESSED_DATA_DIR.glob(pattern):
            path.unlink()
            logger.info("Removed %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Configurable call-center LLM pipeline")
    parser.add_argument(
        "--stage",
        choices=[
            "reset",
            "download",
            "audit",
            "prepare",
            "validate",
            "self-sup",
            "sft",
            "dpo",
            "promote",
            "eval",
            "serve",
            "all",
            "full",
        ],
        default="all",
    )
    parser.add_argument("--max-sft-samples", type=int, default=None)
    parser.add_argument("--max-self-sup-texts", type=int, default=None)
    parser.add_argument("--clean-before-run", action="store_true")
    parser.add_argument("--reset-processed", action="store_true")
    parser.add_argument("--serve-after-promote", action="store_true")
    parser.add_argument("--fail-on-no-go", action="store_true")
    parser.add_argument("--self-sup-epochs", type=int, default=None)
    parser.add_argument("--self-sup-max-steps", type=int, default=None)
    parser.add_argument("--self-sup-max-seq-len", type=int, default=None)
    parser.add_argument("--self-sup-fresh-adapter", action="store_true")
    parser.add_argument("--sft-epochs", type=int, default=None)
    parser.add_argument("--sft-max-steps", type=int, default=None)
    parser.add_argument("--sft-max-seq-len", type=int, default=None)
    parser.add_argument("--dpo-epochs", type=int, default=None)
    parser.add_argument("--dpo-max-steps", type=int, default=None)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--prod-source-variant", default="sft")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--eval-model-variant", default="prod")
    parser.add_argument("--eval-runtime-mode", default="autonomous")
    parser.add_argument("--eval-max-cases", type=int, default=None)
    args = parser.parse_args()

    logger.info("Domain config: %s", DOMAIN_CFG.get("domain_name"))
    logger.info("Runtime: %s", RUNTIME)
    logger.info("Project config: %s", CFG.to_dict())

    stages = {
        "reset": [_reset],
        "download": [_download],
        "audit": [_audit],
        "prepare": [_prepare],
        "validate": [_validate],
        "self-sup": [_self_sup],
        "sft": [_sft],
        "dpo": [_dpo],
        "promote": [_promote],
        "eval": [_eval],
        "serve": [_serve],
        "all": [_download, _audit, _prepare, _self_sup, _sft, _dpo, _promote, _eval, _validate],
        "full": [_download, _audit, _prepare, _self_sup, _sft, _dpo, _promote, _eval, _validate],
    }

    if args.clean_before_run and args.stage != "reset":
        _reset(args)

    for step in stages[args.stage]:
        step(args)

    if args.stage == "full" and args.serve_after_promote:
        _serve(args)


def _reset(args) -> None:
    for target in [ADAPTERS_DIR, CHECKPOINTS_DIR, CACHE_DIR]:
        _clear_directory(target)
        logger.info("Cleared %s", target)

    _clear_directory(HISTORY_DIR)
    logger.info("Cleared %s", HISTORY_DIR)

    _clear_directory(REPORTS_DIR, preserve_names={"run.log"})
    run_log = REPORTS_DIR / "run.log"
    run_log.parent.mkdir(parents=True, exist_ok=True)
    run_log.write_text("", encoding="utf-8")
    logger.info("Cleared generated reports in %s", REPORTS_DIR)

    if args.reset_processed or args.stage == "reset":
        _reset_generated_processed_files()

    active_marker = ADAPTERS_DIR / "active_production.json"
    if active_marker.exists():
        active_marker.unlink()
        logger.info("Removed %s", active_marker)


def _download(args) -> None:
    from src.tounsi_llm.data_sources import download_configured_datasets

    paths = download_configured_datasets()
    logger.info("Downloaded datasets: %s", {key: str(value) for key, value in paths.items()})


def _audit(args) -> None:
    from src.tounsi_llm.data_audit import audit_raw_datasets
    from src.tounsi_llm.data_sources import download_configured_datasets

    raw_paths = download_configured_datasets()
    reports = audit_raw_datasets(raw_paths)
    logger.info("Data audit reports: %s", reports)


def _prepare(args) -> None:
    from src.tounsi_llm.data_prep import compute_quality_stats, prepare_all_data
    from src.tounsi_llm.data_sources import download_configured_datasets

    raw_paths = download_configured_datasets()
    effective_max_self_sup_texts = args.max_self_sup_texts
    if effective_max_self_sup_texts is None:
        effective_max_self_sup_texts = CFG.self_sup_target_texts

    outputs = prepare_all_data(
        raw_paths,
        max_sft_samples=args.max_sft_samples,
        max_self_sup_texts=effective_max_self_sup_texts,
    )
    logger.info("Prepared datasets: %s", {key: {k: str(v) for k, v in value.items()} for key, value in outputs.items()})

    stats_source = []
    for split_name in ["train", "val"]:
        path = outputs.get("self_sup", {}).get(split_name)
        if not path:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                stats_source.append(json.loads(line)["text"])
    if stats_source:
        logger.info(
            "Self-supervised stats: %s",
            json.dumps(compute_quality_stats(stats_source[:500]), ensure_ascii=True),
        )


def _validate(args) -> None:
    from src.tounsi_llm.validation import validate_domain_assets

    report = validate_domain_assets(write_report=True)
    logger.info("validation summary: %s", report.get("summary", {}))
    readiness = report.get("production_readiness", {})
    logger.info("production readiness: %s", readiness)
    enforce_no_go = args.fail_on_no_go or args.stage == "full"
    if enforce_no_go and readiness.get("go_no_go") != "GO":
        failing = readiness.get("failing_checks", [])
        raise SystemExit(f"Validation verdict is NO_GO. failing_checks={failing}")


def _self_sup(args) -> None:
    from src.tounsi_llm.training import train_self_supervised

    metrics = train_self_supervised(
        epochs=args.self_sup_epochs,
        max_steps=args.self_sup_max_steps,
        max_seq_length=args.self_sup_max_seq_len,
        continue_from_adapter=not args.self_sup_fresh_adapter,
    )
    logger.info("self_sup metrics: %s", metrics)


def _sft(args) -> None:
    from src.tounsi_llm.training import train_sft

    metrics = train_sft(
        epochs=args.sft_epochs,
        max_steps=args.sft_max_steps,
        max_seq_length=args.sft_max_seq_len,
    )
    logger.info("sft metrics: %s", metrics)


def _dpo(args) -> None:
    from src.tounsi_llm.training import train_dpo

    metrics = train_dpo(epochs=args.dpo_epochs, max_steps=args.dpo_max_steps, beta=args.dpo_beta)
    logger.info("dpo metrics: %s", metrics)


def _promote(args) -> None:
    from src.tounsi_llm.training import promote_adapter

    result = promote_adapter(source_variant=args.prod_source_variant)
    logger.info("Promoted adapter: %s", result)


def _eval(args) -> None:
    from src.tounsi_llm.evaluation import run_evaluation

    metrics = run_evaluation(
        model_variant=args.eval_model_variant,
        runtime_mode=args.eval_runtime_mode,
        max_cases=args.eval_max_cases,
    )
    logger.info(
        "eval metrics: %s",
        {key: value for key, value in metrics.get("inference_summary", {}).items() if key != "details"},
    )


def _serve(args) -> None:
    uvicorn.run("src.tounsi_llm.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
