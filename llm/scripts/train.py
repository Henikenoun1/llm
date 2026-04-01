#!/usr/bin/env python3
"""
Pipeline entry point.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import uvicorn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tounsi_llm.config import CFG, DOMAIN_CFG, RUNTIME, logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Configurable call-center LLM pipeline")
    parser.add_argument(
        "--stage",
        choices=["download", "prepare", "self-sup", "sft", "dpo", "promote", "eval", "docs", "serve", "all"],
        default="all",
    )
    parser.add_argument("--max-sft-samples", type=int, default=None)
    parser.add_argument("--max-self-sup-texts", type=int, default=None)
    parser.add_argument("--self-sup-epochs", type=int, default=None)
    parser.add_argument("--sft-epochs", type=int, default=None)
    parser.add_argument("--dpo-epochs", type=int, default=None)
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
        "download": [_download],
        "prepare": [_prepare],
        "self-sup": [_self_sup],
        "sft": [_sft],
        "dpo": [_dpo],
        "promote": [_promote],
        "eval": [_eval],
        "docs": [_docs],
        "serve": [_serve],
        "all": [_download, _prepare, _self_sup, _sft, _dpo, _eval],
    }

    for step in stages[args.stage]:
        step(args)


def _download(args) -> None:
    from src.tounsi_llm.data_prep import download_configured_datasets

    paths = download_configured_datasets()
    logger.info("Downloaded datasets: %s", {key: str(value) for key, value in paths.items()})


def _prepare(args) -> None:
    from src.tounsi_llm.data_prep import compute_quality_stats, download_configured_datasets, prepare_all_data

    raw_paths = download_configured_datasets()
    outputs = prepare_all_data(
        raw_paths,
        max_sft_samples=args.max_sft_samples,
        max_self_sup_texts=args.max_self_sup_texts,
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


def _self_sup(args) -> None:
    from src.tounsi_llm.training import train_self_supervised

    metrics = train_self_supervised(epochs=args.self_sup_epochs)
    logger.info("self_sup metrics: %s", metrics)


def _sft(args) -> None:
    from src.tounsi_llm.training import train_sft

    metrics = train_sft(epochs=args.sft_epochs)
    logger.info("sft metrics: %s", metrics)


def _dpo(args) -> None:
    from src.tounsi_llm.training import train_dpo

    metrics = train_dpo(epochs=args.dpo_epochs)
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


def _docs(args) -> None:
    from scripts.generate_data_guide import generate_data_guide

    path = generate_data_guide()
    logger.info("Generated data guide: %s", path)


def _serve(args) -> None:
    uvicorn.run("src.tounsi_llm.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
