#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]


def _call_without_wrapper_args(callback: Callable[..., Any], /, **kwargs: Any) -> Any:
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0]]
        return callback(**kwargs)
    finally:
        sys.argv = original_argv


def _patch_axolotl_runtime_compat() -> None:
    """
    Axolotl 0.5.2 imports `importlib.metadata.version` as `get_distribution`
    but later calls `.version` on the returned string. Patch that runtime bug
    in-process so training can start without editing site-packages.
    """
    import axolotl.train as ax_train

    current = getattr(ax_train, "get_distribution", None)
    if current is None:
        return

    try:
        sample = current("axolotl")
    except Exception:
        return

    if not isinstance(sample, str):
        return

    def _compat_get_distribution(name: str) -> SimpleNamespace:
        return SimpleNamespace(version=current(name))

    ax_train.get_distribution = _compat_get_distribution


def _run_train(config: str) -> None:
    from axolotl.cli.train import do_cli

    _patch_axolotl_runtime_compat()
    _call_without_wrapper_args(do_cli, config=str(Path(config)))


def _run_merge_lora(config: str, lora_model_dir: str | None) -> None:
    from axolotl.cli.merge_lora import do_cli

    _patch_axolotl_runtime_compat()
    kwargs: dict[str, Any] = {"config": str(Path(config))}
    if lora_model_dir:
        kwargs["lora_model_dir"] = lora_model_dir
    _call_without_wrapper_args(do_cli, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Thin Axolotl wrapper for environments without the axolotl CLI entrypoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run Axolotl training for the given config.")
    train_parser.add_argument("config", help="Path to the Axolotl config YAML.")

    merge_parser = subparsers.add_parser("merge-lora", help="Merge a LoRA adapter into its base model.")
    merge_parser.add_argument("config", help="Path to the Axolotl config YAML.")
    merge_parser.add_argument("--lora-model-dir", default=None, help="Directory containing the trained LoRA adapter.")

    args = parser.parse_args()

    if args.command == "train":
        _run_train(args.config)
        return

    if args.command == "merge-lora":
        _run_merge_lora(args.config, args.lora_model_dir)
        return

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
