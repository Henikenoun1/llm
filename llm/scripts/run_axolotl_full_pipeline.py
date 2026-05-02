#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
PHASE1_CONFIG_PATH = ROOT / "configs" / "1_ssf.yaml"
RUN_SUMMARY_PATH = ROOT / "artifacts" / "manifests" / "axolotl_train_run.json"


def _run_command(cmd: list[str]) -> None:
    print("[run] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def _prepend_bin_dir(tool_path: str) -> None:
    path = Path(tool_path)
    if not path.exists():
        return

    bin_dir = path.resolve().parent.as_posix()
    current_path = os.environ.get("PATH", "")
    if current_path:
        os.environ["PATH"] = f"{bin_dir}:{current_path}"
    else:
        os.environ["PATH"] = bin_dir


def _resolve_axolotl_binary(axolotl_bin: str) -> str | None:
    candidate = Path(axolotl_bin)
    if candidate.exists():
        return candidate.resolve().as_posix()

    resolved = shutil.which(axolotl_bin)
    if resolved:
        return resolved
    return None


def _relative_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _read_phase1_num_epochs(config_path: Path = PHASE1_CONFIG_PATH) -> float:
    pattern = re.compile(r"^\s*num_epochs:\s*([0-9]+(?:\.[0-9]+)?)\s*$")
    for line in config_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            return float(match.group(1))
    raise RuntimeError(f"Could not find num_epochs in {config_path}")


def _resolve_config_path(path_str: str) -> Path:
    candidate = Path(path_str.strip())
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def _ensure_phase1_resume_compatibility() -> dict[str, Any] | None:
    text = PHASE1_CONFIG_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()
    resume_pattern = re.compile(r"^\s*resume_from_checkpoint:\s*(.+?)\s*$")
    scalar_patterns = {
        "lora_r": re.compile(r"^\s*lora_r:\s*([0-9]+(?:\.[0-9]+)?)\s*$"),
        "lora_alpha": re.compile(r"^\s*lora_alpha:\s*([0-9]+(?:\.[0-9]+)?)\s*$"),
        "lora_dropout": re.compile(r"^\s*lora_dropout:\s*([0-9]+(?:\.[0-9]+)?)\s*$"),
    }

    resume_path: str | None = None
    for line in lines:
        match = resume_pattern.match(line)
        if match:
            resume_path = match.group(1).strip()
            break

    if not resume_path:
        return None

    adapter_config_path = _resolve_config_path(resume_path) / "adapter_config.json"
    if not adapter_config_path.exists():
        return None

    data = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    expected = {
        "lora_r": data.get("r"),
        "lora_alpha": data.get("lora_alpha"),
        "lora_dropout": data.get("lora_dropout"),
    }

    replacements: dict[str, str] = {}
    for key, value in expected.items():
        if isinstance(value, (int, float)) and value >= 0:
            if isinstance(value, float) and value.is_integer():
                replacements[key] = str(int(value))
            else:
                replacements[key] = str(value)

    if not replacements:
        return None

    updated = False
    rewritten: list[str] = []
    for line in lines:
        replaced = False
        for key, pattern in scalar_patterns.items():
            match = pattern.match(line)
            if match and key in replacements:
                current = match.group(1)
                target = replacements[key]
                if current != target:
                    rewritten.append(f"{key}: {target}")
                    updated = True
                else:
                    rewritten.append(line)
                replaced = True
                break
        if not replaced:
            rewritten.append(line)

    if updated:
        PHASE1_CONFIG_PATH.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
        print(
            "[phase1-compat] aligned SSF config with resume checkpoint "
            f"{_relative_path(adapter_config_path.parent)} "
            f"(lora_r={replacements.get('lora_r')}, "
            f"lora_alpha={replacements.get('lora_alpha')}, "
            f"lora_dropout={replacements.get('lora_dropout')})",
            flush=True,
        )

    return {
        "resume_from_checkpoint": _relative_path(adapter_config_path.parent),
        "updated": updated,
        "lora_r": replacements.get("lora_r"),
        "lora_alpha": replacements.get("lora_alpha"),
        "lora_dropout": replacements.get("lora_dropout"),
    }


def _update_phase1_config(*, num_epochs: float | None = None, resume_from_checkpoint: str | None = None) -> None:
    text = PHASE1_CONFIG_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()

    epochs_pattern = re.compile(r"^\s*num_epochs:\s*([0-9]+(?:\.[0-9]+)?)\s*$")
    resume_pattern = re.compile(r"^\s*resume_from_checkpoint:\s*(.+?)\s*$")

    found_epochs = False
    found_resume = False
    updated: list[str] = []

    for line in lines:
        if num_epochs is not None and epochs_pattern.match(line):
            found_epochs = True
            as_int = int(num_epochs)
            value = str(as_int) if abs(num_epochs - as_int) < 1e-9 else str(num_epochs)
            updated.append(f"num_epochs: {value}")
            continue

        if resume_pattern.match(line):
            found_resume = True
            if resume_from_checkpoint:
                updated.append(f"resume_from_checkpoint: {resume_from_checkpoint}")
            else:
                updated.append(line)
            continue

        updated.append(line)

    if num_epochs is not None and not found_epochs:
        raise RuntimeError("num_epochs key is missing from phase 1 config")

    if resume_from_checkpoint and not found_resume:
        insert_idx = next((idx for idx, line in enumerate(updated) if line.startswith("datasets:")), len(updated))
        updated.insert(insert_idx, f"resume_from_checkpoint: {resume_from_checkpoint}")

    PHASE1_CONFIG_PATH.write_text("\n".join(updated) + "\n", encoding="utf-8")


def _find_latest_checkpoint() -> Path | None:
    candidates: list[Path] = []
    roots = [ROOT / "outputs" / "1_ssf", ROOT / "artifacts" / "checkpoints" / "self_sup"]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("checkpoint-*"):
            if not path.is_dir():
                continue
            if (path / "trainer_state.json").exists() or (path / "adapter_model.safetensors").exists():
                candidates.append(path)

    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _read_best_eval_loss(checkpoint_path: Path | None) -> float | None:
    if checkpoint_path is None:
        return None

    trainer_state_path = checkpoint_path / "trainer_state.json"
    if not trainer_state_path.exists():
        return None

    payload = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    best_metric = payload.get("best_metric")
    if isinstance(best_metric, (int, float)):
        return float(best_metric)

    best_eval_loss: float | None = None
    for row in payload.get("log_history", []):
        if not isinstance(row, dict):
            continue
        value = row.get("eval_loss")
        if isinstance(value, (int, float)):
            current = float(value)
            if best_eval_loss is None or current < best_eval_loss:
                best_eval_loss = current
    return best_eval_loss


def _train_ssf_with_auto_extension(
    *,
    axolotl_command: Callable[..., list[str]],
    target_best_eval_loss: float,
    max_auto_extensions: int,
) -> dict[str, Any]:
    attempts = 0
    history: list[dict[str, Any]] = []

    while True:
        _ensure_phase1_resume_compatibility()
        _run_command(axolotl_command("train", "configs/1_ssf.yaml"))

        latest_checkpoint = _find_latest_checkpoint()
        best_eval_loss = _read_best_eval_loss(latest_checkpoint)
        passed = best_eval_loss is not None and best_eval_loss <= target_best_eval_loss

        history.append(
            {
                "attempt": attempts + 1,
                "best_eval_loss": best_eval_loss,
                "target_best_eval_loss": target_best_eval_loss,
                "checkpoint": _relative_path(latest_checkpoint),
                "passed": passed,
            }
        )

        if passed:
            return {
                "attempts": attempts + 1,
                "extensions": attempts,
                "history": history,
                "final_best_eval_loss": best_eval_loss,
                "final_checkpoint": _relative_path(latest_checkpoint),
            }

        if attempts >= max_auto_extensions:
            raise RuntimeError(
                "SSF metric gate did not pass after automatic +1 epoch extensions. "
                f"best_eval_loss={best_eval_loss}, target={target_best_eval_loss}, max_extensions={max_auto_extensions}"
            )

        attempts += 1
        current_epochs = _read_phase1_num_epochs()
        next_epochs = current_epochs + 1
        resume_path = _relative_path(latest_checkpoint)
        _update_phase1_config(num_epochs=next_epochs, resume_from_checkpoint=resume_path)

        print(
            "[ssf-gate] eval gate not passed; extending phase 1 by +1 epoch "
            f"(next num_epochs={next_epochs}, resume={resume_path})",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full Axolotl training pipeline with SSF metric gating.")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--axolotl-bin", default="axolotl")
    parser.add_argument("--ssf-target-best-eval-loss", type=float, default=3.20)
    parser.add_argument("--ssf-max-auto-extensions", type=int, default=2)
    args = parser.parse_args()

    # Ensure nested calls resolve to the intended environment.
    _prepend_bin_dir(args.python_bin)

    resolved_axolotl_bin = _resolve_axolotl_binary(args.axolotl_bin)
    if resolved_axolotl_bin:
        _prepend_bin_dir(resolved_axolotl_bin)
        print(f"[axolotl-runner] using CLI binary: {resolved_axolotl_bin}", flush=True)
    else:
        print(
            "[axolotl-runner] no axolotl CLI binary found; "
            "falling back to scripts/axolotl_cli.py via the active Python environment",
            flush=True,
        )

    def axolotl_command(subcommand: str, *command_args: str) -> list[str]:
        if resolved_axolotl_bin:
            return [resolved_axolotl_bin, subcommand, *command_args]
        return [args.python_bin, "scripts/axolotl_cli.py", subcommand, *command_args]

    _run_command([args.python_bin, "scripts/prepare_axolotl_pipeline.py", "--mode", "bootstrap"])
    _run_command([args.python_bin, "scripts/prepare_axolotl_pipeline.py", "--mode", "validate"])

    ssf_summary = _train_ssf_with_auto_extension(
        axolotl_command=axolotl_command,
        target_best_eval_loss=args.ssf_target_best_eval_loss,
        max_auto_extensions=args.ssf_max_auto_extensions,
    )

    _run_command(axolotl_command("merge-lora", "configs/1_ssf.yaml", "--lora-model-dir", "outputs/1_ssf"))
    _run_command([args.python_bin, "scripts/prepare_axolotl_pipeline.py", "--mode", "after-ssf"])
    _run_command([args.python_bin, "scripts/prepare_axolotl_pipeline.py", "--mode", "validate"])

    _run_command(axolotl_command("train", "configs/2_sft.yaml"))
    _run_command(axolotl_command("merge-lora", "configs/2_sft.yaml", "--lora-model-dir", "outputs/2_sft"))
    _run_command([args.python_bin, "scripts/prepare_axolotl_pipeline.py", "--mode", "after-sft"])
    _run_command([args.python_bin, "scripts/prepare_axolotl_pipeline.py", "--mode", "validate"])

    _run_command(axolotl_command("train", "configs/3_intent_slot.yaml"))
    _run_command(axolotl_command("merge-lora", "configs/3_intent_slot.yaml", "--lora-model-dir", "outputs/3_intent_slot"))
    _run_command([args.python_bin, "scripts/prepare_axolotl_pipeline.py", "--mode", "after-intent-slot"])
    _run_command([args.python_bin, "scripts/prepare_axolotl_pipeline.py", "--mode", "validate"])

    _run_command(axolotl_command("train", "configs/4_orpo.yaml"))

    summary = {
        "pipeline": "axolotl_full",
        "ssf_metric_gate": ssf_summary,
        "configs": {
            "phase1": "configs/1_ssf.yaml",
            "phase2": "configs/2_sft.yaml",
            "phase3": "configs/3_intent_slot.yaml",
            "phase4": "configs/4_orpo.yaml",
        },
    }
    RUN_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUN_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
