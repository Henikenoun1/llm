#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"

STAGES = [
    {
        "label": "1_ssf",
        "name": "SSF",
        "output_dir": ROOT / "outputs" / "1_ssf",
        "extra_roots": [ROOT / "artifacts" / "checkpoints" / "self_sup"],
    },
    {
        "label": "2_sft",
        "name": "SFT",
        "output_dir": ROOT / "outputs" / "2_sft",
        "extra_roots": [],
    },
    {
        "label": "3_intent_slot",
        "name": "INTENT_SLOT",
        "output_dir": ROOT / "outputs" / "3_intent_slot",
        "extra_roots": [],
    },
    {
        "label": "4_orpo",
        "name": "ORPO",
        "output_dir": ROOT / "outputs" / "4_orpo",
        "extra_roots": [],
    },
]


def _latest_log_path() -> Path | None:
    preferred = LOGS_DIR / "latest_train.log"
    if preferred.exists():
        return preferred.resolve()

    candidates = sorted(LOGS_DIR.glob("lancer_*.log"), key=lambda item: item.stat().st_mtime)
    return candidates[-1] if candidates else None


def _tail_lines(path: Path, limit: int = 12) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-limit:]


def _latest_stage_from_log(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None

    latest: str | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "[run]" not in line or "configs/" not in line:
            continue
        for stage in STAGES:
            marker = f"configs/{stage['label']}.yaml"
            if marker in line:
                latest = stage["name"]
    return latest


def _find_latest_checkpoint(stage: dict[str, Any]) -> Path | None:
    candidates: list[Path] = []
    roots = [stage["output_dir"], *stage["extra_roots"]]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("checkpoint-*"):
            if path.is_dir() and (path / "trainer_state.json").exists():
                candidates.append(path)

    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _read_trainer_state(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None

    trainer_state_path = path / "trainer_state.json"
    if not trainer_state_path.exists():
        return None
    return json.loads(trainer_state_path.read_text(encoding="utf-8"))


def _latest_log_value(rows: list[dict[str, Any]], key: str) -> float | None:
    for row in reversed(rows):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _stage_summary(stage: dict[str, Any]) -> str:
    checkpoint = _find_latest_checkpoint(stage)
    if checkpoint is None:
        return f"{stage['name']:<12} not started"

    state = _read_trainer_state(checkpoint) or {}
    log_history = state.get("log_history", [])
    if not isinstance(log_history, list):
        log_history = []

    global_step = state.get("global_step", "?")
    epoch = state.get("epoch", "?")
    best_metric = state.get("best_metric")
    train_loss = _latest_log_value(log_history, "loss")
    eval_loss = _latest_log_value(log_history, "eval_loss")

    parts = [
        f"{stage['name']:<12}",
        f"step={global_step}",
        f"epoch={epoch}",
        f"last_loss={train_loss if train_loss is not None else 'n/a'}",
        f"eval_loss={eval_loss if eval_loss is not None else 'n/a'}",
        f"best={best_metric if best_metric is not None else 'n/a'}",
        f"checkpoint={checkpoint.name}",
    ]
    return " | ".join(parts)


def main() -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    latest_log = _latest_log_path()
    active_stage = _latest_stage_from_log(latest_log)

    print(f"Training status | {now}")
    print(f"Root: {ROOT}")
    print(f"Latest log: {latest_log if latest_log else 'none'}")
    print(f"Active stage (from log): {active_stage or 'unknown'}")
    print("")
    print("Stages")
    for stage in STAGES:
        print(_stage_summary(stage))

    if latest_log:
        print("")
        print("Recent log tail")
        for line in _tail_lines(latest_log):
            print(line)


if __name__ == "__main__":
    main()
