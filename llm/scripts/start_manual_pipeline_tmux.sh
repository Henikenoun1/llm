#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="tounsi-manual-full"
DRY_RUN=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-name)
      SESSION_NAME="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS_DIR="$ROOT_DIR/logs"
mkdir -p "$LOGS_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -x "$ROOT_DIR/../.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/../.venv/bin/python"
fi

timestamp="$(date +%F_%H%M%S)"
LOG_FILE="$LOGS_DIR/manual_pipeline_${timestamp}.log"
LATEST_LINK="$LOGS_DIR/latest_manual_pipeline.log"
ln -sfn "$(basename "$LOG_FILE")" "$LATEST_LINK"

EXTRA_ARGS_ESCAPED=""
if ((${#EXTRA_ARGS[@]})); then
  printf -v EXTRA_ARGS_ESCAPED '%q ' "${EXTRA_ARGS[@]}"
fi
printf -v TRAIN_LOOP 'stdbuf -oL -eL %q scripts/manual_full_pipeline.py %s2>&1 | tee %q' "$PYTHON_BIN" "$EXTRA_ARGS_ESCAPED" "$LOG_FILE"
printf -v STATUS_LOOP 'while true; do clear; %q scripts/training_status.py; sleep 5; done' "$PYTHON_BIN"
GPU_LOOP='while true; do clear; nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader; sleep 2; done'

if (( DRY_RUN )); then
  printf 'session=%s\n' "$SESSION_NAME"
  printf 'log=%s\n' "$LOG_FILE"
  printf 'train=%s\n' "$TRAIN_LOOP"
  printf 'status=%s\n' "$STATUS_LOOP"
  printf 'gpu=%s\n' "$GPU_LOOP"
  exit 0
fi

if pgrep -af 'scripts/manual_full_pipeline.py|scripts/prepare_axolotl_pipeline.py|scripts/axolotl_cli.py|scripts/run_axolotl_full_pipeline.py' >/dev/null; then
  echo "another pipeline is already running"
  pgrep -af 'scripts/manual_full_pipeline.py|scripts/prepare_axolotl_pipeline.py|scripts/axolotl_cli.py|scripts/run_axolotl_full_pipeline.py'
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  echo "Attach with: tmux attach -t $SESSION_NAME"
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" -n live -c "$ROOT_DIR" "bash -lc '$TRAIN_LOOP'"
tmux split-window -h -t "$SESSION_NAME:live" -c "$ROOT_DIR" "bash -lc '$STATUS_LOOP'"
tmux split-window -v -t "$SESSION_NAME:live.1" -c "$ROOT_DIR" "bash -lc '$GPU_LOOP'"
tmux select-layout -t "$SESSION_NAME:live" main-vertical >/dev/null
tmux select-pane -t "$SESSION_NAME:live.0"

echo "tmux session started: $SESSION_NAME"
echo "Training log: $LOG_FILE"
echo "Attach with: tmux attach -t $SESSION_NAME"
