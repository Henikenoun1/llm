#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
AXOLOTL_BIN="${AXOLOTL_BIN:-axolotl}"
SSF_TARGET_BEST_EVAL_LOSS="${SSF_TARGET_BEST_EVAL_LOSS:-3.20}"
SSF_MAX_AUTO_EXTENSIONS="${SSF_MAX_AUTO_EXTENSIONS:-2}"

if [ -x "../.venv/bin/python" ]; then
  PYTHON_BIN="../.venv/bin/python"
fi

if [ -x "../.venv/bin/axolotl" ]; then
  AXOLOTL_BIN="../.venv/bin/axolotl"
fi

if [ -d "../.venv/bin" ]; then
  export PATH="$(cd ../.venv/bin && pwd):$PATH"
fi

echo "[pipeline] Axolotl full run with SSF metric gate and auto +1 epoch extension"
"$PYTHON_BIN" scripts/run_axolotl_full_pipeline.py \
  --python-bin "$PYTHON_BIN" \
  --axolotl-bin "$AXOLOTL_BIN" \
  --ssf-target-best-eval-loss "$SSF_TARGET_BEST_EVAL_LOSS" \
  --ssf-max-auto-extensions "$SSF_MAX_AUTO_EXTENSIONS" \
  "$@"
