#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/seal_env/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
exec "$PYTHON_BIN" "$ROOT/ttcl/generic_agent/run_benchmark.py" \
  --output-dir "${RUN_ROOT:-$ROOT/ttcl/results/generic_agent/run_$(date +%Y%m%d_%H%M%S)}" \
  --num-scans "${NUM_SCANS:-12}" --mode "${MODE:-both}" "$@"
