#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/seal_env/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
exec "$PYTHON_BIN" "$ROOT/ttcl/llm_memory/run_benchmark.py" \
  --output-dir "${RUN_ROOT:-$ROOT/ttcl/results/llm_memory/run_$(date +%Y%m%d_%H%M%S)}" \
  --num-scans "${NUM_SCANS:-12}" --mode "${MODE:-all}" "$@"
