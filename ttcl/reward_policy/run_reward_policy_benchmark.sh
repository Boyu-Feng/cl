#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/seal_env/bin/python}"
MODEL="${MODEL:-$ROOT/current_work/delta-Mem/model/Qwen3-4B-Instruct-2507}"
RUN_ROOT="${RUN_ROOT:-$ROOT/ttcl/results/reward_bsm_$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }
[[ ! -e "$RUN_ROOT" ]] || { echo "Choose a new RUN_ROOT: $RUN_ROOT" >&2; exit 1; }
mkdir -p "$RUN_ROOT"
echo "Reward-feedback BSM experiment: $RUN_ROOT"
for mode in ${MODES:-frozen online}; do
    args=("$ROOT/ttcl/reward_policy/run_reward_policy_benchmark.py" --model "$MODEL" --mode "$mode"
        --output-dir "$RUN_ROOT/$mode" --num-scans "${NUM_SCANS:-12}"
        --history-scans "${HISTORY_SCANS:-4}" --warmup "${WARMUP:-4}"
        --learning-rate "${LEARNING_RATE:-1e-5}" --update-steps "${UPDATE_STEPS:-2}")
    "$PYTHON_BIN" "${args[@]}" "$@" 2>&1 | tee "$RUN_ROOT/$mode.log"
done
echo "Finished. Compare frozen/metrics.json and online/metrics.json in $RUN_ROOT"
