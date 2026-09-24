#!/usr/bin/env bash
# Supports both `sh script.sh` and `bash script.sh`.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/seal_env/bin/python}"
SEAL_MODEL="${SEAL_MODEL:-$ROOT/current_work/SEAL/models/iter2}"
BASE_MODEL="${BASE_MODEL:-$ROOT/current_work/delta-Mem/model/Qwen3-4B-Instruct-2507}"
RUN_ROOT="${RUN_ROOT:-$ROOT/ttcl/results/seal_bsm_$(date +%Y%m%d_%H%M%S)}"
MODES="${MODES:-base_frozen seal_frozen base_ttt seal_ttt}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }
if [[ -e "$RUN_ROOT" ]]; then
    echo "Use a new RUN_ROOT; existing results are preserved: $RUN_ROOT" >&2
    exit 1
fi
mkdir -p "$RUN_ROOT"
echo "SEAL BSM evaluation: $RUN_ROOT"
echo "GPU=$CUDA_VISIBLE_DEVICES modes=$MODES"
for label in $MODES; do
    case "$label" in
        base_frozen) model="$BASE_MODEL"; mode=frozen ;;
        seal_frozen) model="$SEAL_MODEL"; mode=frozen ;;
        base_ttt) model="$BASE_MODEL"; mode=ttt ;;
        seal_ttt) model="$SEAL_MODEL"; mode=ttt ;;
        *) echo "Unknown mode: $label" >&2; exit 1 ;;
    esac
    args=(
        "$ROOT/ttcl/seal/run_seal_benchmark.py"
        --model "$model" --mode "$mode" --output-dir "$RUN_ROOT/$label"
        --data-path "${DATA_PATH:-$ROOT/current_work/continual-learning-bench/data/blind_spectrum_monitoring/mixed_grid_lifecycle.jsonl}"
        --update-every "${UPDATE_EVERY:-4}" --memory-window "${MEMORY_WINDOW:-4}"
        --learning-rate "${LEARNING_RATE:-1e-4}" --train-epochs "${TRAIN_EPOCHS:-5}"
        --max-new-tokens "${MAX_NEW_TOKENS:-1024}" --material-tokens "${MATERIAL_TOKENS:-1536}"
        --recall-tokens "${RECALL_TOKENS:-512}"
    )
    if [[ -n "${NUM_SCANS:-}" ]]; then args+=(--num-scans "$NUM_SCANS"); fi
    echo "Starting $label: $model"
    "$PYTHON_BIN" "${args[@]}" "$@" 2>&1 | tee "$RUN_ROOT/$label.log"
done
"$PYTHON_BIN" - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {}
print("\nBSM comparison (higher IoU is better):")
for path in sorted(root.glob("*/metrics.json")):
    metrics = json.loads(path.read_text())
    summary[path.parent.name] = {key: metrics[key] for key in (
        "mean_score", "completed", "num_updates", "invalid_reports", "elapsed_seconds"
    )}
    print(f"{path.parent.name:16s} IoU={metrics['mean_score']:.4f} "
          f"scans={metrics['completed']} updates={metrics['num_updates']} "
          f"invalid={metrics['invalid_reports']}")
(root / "comparison.json").write_text(json.dumps(summary, indent=2))
PY
