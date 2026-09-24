#!/usr/bin/env bash
set -euo pipefail

# Start both standalone Delta-Mem forms in the background. The stateful run
# keeps history across scans; the stateless run resets before each scan.
# Override paths and CUDA_VISIBLE_DEVICES when invoking this script.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TTCL_ROOT="$ROOT/ttcl"
RUN_ROOT="${RUN_ROOT:-$TTCL_ROOT/results/delta_bsm}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/combined.log}"
STATEFUL_OUT="${STATEFUL_OUT:-$RUN_ROOT/stateful}"
STATELESS_OUT="${STATELESS_OUT:-$RUN_ROOT/stateless}"
PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p "$LOG_DIR" "$STATEFUL_OUT" "$STATELESS_OUT"

BASE_MODEL="${BASE_MODEL:-$ROOT/current_work/delta-Mem/model/Qwen3-4B-Instruct-2507}"
DELTA_ADAPTER="${DELTA_ADAPTER:-$ROOT/current_work/delta-Mem/model/delta-mem_qwen3_4b-instruct}"
DATA_PATH="${DATA_PATH:-$ROOT/current_work/continual-learning-bench/data/blind_spectrum_monitoring/mixed_grid_lifecycle.jsonl}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
CONDA_LIB_DIR="${CONDA_PREFIX:-}/lib"
if [[ -d "$CONDA_LIB_DIR" ]]; then
  LD_LIBRARY_PATH="$CONDA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
fi
RESUME="${RESUME:-1}"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH

cat > "$RUN_ROOT/run_config.txt" <<EOF
started_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
base_model=$BASE_MODEL
delta_adapter=$DELTA_ADAPTER
data_path=$DATA_PATH
device=$DEVICE
dtype=$DTYPE
attn_implementation=$ATTN_IMPLEMENTATION
max_new_tokens=$MAX_NEW_TOKENS
num_trajectories=${NUM_TRAJECTORIES:-all}
modes=stateful,stateless
EOF

run_mode() {
  local mode="$1" out_dir="$2" mode_log="$3"
  local args=(
    "$TTCL_ROOT/delta_mem/run_deltamem_standalone.py"
    --base-model "$BASE_MODEL"
    --delta-adapter "$DELTA_ADAPTER"
    --data-path "$DATA_PATH"
    --output-dir "$out_dir"
    --device "$DEVICE"
    --dtype "$DTYPE"
    --attn-implementation "$ATTN_IMPLEMENTATION"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --mode "$mode"
  )
  if [[ -n "$NUM_TRAJECTORIES" ]]; then
    args+=(--num-trajectories "$NUM_TRAJECTORIES")
  fi
  if [[ "$RESUME" == 0 ]]; then
    args+=(--no-resume)
  fi
  echo "===== starting $mode run ====="
  "$PYTHON_BIN" "${args[@]}" 2>&1 | tee -a "$mode_log"
}

if [[ "${RUNNER_CHILD:-0}" != 1 ]]; then
  echo "Starting stateful and stateless Delta-Mem benchmark runs."
  nohup env RUNNER_CHILD=1 "$0" > >(tee -a "$LOG_FILE") 2>&1 < /dev/null &
  pid=$!
  echo "$pid" > "$RUN_ROOT/run.pid"
  echo "pid=$pid"
  echo "Live combined log: tail -f '$LOG_FILE'"
  echo "Stateful progress: watch -n 5 'cat \"$STATEFUL_OUT/progress.json\"'"
  echo "Stateless progress: watch -n 5 'cat \"$STATELESS_OUT/progress.json\"'"
  exit 0
fi

run_mode stateful "$STATEFUL_OUT" "$LOG_DIR/stateful.log"
run_mode stateless "$STATELESS_OUT" "$LOG_DIR/stateless.log"
echo "===== both runs completed ====="
