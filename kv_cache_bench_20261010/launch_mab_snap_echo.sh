#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 ECHOPRESS_PILOT_PID" >&2
  exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
while kill -0 "$1" 2>/dev/null; do
  sleep 15
done

export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python_exec="$root/ttcl/.runtime/alf_delta_env/bin/python"
output="$root/results/kv_cache_bench_20261010/mab_snap_echo_seed42.jsonl"

"$python_exec" -u kv_cache_bench_20261010/run_grouped.py \
  --suite mab --all-native --methods snapkv echopress --device cuda:0 \
  --categories selective_forgetting test_time_learning --output "$output"

"$python_exec" -u kv_cache_bench_20261010/run_grouped.py \
  --suite mab --all-native --methods snapkv echopress --device cuda:0 \
  --categories accurate_retrieval long_range_understanding --output "$output"

python3 kv_cache_bench_20261010/summarize.py "$output" \
  --output "$root/results/kv_cache_bench_20261010/mab_snap_echo_summary.md"
python3 kv_cache_bench_20261010/mab_diagnostics.py "$output" \
  > "$root/results/kv_cache_bench_20261010/mab_snap_echo_diagnostics.md"
