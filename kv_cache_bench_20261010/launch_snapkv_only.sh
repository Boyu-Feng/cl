#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
python_exec="$root/ttcl/.runtime/alf_delta_env/bin/python"
out="$root/results/kv_cache_bench_20261010"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case "${1:-}" in
  locomo)
    export CUDA_VISIBLE_DEVICES=6
    "$python_exec" -u kv_cache_bench_20261010/run_grouped.py \
      --suite locomo --methods snapkv kvzip --device cuda:0 \
      --output "$out/locomo_snapkv_kvzip_seed42.jsonl"
    ;;
  mab)
    export CUDA_VISIBLE_DEVICES=2
    "$python_exec" -u kv_cache_bench_20261010/run_grouped.py \
      --suite mab --all-native --methods snapkv kvzip --device cuda:0 \
      --categories selective_forgetting test_time_learning \
      --output "$out/mab_snapkv_kvzip_seed42.jsonl"
    "$python_exec" -u kv_cache_bench_20261010/run_grouped.py \
      --suite mab --all-native --methods snapkv kvzip --device cuda:0 \
      --categories accurate_retrieval long_range_understanding \
      --output "$out/mab_snapkv_kvzip_seed42.jsonl"
    ;;
  *)
    echo "usage: $0 locomo|mab" >&2
    exit 2
    ;;
esac
