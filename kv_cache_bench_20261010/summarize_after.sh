#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 RUN_PID INPUT_JSONL OUTPUT_MD" >&2
  exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
while kill -0 "$1" 2>/dev/null; do
  sleep 30
done
python3 kv_cache_bench_20261010/summarize.py "$2" --output "$3"
