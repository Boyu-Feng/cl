#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/seal_env/bin/python}"
exec "$PYTHON_BIN" -m ttcl.memory_writer.launch --gpus "${GPUS:-0,1}" "$@"
