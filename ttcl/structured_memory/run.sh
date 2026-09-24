#!/usr/bin/env bash
set -euo pipefail
TTCL_WORKSPACE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec "${PYTHON:-/home/fengboyu/miniconda3/envs/seal_env/bin/python}" \
  "$TTCL_WORKSPACE/ttcl/structured_memory/launch.py" "$@"
