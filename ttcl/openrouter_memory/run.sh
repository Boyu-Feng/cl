#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_PYTHON="${EXPERIMENT_PYTHON:-/home/fengboyu/miniconda3/envs/seal_env/bin/python}"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$PROJECT_ROOT/ttcl/.runtime/structured_memory_deps:$PROJECT_ROOT:$PROJECT_ROOT/current_work/continual-learning-bench${PYTHONPATH:+:$PYTHONPATH}"
exec "$EXPERIMENT_PYTHON" "$PROJECT_ROOT/ttcl/openrouter_memory/launch.py" "$@"
