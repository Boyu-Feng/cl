#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:$PWD/ttcl/.runtime/structured_memory_deps"
python_bin="${TTCL_PYTHON:-ttcl/.runtime/alf_delta_env/bin/python}"
"$python_bin" -u -m ttcl.trajectory_hyperlora.alfworld_online_shared_trajectory_v53b prepare
"$python_bin" -u -m ttcl.trajectory_hyperlora.alfworld_online_shared_trajectory_v53b evaluate
"$python_bin" -u -m ttcl.trajectory_hyperlora.audit_alfworld_online_shared_trajectory_v53b evaluate
