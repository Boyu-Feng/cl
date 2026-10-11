#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:$PWD/ttcl/.runtime/structured_memory_deps"

python_bin="ttcl/.runtime/alf_delta_env/bin/python"
teacher_log="results/trajectory_hyperlora/alfworld_online_own_trajectory_v52b_run_20261010.log"
teacher_model="results/trajectory_hyperlora/alfworld_online_own_trajectory_v52b_20261010.pt"

while [[ ! -f "$teacher_model" ]]; do
    if rg -q '^Traceback' "$teacher_log"; then
        echo 'v52b failed; v53 was not started' >&2
        exit 1
    fi
    sleep 30
done

# Let the teacher finish writing its checkpoint before binding the checksum.
sleep 30
"$python_bin" -m ttcl.trajectory_hyperlora.audit_alfworld_online_own_trajectory_v52b evaluate
"$python_bin" -m ttcl.trajectory_hyperlora.alfworld_online_shared_trajectory_v53 prepare
"$python_bin" -u -m ttcl.trajectory_hyperlora.alfworld_online_shared_trajectory_v53 evaluate
"$python_bin" -m ttcl.trajectory_hyperlora.audit_alfworld_online_shared_trajectory_v53 evaluate
