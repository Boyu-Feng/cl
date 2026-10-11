#!/usr/bin/env bash
set -euo pipefail
cd "${TTCL_WORKSPACE:-$(cd "$(dirname "$0")/../.." && pwd)}"
export PYTHONPATH="$PWD:$PWD/ttcl/.runtime/structured_memory_deps${PYTHONPATH:+:$PYTHONPATH}"
ttcl_python="${TTCL_PYTHON:-$PWD/ttcl/.runtime/alf_delta_env/bin/python}"
wait_pid="${1:?Pass the existing v61 GPU2 evaluator PID}"
echo "Waiting for GPU2 evaluator PID $wait_pid before the cold joint experiment"
while kill -0 "$wait_pid" 2>/dev/null; do sleep 30; done
for mode in hybrid hyper residual; do
    echo "Starting v63 $mode: fresh H, own stream, no old checkpoints"
    "$ttcl_python" -u -m ttcl.trajectory_hyperlora.alfworld_cold_joint_memory_v63 evaluate --mode "$mode" --device cuda:2 --gpu-fraction .68
    "$ttcl_python" -u -m ttcl.trajectory_hyperlora.audit_alfworld_cold_joint_memory_v63 evaluate --mode "$mode"
done
