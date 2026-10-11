#!/usr/bin/env bash
set -euo pipefail
cd "${TTCL_WORKSPACE:-$(cd "$(dirname "$0")/../.." && pwd)}"
export PYTHONPATH="$PWD:$PWD/ttcl/.runtime/structured_memory_deps${PYTHONPATH:+:$PYTHONPATH}"
ttcl_python="${TTCL_PYTHON:-$PWD/ttcl/.runtime/alf_delta_env/bin/python}"
wait_pid="${1:?Pass the existing v60 GPU3 evaluator PID}"
echo "Waiting for GPU3 evaluator PID $wait_pid before own-stream static LoRA"
while kill -0 "$wait_pid" 2>/dev/null; do sleep 30; done
for eligibility in matched unrestricted; do
    "$ttcl_python" -u -m ttcl.trajectory_hyperlora.alfworld_cold_own_static_v65 evaluate --eligibility "$eligibility" --device cuda:3 --gpu-fraction .80
    "$ttcl_python" -u -m ttcl.trajectory_hyperlora.alfworld_cold_own_static_v65 audit --eligibility "$eligibility"
done
