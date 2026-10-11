#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:$PWD/ttcl/.runtime/structured_memory_deps"
python_bin="ttcl/.runtime/alf_delta_env/bin/python"
prepare_pid="${1:?Pass the confirmed live preparation PID}"
while kill -0 "$prepare_pid" 2>/dev/null; do
    sleep 15
done
test -f data/annotations/alfworld_feedback_stream_v60_20261010.json
"$python_bin" -u -m ttcl.trajectory_hyperlora.train_alfworld_feedback_stream_v60
"$python_bin" -u -m ttcl.trajectory_hyperlora.evaluate_alfworld_feedback_stream_v60
"$python_bin" -u -m ttcl.trajectory_hyperlora.audit_alfworld_feedback_stream_v60
