#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:$PWD/ttcl/.runtime/structured_memory_deps"
python_bin="${TTCL_PYTHON:-ttcl/.runtime/alf_delta_env/bin/python}"
# The v60 auditor must finish successfully before binding the final models.
audit_pid="${1:-32898}"
while kill -0 "$audit_pid" 2>/dev/null; do sleep 15; done
"$python_bin" -u -m ttcl.trajectory_hyperlora.evaluate_feedback_unseen_v68 prepare
"$python_bin" -u -m ttcl.trajectory_hyperlora.evaluate_feedback_unseen_v68 evaluate
"$python_bin" -u -m ttcl.trajectory_hyperlora.evaluate_feedback_unseen_v68 audit
