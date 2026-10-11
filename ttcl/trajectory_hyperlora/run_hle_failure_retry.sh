#!/usr/bin/env bash
set -euo pipefail
cd "${TTCL_WORKSPACE:-$(cd "$(dirname "$0")/../.." && pwd)}"
export PYTHONPATH="$PWD:$PWD/ttcl/.runtime/structured_memory_deps${PYTHONPATH:+:$PYTHONPATH}"
task_python="${TTCL_PYTHON:-$PWD/ttcl/.runtime/alf_delta_env/bin/python}"
baseline_pid="${1:?Pass the live base collector PID}"
[[ "$baseline_pid" =~ ^[0-9]+$ ]]
while kill -0 "$baseline_pid" 2>/dev/null; do sleep 15; done
"$task_python" - <<'PY'
import hashlib,json
from pathlib import Path
frozen=json.loads(Path('data/annotations/hle_retry_code_freeze_20261010.json').read_text())
for name,expected in frozen['files'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected, name
print('HLE retry code freeze verified',flush=True)
PY
"$task_python" -m ttcl.trajectory_hyperlora.score_hle_mcq_pilot \
  --inputs-dir data/hle/failure_retry_pilot_20261010 \
  --responses results/trajectory_hyperlora/hle_base_responses_20261010.json \
  --output results/trajectory_hyperlora/hle_base_scores_20261010.json
"$task_python" -m ttcl.trajectory_hyperlora.prepare_hle_retry_controls \
  --inputs data/hle/failure_retry_pilot_20261010/actor_inputs.json \
  --responses results/trajectory_hyperlora/hle_base_responses_20261010.json \
  --scores results/trajectory_hyperlora/hle_base_scores_20261010.json \
  --output data/annotations/hle_retry_controls_20261010.json
"$task_python" -u -m ttcl.trajectory_hyperlora.evaluate_hle_retry_controls \
  --plan data/annotations/hle_retry_controls_20261010.json \
  --inputs data/hle/failure_retry_pilot_20261010/actor_inputs.json \
  --responses results/trajectory_hyperlora/hle_base_responses_20261010.json \
  --scores results/trajectory_hyperlora/hle_base_scores_20261010.json \
  --output results/trajectory_hyperlora/hle_retry_controls_20261010.json --device cuda:0
"$task_python" -m ttcl.trajectory_hyperlora.audit_hle_retry_controls \
  --plan data/annotations/hle_retry_controls_20261010.json \
  --baseline results/trajectory_hyperlora/hle_base_responses_20261010.json \
  --scores results/trajectory_hyperlora/hle_base_scores_20261010.json \
  --retries results/trajectory_hyperlora/hle_retry_controls_20261010.json \
  --keys data/hle/failure_retry_pilot_20261010/grading_keys.json \
  --output results/trajectory_hyperlora/hle_retry_controls_20261010.audit.json
