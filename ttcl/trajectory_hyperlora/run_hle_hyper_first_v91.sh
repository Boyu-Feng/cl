#!/usr/bin/env bash
set -euo pipefail
cd "${TTCL_WORKSPACE:-$(cd "$(dirname "$0")/../.." && pwd)}"
export PYTHONPATH="$PWD:$PWD/ttcl/.runtime/structured_memory_deps${PYTHONPATH:+:$PYTHONPATH}"
task_python="${TTCL_PYTHON:-$PWD/ttcl/.runtime/alf_delta_env/bin/python}"
"$task_python" -u -m ttcl.trajectory_hyperlora.hle_hyper_first_v91 evaluate --device cuda:1
"$task_python" -m ttcl.trajectory_hyperlora.hle_hyper_first_v91 audit
"$task_python" -m ttcl.trajectory_hyperlora.score_hle_mcq_pilot \
  --inputs-dir data/hle/failure_retry_pilot_20261010 \
  --responses results/trajectory_hyperlora/hle_hyper_first_v91_responses_20261010.json \
  --output results/trajectory_hyperlora/hle_hyper_first_v91_scores_20261010.json
"$task_python" -m ttcl.trajectory_hyperlora.analyze_hle_hyper_first_v91
"$task_python" - <<'PY'
import hashlib,json
from pathlib import Path
frozen=json.loads(Path('data/annotations/hle_hyper_v91_code_freeze_20261010.json').read_text())
for name,expected in frozen['files'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected, name
print('H-first pipeline code freeze verified',flush=True)
PY
"$task_python" -m ttcl.trajectory_hyperlora.prepare_hle_hyper_retry_v91 \
  --inputs data/hle/failure_retry_pilot_20261010/actor_inputs.json \
  --responses results/trajectory_hyperlora/hle_hyper_first_v91_responses_20261010.json \
  --scores results/trajectory_hyperlora/hle_hyper_first_v91_scores_20261010.json \
  --output data/annotations/hle_hyper_retry_v91_20261010.json
"$task_python" -u -m ttcl.trajectory_hyperlora.evaluate_hle_hyper_retry_v91 \
  --plan data/annotations/hle_hyper_retry_v91_20261010.json \
  --inputs data/hle/failure_retry_pilot_20261010/actor_inputs.json \
  --responses results/trajectory_hyperlora/hle_hyper_first_v91_responses_20261010.json \
  --scores results/trajectory_hyperlora/hle_hyper_first_v91_scores_20261010.json \
  --output results/trajectory_hyperlora/hle_hyper_retry_v91_20261010.json --device cuda:1
"$task_python" -m ttcl.trajectory_hyperlora.audit_hle_hyper_retry_v91 \
  --plan data/annotations/hle_hyper_retry_v91_20261010.json \
  --baseline results/trajectory_hyperlora/hle_hyper_first_v91_responses_20261010.json \
  --scores results/trajectory_hyperlora/hle_hyper_first_v91_scores_20261010.json \
  --retries results/trajectory_hyperlora/hle_hyper_retry_v91_20261010.json \
  --keys data/hle/failure_retry_pilot_20261010/grading_keys.json \
  --output results/trajectory_hyperlora/hle_hyper_retry_v91_20261010.audit.json
