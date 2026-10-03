# Generic public-action projection candidate

`public_action_projection.py` replaces the task-shaped final-action operators
from `general_evidence.py` with one evidence representation for ALFWorld
commands and CLBench structured actions, plus one schema-driven review step
for CLBench actions. The old implementation and frozen runs remain separate.

After a completed episode, the method records executed public actions and
their public environment responses. Each record is bound to the exact public
trace and step by SHA-256. Actor thoughts are removed. The official scalar
reward goes only to native MemRL's Q update. At retrieval, native MemRL ranks
memories; the projection shows at most two recent public action/response pairs
per selected memory within the same 2,048-token context limit. It labels the
actor action as unverified and does not infer that an acknowledgment such as
`report recorded` confirms the action's factual content.

On a CLBench turn with retrieved evidence, the frozen actor first produces its
usual schema-valid action. At most one extra model call per instance may
revise a direct structured action. A revision must cite an available prior
source hash and validate against the current response schema; an invalid
review is logged and the original action is submitted. Tool-call actions are
passed through: their schema does not generically identify which calls are
terminal, and repeated review caused a long tool loop in a development run.
Every review call and raw/final action is saved and counted. No frequency,
width, cohort, or task-name rule selects an action. ALFWorld has no structured
final-action hook in the shared runner, so it receives the generic evidence
context but no second action review.

This is a **candidate**, not a demonstrated cross-task improvement. The runner
compares native MemRL, evidence context alone, and evidence plus action review.
This separates effects of presenting evidence from effects of the extra review
call. The existing `structured_evidence` partial run already shows that more
public evidence alone can cause negative transfer.

The 2026-10-02 pilot used the frozen `v3` implementation, before the tool-call
safety limit above. On Spectrum seed 303, all 30 three-arm cells completed.
The 12 scored suffix pairs averaged 0.215783 native versus 0.216242 for each
evidence arm; 29 extra review calls produced zero action changes. The Cohort
repeat-404 pilot was stopped after a multi-turn review loop; its first three
instances are incomplete as a scored comparison. These are development
diagnostics, not results for the current `v4` implementation. The ignored
`results/memrl_public_action_projection/20261002_pilot/` directory contains
frozen source copies, all scored cells and the stopped trajectories.

From the repository root, with the pinned mainline environment, frozen MemRL
origin and actor server prepared according to `EXPERIMENTS.md` section 4:

```bash
export TTCL_WORKSPACE="${TTCL_WORKSPACE:-$PWD}"
export TTCL_PYTHON="${TTCL_PYTHON:-$TTCL_WORKSPACE/ttcl/.runtime/alf_delta_env/bin/python}"
"$TTCL_PYTHON" -m ttcl.memrl_comparison.evaluate_public_action_projection \
  --origin "$TTCL_WORKSPACE/results/memrl_comparison/20260928_budgeted" \
  --output "$TTCL_WORKSPACE/results/memrl_public_action_projection/new_run" \
  --benchmark clbench --task blind_spectrum_monitoring --repeat 303 \
  --url http://127.0.0.1:18527
```

Use a fresh output directory for each run. `design.json` binds the input plan,
item content, implementation and runner hashes. Each arm builds its own memory
from empty; actor seeds and budgets match. Results must be reported per task on
complete common scored pairs, retaining failures and separating the projection
effect from any task-specific final-action operator.
