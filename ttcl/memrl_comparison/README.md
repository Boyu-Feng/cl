# MemRL: matched ALFWorld / CLBench adaptation

Official repository: https://github.com/MemTensor/MemRL, commit
`c1b322ca43de36ddf64c6712f89d0095bfc35ce0` (MIT). The full downloaded
source is in `current_work/MemRL`; downloaded datasets are ignored local assets.

This experiment measures the method under the existing Qwen3-4B protocol. It
does **not** reproduce the paper's original models or 3553-task × 10-epoch
training budget. All actor weights remain frozen; Q values in episodic memory
are learned online. No pretrained memory bank or historical labels are reused.

The current outcome table and stopped-run status are in
`RESULTS_20261003.md`; detailed evidence and credit design are in
`GROUNDED_EVIDENCE.md` and `PAIRED_CREDIT_RL.md`. v21 has repeat-consistent
CLBench gains on Spectrum and Cohort, but regressed on complete ALFWorld
families; Database remains mixed and Poker uncertain. The v30 Database
readout reversed sign across two full repeats, and v32's full repeat tied
on mean score. Guided-command decoding v31 and v34 regressed on complete
ALFWorld train families. The more conservative v35 was stopped at the
user's request after 3/36 full-train pairs; it has no cross-family claim.
No single method has yet improved ALFWorld and every CLBench domain.
`counterfactual_writer_credit.py` is a CPU-only, source-bound full-horizon
writer-credit scorer; it has not trained a writer or produced benchmark scores.

## What runs unchanged

`upstream.py` compiles the official class/function definitions without modifying
their bodies. It replaces top-level dependency imports and skips server
initialization. The selected sources and ASTs are hashed in every memory store.

- `MemoryService.retrieve_query`: top five semantic query keys, then native
  similarity/Q normalization and weighted scoring, selecting up to three entries.
- `QValueUpdater.update`: `Q ← 0.7 Q + 0.3 reward`, including visits and reward EMA.
- Native proceduralization builder, failure reflection prompt, append adjustment,
  query grouping and memory addition. New memories have Q = 0; epsilon = 0.

## Declared adaptations

- Local JSON storage implements the MemOS `get/add/update` interface. The full
  upstream MemOS/Qdrant service and external OpenAI APIs are not started.
- Local BGE-M3 uses CLS pooling, L2 normalization and explicit chunk averaging
  for texts exceeding its input limit. Original defaults use OpenAI embeddings.
- Similarity statistics and threshold are fixed from input-only calibration:
  historical ALFWorld training prompts and each CLBench domain's first 20%
  public prompts. Threshold is the 80th percentile of pairwise similarities.
  No reward or test suffix is used for calibration. CLBench's primary score
  therefore uses the last 80%; all/after-first scores are secondary.
- The common actor prompts and parsing are retained. Native few-shot/ReAct
  prompts are not injected. Full trajectories remain in memory storage, while
  the actor receives whole task-plus-script/reflection entries within the prior
  2048-token memory budget. Entries that do not fit are logged and receive no Q
  credit. This projection differs from native full-trajectory prompting.
  The writer has a 768-token generation cap, matching the earlier ALF protocol.
  As in the native provider, nonempty capped generations are retained verbatim;
  `writer_token_limit_hit` marks potentially unfinished text. They are never
  extended or retried with extra tokens.
- ALFWorld uses the same 134 `valid_unseen` tasks, orders, seeds 92601/92602/92603,
  50 commands and at most three attempts. Official binary outcome maps to +1/-1
  for Q updates; reported successes retain the official 0/1 scoring.
- CLBench uses seed 42, repeats 303/404, the same four available domains and
  official scoring: spectrum 90, poker 120, database 30, cohort 20. Scalar reward
  updates Q directly; writer receives the public interactions and the official
  success/failure bit through the native branch selection and failure instruction.
  Scalar scores, grader metadata and ground-truth labels are not in writer text. When the
  official success flag is null, a script is generated without asserting
  success; the stored flag remains null. Unlike previous Mem0, MemRL consumes
  scalar rewards in its controller. CRM/codebase Docker tasks remain unavailable.
  The ALF Q cutoff of -10 is disabled for CLBench's unbounded reward scale;
  Q normalization, ranking and update equations remain unchanged.
- Both arms get a new BF16, 65536-context server baseline. Historical Mem0 had
  different KV precision/context and feedback access, so historical comparisons
  are descriptive, not controlled algorithm-only comparisons.

No new annotation targets are created: this is an online memory experiment,
not supervised fitting of a writer. Every new memory binds to its actual public
input content. Infrastructure failures and invalid outputs retain their records;
missing scores never become zero. Completed cells can be restored from their
exact memory snapshots; partial episodes cannot be silently replayed.

## Run

Use the pinned Python 3.12 mainline environment plus CLBench dependency targets
from `EXPERIMENTS.md` section 4. There is no installation into an active runtime.
Set `TTCL_WORKSPACE` / `TTCL_PYTHON` to override local paths. The historical split
and training manifests must exist; they cannot be reconstructed from invented
scores or absent checkpoints.

```bash
export TTCL_WORKSPACE="$PWD"
export TTCL_PYTHON="$PWD/ttcl/.runtime/alf_delta_env/bin/python"
export TTCL_BENCH="$PWD/current_work/continual-learning-bench"
export PYTHONPATH="$PWD:$TTCL_BENCH:$PWD/ttcl/.runtime/structured_memory_deps:$PWD/ttcl/.runtime/deltamem_benchmark_deps"
"$TTCL_PYTHON" -m unittest ttcl.memrl_comparison.test_protocol -v
"$TTCL_PYTHON" -m ttcl.memrl_comparison.run calibrate \
  --output "$PWD/results/memrl_preflight/new_calibration.json"
"$TTCL_PYTHON" -m ttcl.memrl_comparison.run prepare \
  --root "$PWD/results/memrl_comparison/new_run" \
  --calibration "$PWD/results/memrl_preflight/new_calibration.json" --gpus 6 0 --port 18527
"$TTCL_PYTHON" -m ttcl.memrl_comparison.run supervise \
  --root "$PWD/results/memrl_comparison/new_run"
```

Run `supervise` under a durable process manager. It waits for three consecutive
free-GPU observations, starts its own server, validates frozen inputs and performs
a real training-only / CL calibration-prefix smoke test before evaluating.
It never stops existing GPU jobs. `status.json` distinguishes waiting, preflight,
running, finished-with-failures and complete; `summary.json` contains current
paired scores. `run status --root ...` refreshes the summary on demand.

## CLBench public-state improvement experiment

`improved_cl.py` is an optional task-aware extension to the frozen MemRL
adaptation. It leaves the original memory store, writer, Q update and official
task scorer intact. It adds only information already available to the agent:

- **Blind Spectrum Monitoring:** cluster peaks from previous public scans,
  require recurrence, and add credible dormant channels to the final report.
  The report still contains the actor's current detections; the final action
  and all added peaks are recorded in `policy_action.json`.
- **Cohort Studies:** retain prior raw 108-field submissions and submit their
  per-field running mean on later studies. This reduces dependence on any one
  study's selection bias. A compact record of earlier public group-survival
  tool measurements is also exposed; native retrieval can search across
  studies without the original same-stage-only similarity cutoff.

State is updated only after each instance from its public input, tool feedback
or raw actor submission. Hidden grader values are not used to construct state
or final actions. The original MemRL Q updater still receives the official
reward as in the frozen comparison, so this is a controlled method extension,
not a score-hidden CLBench submission. Because the extension was designed
after inspecting the original CLBench run, the same 90/20 instances are a
development validation, not an untouched final test.

With the original frozen plan and a compatible frozen actor server running,
evaluate each domain/repeat into a new directory:

```bash
python -m ttcl.memrl_comparison.evaluate_improved_cl \
  --origin "$PWD/results/memrl_comparison/20260928_budgeted" \
  --output "$PWD/results/memrl_improved_cl/new_bsm_303" \
  --task blind_spectrum_monitoring --repeat 303 \
  --url http://127.0.0.1:18537 --temperature .7
```

The runner freezes its design, source files, input hashes, memory snapshots,
actor completions and final actions. Run the analogous commands for repeat
404 and `cohort_studies`, then use `analyze_improved_cl.py` with `--root` and
`--output` to compute paired last-80% scores. The two `offline_*_check.py`
scripts isolate the public-state mechanisms with the official scorer; they
do not replace the online actor evaluation.

### Completed development validation, 2026-10-01

Both repeats completed all scheduled cells. Scores below are paired on each
domain's last 80%, with unscored cells excluded from both arms, never filled
with zero. The actor temperature is 0.7, as in the original MemRL CL run.

| Domain | Scored pairs | Vanilla MemRL | Public-state extension | Difference | Wins/losses/ties | Task-cluster bootstrap 95% interval for difference |
|---|---:|---:|---:|---:|---:|---:|
| Blind Spectrum Monitoring | 144/144 | 0.218928 | 0.545759 | +0.326831 | 144/0/0 | +0.307222 to +0.345653 |
| Cohort Studies | 30/32 | -0.000754 | 0.083451 | +0.084205 | 29/1/0 | +0.050773 to +0.120646 |

All 360 BSM cells scored. Cohort recorded 80/80 cells; two schema-invalid
actions in repeat 404 (one in each policy, on different studies) left two
unscored pairs. The mean is computed over the 30 shared scored pairs. The
bootstrap resamples canonical instance indices while keeping both repeats
together; the interval is descriptive because the method was developed after
inspecting this benchmark.

The saved final actions were independently re-scored against the official
task scorers. In BSM, the actor's raw report averaged 0.245658 and the final
report 0.545759; adding historical channels produced the large gain. In
Cohort, the actor's raw submission averaged -0.003417 and the running-mean
submission 0.083451; the numerical aggregation, rather than simply showing
the agent more text, produced the gain. These checks and the per-stage tables
are saved under the ignored local `results/memrl_improved_cl/` directory.

## Cross-benchmark MemRL candidate

See [GENERAL_METHOD.md](GENERAL_METHOD.md) for the causal-utility research
direction, the frozen contextual-gate implementation, paired online results,
and the disjoint-game regression that rejects that gate as a general method.
`credit_probe.py` supports paired leave-one-out replay of individual memories.
See [PAIRED_CREDIT_RL.md](PAIRED_CREDIT_RL.md) for the source-bound CPU
credit learner and its negative held-out diagnostic; it has not produced
a useful trained selector. The same document reports train-only,
same-service full-versus-empty, evidence-framing, and retry probes.

## Generic structured-evidence extension

`structured_evidence.py` is a separate cross-benchmark candidate. Each new
MemRL item gets a `structured_evidence` metadata field containing bounded
public action/observation events, the last public action, and a hash of the
source trajectory. Free-form reasoning fields are omitted. The native writer,
embedding/Q ranking and Q update remain in place. Retrieval softens the fixed
absolute similarity cutoff, then packs the selected memory's short lesson and
structured public evidence into the same 2,048-token budget. The same code is
used for ALFWorld and every available CLBench domain; it has no task-name or
domain-specific action rewrite. Earlier `improved_cl.py`,
`general_evidence.py`, and `contextual_utility.py` remain available for their
documented comparisons.

`evaluate_structured_evidence.py` runs an immutable paired chain. The full
queue entrypoint is `run_structured_full.py`; it schedules all six ALFWorld
families × three seeds and four CLBench domains × two repeats from the frozen
MemRL plan. Each cell independently builds memory from empty using the same
actor seeds in both arms. Its `summary.json` reports ALFWorld first-attempt and
within-three success, and each CLBench domain on the prespecified last 80%,
using only common valid pairs. `status.json` and per-job logs retain failures.
The full run takes substantially longer than the two-episode smoke checks;
scores should be quoted only after the queue reaches a terminal state.

## One general CLBench memory-update candidate

`universal_evidence.py` is a single method applied without task-name, action
schema, or final-action branches. Every CLBench episode contributes public
query/action/feedback events bound to hashes. One frozen model prompt proposes
`add`, `revise`, or `retire` operations on a cumulative evidence notebook;
the same executor checks new-event citations and stable entry IDs in every
domain. Invalid operations are recorded and rejected individually, and
complete operations in a capped response can still be checked. Untouched
entries persist. Retrieval ranks notebook claims with the same embedder for
every query, then fits claims and native MemRL `Task` and `Experience` text in
the declared 2,048 memory-token budget. The actor's final action is not
rewritten. Native MemRL
writer and Q updating remain in place, with official reward excluded from the
notebook prompt and used only by native MemRL. Extra notebook model tokens are
recorded separately per episode.

This is a research candidate, not a validated improvement. Short paired
pilots live under ignored `results/memrl_universal_evidence/20261002_dev/`.
They verify that the same update path executes on all four CLBench domains;
the tiny prefixes do not measure the prespecified last-80% outcome. The
initial full queue was stopped when its actor context was found to omit the
native `Task` text; its records remain frozen, and the corrected method uses
a separate run directory. The
schema-specific `cl_public_state` attempt was stopped and removed from code
after it was identified as a collection of task-specific rules, rather than
this general method. Its ignored result records remain marked stopped.

## Source-grounded cross-benchmark candidate

See [GROUNDED_EVIDENCE.md](GROUNDED_EVIDENCE.md) for a separate, task-name-free
candidate that stores exact public events, distinguishes unverified submissions
from environment responses, and uses one bounded retrieval rule in ALFWorld and
CLBench. Its schema-valid direct-action review is an extra CLBench interface
step. The frozen `v21` implementation completed two online repeats in all
four CLBench domains, but its two complete ALFWorld families regressed,
so it is not a general improvement. `v23` removes the unsupported
unconditional third-attempt text-memory dropout while retaining the
structured-action projection. It has only a small official-train wiring
check and needs independent online evaluation before any benchmark claim.
The separate `v24` candidate tests native/empty/native context across
retryable text-command attempts under the existing three-attempt budget;
its six-family official-train pilot finished at 9 wins, 2 losses and
25 ties. The first complete valid-unseen family then regressed (simple
pick-and-place: 2 wins, 4 losses, 18 ties), so the frozen run stopped
with missing tasks preserved. v24 is not a general ALFWorld improvement.
`v27` restores native retrieval for text retries while retaining the
v26 structured-action projection; it removes the known regression but
has no demonstrated ALFWorld gain.
