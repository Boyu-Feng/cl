# MemRL: matched ALFWorld / CLBench adaptation

Official repository: https://github.com/MemTensor/MemRL, commit
`c1b322ca43de36ddf64c6712f89d0095bfc35ce0` (MIT). The full downloaded
source is in `current_work/MemRL`; downloaded datasets are ignored local assets.

This experiment measures the method under the existing Qwen3-4B protocol. It
does **not** reproduce the paper's original models or 3553-task × 10-epoch
training budget. All actor weights remain frozen; Q values in episodic memory
are learned online. No pretrained memory bank or historical labels are reused.

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
