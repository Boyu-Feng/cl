# General hypernetwork transfer protocol

## Objective

Train one frozen trajectory-to-LoRA hypernetwork on a domain-independent
trajectory corpus, then evaluate it without updating hypernetwork parameters on
held-out XLand, ALFWorld, and HLE tasks. The transfer evaluation uses the same
base model, LoRA rank, action scoring, and `B` norm limits for every domain.

## Training corpus

### Audit correction

The initial `general_corpus_v99_20261010` artifact is NOT an eligible general
transfer corpus: its 412 rows are drawn from XLand, ALFWorld and HLE themselves.
Removing domain metadata does not remove domain exposure, and copying raw
source/query strings does not establish that benchmark names or labels are
absent from their contents. Its `answer` field is supervised training data;
earlier statements that the training inputs contain no answers were too broad.
Keep this artifact as a historical joint-domain preparation, not evidence for
unseen-dataset transfer.

The associated v99 training attempt was stopped and invalidated because
copying A under no_grad disconnects direct-LoRA A from its optimizer and
overwrites hypernetwork A on every step. The corrected mounting code keeps
separate Parameter objects; a gradient check confirmed independent A updates.
The old trainer is disabled pending a new corpus, reviewed bindings and fresh
run directory. No v99 checkpoint is eligible for evaluation.

For the next run, ALL training sources and supervision must exclude XLand,
ALFWorld and HLE, not only their held-out tasks. Domain-neutral formatting
alone is insufficient. Freeze source lineage and content checks before training.

The corpus is built from trajectories whose source tasks are disjoint from all
three transfer test sets. Each row is text-normalized as:

```text
goal; state_before; action; feedback; state_after; reward; done
```

The encoder never receives a dataset name, benchmark identifier, target query,
or expert answer. Action labels are retained only for the future-query loss;
failure rows are not assigned guessed corrective actions.

## Training objective

For each corpus episode, generate a bounded residual `Delta B` and measure the
action loss on a different future query from the same held-out task family:

```text
future-query cross entropy + 0.25 * first-query replay loss
             + effective-update norm penalty
```

The split is by task and source history, never by individual transitions.
The trained updater checkpoint, corpus manifest, source hashes, and target
bindings are frozen before transfer evaluation.

## Transfer arms

1. `base`: no adapter.
2. `frozen_general_hyper`: frozen general hypernetwork emits `B` from the
   current task trajectory.
3. `direct_lora`: direct LoRA trained with the same general corpus budget.
4. `random_hyper`: random frozen updater control.
5. `shuffled_trajectory`: frozen updater with state/action/feedback alignment
   destroyed while preserving row counts.
6. `reset_hyper`: generate each update from zero `B`, testing accumulation.

No arm updates hypernetwork weights during transfer. The only allowed online
operation is the declared accumulation of the generated `B` on completed
trajectories. A separate online-learning experiment must not be mixed with
this transfer result.

## Required reporting

Report first-attempt and cumulative success separately for each domain, plus
the same-task and cross-task transfer slices. Include source and target hash
audits, complete raw episode counts, and a permutation test against the direct
LoRA and shuffled controls. A gain is attributed to general experience only if
it survives the shuffled-trajectory and random-hyper controls and is present
on at least two domains.
