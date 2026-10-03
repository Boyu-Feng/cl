# Toward a cross-benchmark MemRL improvement

## Problem and evidence

Native MemRL assigns the same episode reward to every injected memory, ranks
them with a fixed query-similarity gate, and exposes long generated reflections.
In the local comparison, CLBench Cohort injected no memory in its scored suffix;
Blind Spectrum Monitoring injected text but did not convert history into final
actions. ALFWorld showed both positive transfer in five task families and
negative transfer in `look_at_obj_in_light`. A first-only-Q ablation had mixed
directions across ALFWorld seeds. Thus neither indiscriminate credit nor
always disabling memories is an adequate cross-benchmark solution.

In the original three-seed first-attempt retrieval records, ALFWorld often
injects three memories: `look_at_obj_in_light` 44/54 cases and clean 86/93.
By contrast, CLBench Cohort retrieves none in 36/40 repeat cases and Spectrum
never retrieves more than one (52 zero, 128 one). Individual-memory credit
assignment is therefore relevant mainly to ALFWorld and Poker in these runs;
it cannot by itself repair Spectrum or Cohort.

## Research candidate: evidence plus causal utility

Store two linked parts for each experience: (1) public observations, actions,
and provenance in a lossless event record, and (2) a short conditional lesson.
Retrieve candidate experiences with soft ranking, then select only those whose
estimated **incremental utility** is positive under the actual token budget.
On training/development episodes, sample one candidate for a paired
leave-one-out rollout with the same task and actor seed:

`delta_i = R(x, M) - R(x, M without m_i)`.

Update only the tested candidate's utility from `delta_i`; aggregate across
tasks and actor seeds rather than trusting one noisy comparison. The actor sees
compact evidence with provenance and a conditional lesson, not a complete old
task and unverified reflection. Persist public state across episodes so
otherwise useful observations are not lost when the generated lesson omits
numbers or entities. At evaluation, freeze learned utility and do not query a
hidden scorer. This is a proposed method, **not an implemented or validated
result**. Its extra training rollouts and token cost must be reported.

## Current exploratory implementation

`general_evidence.py` is a narrower pilot. It uses native MemRL storage/Q,
softens the absolute retrieval cutoff, injects at most one short, nonfailed
lesson, and has two fixed public-action operators: averaging a long numeric
map and retaining recurrent records with frequency-like coordinates. The
operator library is schema-aware; avoiding benchmark names alone does not make
this a complete general algorithm. Its online run is under ignored
`results/memrl_general_evidence/`. Do not cite its scores as cross-benchmark
improvement unless the same frozen policy passes both full paired evaluations.

The completed one-seed pilots show why the rule is insufficient. On CLBench
Spectrum episodes 19–30 it scores 0.436300 versus vanilla 0.210542 (12/12
paired wins); on Cohort episodes 5–10 it scores 0.021253 versus -0.010495 on
five jointly scorable pairs. On ALFWorld `look_at_obj_in_light` seed 92601,
first-attempt success is 9/18 versus vanilla 5/18. But on ALFWorld
`pick_clean_then_place_in_recep`, vanilla succeeds on 3/5 first attempts and
the fixed-rule pilot on 0/5. The latter run was stopped early as a failed
candidate; it is not a complete benchmark comparison. Official CL scorers
reproduced all 40 saved improved final-action rewards in the two CL pilots.
These are development diagnostics, and the fixed-rule pilot is **rejected** as
the cross-benchmark solution.

## Decision rule for the next experiment

Compare one frozen candidate with native MemRL under identical actor, task
order, decoding, and memory budget. For ALFWorld report first-attempt and
within-three success by all six families and three seeds; for CLBench report
all four domains, both repeats, and the prespecified last-80% metric.
Preserve failed cells and score only common valid pairs. Include no-memory,
latency, tokens, and added training rollouts. If only one family or domain
improves, report a narrow mechanism rather than a general improvement.

## Conservative contextual utility gate (development experiment)

`train_contextual_utility.py` fits one ridge model to paired original
MemRL/no-memory outcomes from ALFWorld seed 92601 and CLBench repeat 303. Its
inputs are query words plus six quantities available before acting: retrieved
count, context length, success/failure fractions, mean Q, and writer truncation
fraction. The target is the clipped *set-level* reward difference. The fixed
decision threshold is -0.2, so native context is withheld only when strongly
predicted to be harmful. `contextual_utility.py` keeps native MemRL retrieval,
writing and Q updates when it injects. It also inherits the schema-aware public
action projection from `general_evidence.py` for structured CLBench actions.

This is a runnable screening baseline, **not** the proposed per-memory
leave-one-out utility learner: it has no individual-memory credit estimate.
The offline potential-outcome splice used to choose the gate is exploratory and
cannot establish an online gain. Evaluation seed 92602 reuses ALFWorld games
from the training seed; CLBench repeat 404 reuses canonical task instances.
Thus even a positive online pilot is development evidence, not an untouched
test-set result. Runs live under ignored `results/memrl_general_evidence/` and
freeze the gate, source files, plan, input bindings and independent memory
chains.

The first online guardrail pilot, ALFWorld clean seed 92602 first six games,
has completed: both arms succeed on 1/6 first attempts and 3/6 within three
attempts. The gate injected native context on all 16 improved-arm attempts,
so the chains and outcomes remained identical. This establishes preservation
when the gate does not fire, not improvement on that family.

The complete `look_at_obj_in_light` seed-92602 online chain has 18/18 paired
games and no failed cells. First-attempt success is 8/18 with the gate versus
2/18 native MemRL (7 paired wins, 1 loss); within three attempts it is 9/18
versus 5/18 (5 wins, 1 loss). The gate suppressed 18 of 38 improved-arm
retrievals. Seed 92603 also completed: first-attempt success 7/18 versus
3/18 native (5 paired wins, 1 loss); within-three success 9/18 versus 7/18
(4 wins, 2 losses). Its gate suppressed 20 of 40 retrievals. Across both
seeds, first-attempt success is 15/36 versus 5/36 and within-three is 18/36
versus 12/36. These seeds reuse games in training, and the threshold was
explored on held-out outcomes offline, so this remains development evidence.

The full `pick_two_obj_and_place` seed-92602 guardrail has also completed:
both arms achieve 8/17 first-attempt and 14/17 within-three success, all 17
games tied, with no gate suppression in 33 improved-arm retrievals. The gate
preserved native MemRL on this positively transferring family.

Two CLBench repeat-404 prefix pilots have completed, using the original
last-80%-of-domain scoring threshold. In Spectrum episodes 19–30, the
contextual-utility arm averages 0.436300 versus native MemRL 0.216242
(12/12 paired wins). In Cohort episodes 5–10, it averages 0.019382 versus
-0.016447 (6/6 wins). All 80 cells completed, and independent official
scorer replay matched all 40 final actions. The gate never suppressed a CL
memory in these pilots; the improvements are attributable to the typed public
action projection. These are still short, development-only chains, not full
domain runs.

The other two CLBench repeat-404 prefix guardrails also completed without
failed cells. Database episodes 7–10 scored 0.0 in both arms and no gate or
action operator fired. Poker episodes 25–30 averaged 0.416667 in both arms;
the gate suppressed six earlier memory contexts, but that did not change the
six scored outcomes. Thus the short-chain evidence shows gains in Spectrum and
Cohort, ties in Database and Poker. It does not establish full-domain gains.

A leave-one-family-out diagnostic on the training records found that excluding
`look_at_obj_in_light` from gate fitting yields zero suppressions on its 18
development episodes. The gate therefore needs task-family calibration to
identify this negative-transfer pattern. It is not zero-shot family-general.

### Paired online pilot table

All rows use empty-to-online memory chains and identical actor seeds within
each pair. CLBench rows use the official last-80%-of-domain threshold applied
to the short prefix, while ALFWorld rows show full-family totals except the
six-game clean guardrail. No pair failed.

| Benchmark / task | Evaluated tasks | Native MemRL | Contextual utility | Mechanism |
| --- | ---: | ---: | ---: | --- |
| ALFWorld look, seed 92602, first attempt | 18 | 2/18 | 8/18 | Gate: 18 suppressions |
| ALFWorld look, seed 92603, first attempt | 18 | 3/18 | 7/18 | Gate: 20 suppressions |
| ALFWorld pick-two, seed 92602, first attempt | 17 | 8/17 | 8/17 | No gate firing |
| ALFWorld clean, seed 92602, first 6 games | 6 | 1/6 | 1/6 | No gate firing |
| ALFWorld look, disjoint valid_seen, first attempt | 13 | 5/13 | 4/13 | Gate: 14 suppressions; regressed |
| ALFWorld pick-two, disjoint valid_seen, first attempt | 24 | 9/24 | 8/24 | Gate: 2 suppressions; regressed |
| CLBench Spectrum, repeat 404, episodes 19–30 | 12 | 0.216242 | 0.436300 | Typed action projection |
| CLBench Cohort, repeat 404, episodes 5–10 | 6 | -0.016447 | 0.019382 | Typed action projection |
| CLBench Database, repeat 404, episodes 7–10 | 4 | 0.000000 | 0.000000 | Neither component fired |
| CLBench Poker, repeat 404, episodes 25–30 | 6 | 0.416667 | 0.416667 | Gate fired earlier; scored tasks tied |

These pilots do not satisfy the full three-seed/six-family/two-repeat decision
rule above. The gate is set-level and calibrated on some of the same task
content; the public-action projection is schema-aware. The next iteration
needs paired per-memory deletion labels on disjoint training tasks, a utility
model that transfers beyond task-family keywords, and a genuinely unseen
evaluation split before a generality claim.

The disjoint-game ALFWorld `valid_seen` check has no game-file hash overlap
with the 20260928 `valid_unseen` evaluation plan. On all 13 look games, the same
frozen gate **regressed**: first-attempt success 4/13 versus native 5/13
(1 paired win, 2 losses), and within-three success 5/13 versus 7/13
(1 win, 3 losses). All 26 cells completed; the gate suppressed 14 of 30
retrievals. On games 6 and 8, native MemRL succeeded after retrieving
reflections related to desklamp tasks, while the gate suppressed some or all
of those contexts and failed. In the native retrievals at game 6 attempt 3
and game 8 attempts 1–2, every selected memory was marked `success=False`,
yet the full-memory chain succeeded within three attempts. This is consistent
with useful corrective failure reflections; because the online chains had
already diverged, it is not an isolated causal effect of a specific memory.
It shows why filtering by success or task-level average reward is unsafe.
The held-out paired regression is concrete evidence that the task-word
gate can remove useful memories. **Reject contextual_utility_v1 as the
cross-benchmark general method.** Its positive same-game seed results remain
development diagnostics, not evidence of task transfer. The 24-game unseen
pick-two guardrail also regressed: first-attempt 8/24 versus native 9/24
(1 paired win, 2 losses), and within-three 14/24 versus 17/24
(0 wins, 3 losses). Its gate suppressed only twice, both in game 9; the
independent online memory chains then diverged, illustrating how one early
decision can affect later tasks. All 48 cells completed. The CLBench default
Spectrum schedule uses a frozen corpus; changing its top-level task seed does
not generate new instances.

A small fixed-snapshot leave-one-out probe on a development game from the
canonical `valid_unseen` evaluation sequence,
`look_at_obj_in_light/92601/episode_012` illustrates the next label source.
With the same game and actor seed in a replay, both retrieved memories yield
first-attempt reward 1, while dropping `memory_000022` yields 0. Both stored
memories were marked `success=False`; the original online episode failed and
gave both negative Q updates. The replay conflicts with those shared updates,
but a single stochastic pair is noisy and must be repeated across games and
seeds before training a utility estimator. The probe artifacts are under
ignored `results/memrl_credit_diagnostic/20261001_paired_drop_train/`.
Despite that historical directory name, these are not train-split games and
must not become training labels for a purported untouched `valid_unseen` test.

The same fixed snapshot with actor seeds 92601, 92602 and 92603 gives
leave-one-out differences +1, 0 and 0 respectively. This one-memory example
still cannot support a general credit rule; it demonstrates why repeated
paired rollouts are needed. The multi-seed record is under ignored
`results/memrl_credit_diagnostic/20261001_paired_drop_multiseed_train/`.
