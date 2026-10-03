# Paired-credit MemRL: training-only development

The earlier experience-writer RL run used the same future task with and
without the generated document as its reward difference. Its ALFWorld
training produced only 14 nonzero advantages in 96 writer actions (4 positive,
10 negative), and the final writer did not beat its untrained comparison on
the 48 held-out post-first tasks. Those facts are recorded in `EXPERIMENTS.md`.
They motivate better signal acquisition and attribution; they do not show that
paired reward differences are useless.

## Implemented CPU learner

`paired_credit_rl.py` learns a *retrieval* contextual bandit from new,
source-bound probes. It does not train the 4B experience writer. For a fixed
task, memory snapshot, and actor seed, the target for memory `m_i` is

```
delta_i = R(task, full retrieved set) - R(task, full set without m_i).
```

The same leave-one-out target is used for ALFWorld train games and CLBench
calibration-prefix instances. ALFWorld has three actor sampling seeds per
selected memory; CLBench also has three. On two-memory CLBench cases, all four
coalitions are saved, so exact two-player Shapley values and interaction can
be reported separately. The prediction target remains leave-one-out because
the current ALFWorld probes do not have every coalition.

The learner audits source split, input content, snapshot, memory text, actual
actor context, actor seed, and official reward before building a label. It
groups sampling replicas by the complete source/content binding rather than
by memory ID. A robust scale is computed from each training domain using the
same formula; scaled deltas are clipped to [-1, 1] so a large Poker return
cannot dominate the fitted policy. A shared ridge model uses only pre-action
numeric memory and retrieval features. It contains no benchmark-name or
task-family feature. Bootstrap units are public input content, with one actor
seed sampled within each unit; repeated CLBench source chains sharing a task
are held out together. This estimates how uncertain its prediction is.

`paired_credit_policy.py` implements the inference rule: it would change
retrieval only if the 95th percentile of the bootstrapped marginal estimate
is below -0.05. It removes at most one memory per task: the labels do not
identify the result of removing several at once. It otherwise preserves
native MemRL retrieval. This is a
conservative experimental rule, not a claim of calibrated 95% coverage or
online improvement. Input-content-held-out cross-fitting is a diagnostic; independent
online memory chains are required for the benchmark claim.

## Current bound inputs and result

The ALFWorld source chain contains 36 audited official `train` games, six per
family. The outcome-blind frozen selection has 12 cases with at least two
retrieved memories. The first-position probe has 36/36 audited paired actor
replays: 3 positive, 1 negative, and 32 tied marginal differences. The
CLBench repeat-404 calibration-prefix probe has 27/27 audited cases and 60
official-result branches; 24 Spectrum/Cohort final actions were independently
re-scored. Those are training/calibration labels, not scored suffix results.

All 31 selected ALFWorld memory positions have now been probed with three
actor seeds: 93 audited differences, 8 positive, 2 negative and 83 tied.
The CLBench repeat-303 prefix probe also finished: 21/21 comparisons,
48 runner-audited branches, and 12 independently re-scored Spectrum final
actions. It has no Cohort memory source in its first 20% prefix; no label was
invented. A second ALFWorld train wave has been frozen by the same
reward-blind eligibility rule for all ten remaining source games with at
least two first-attempt memories. It completed 30/30, 30/30, and 27/27
audited paired replays for memory positions 0, 1, and 2.

One fixed-snapshot Poker-404 prefix case makes the broadcast-credit issue
concrete. At actor seed 92752, the full two-memory return was -9 chips;
removing memory 0 made it -14.5, while removing memory 1 made it -1.
Their leave-one-out effects were therefore +5.5 and -8 chips on the same
task. A native update from the full return would send the same -9 reward
to both entries. Other actor seeds for this case differ, so this illustrates
possible misattribution rather than a stable per-memory ranking.

The CPU model fitted to both ALFWorld waves and both CLBench prefix probes
has 78 source-bound memory examples from 38 source cases and 31 distinct
public inputs; 234 actor-seed-level differences are audited. Its
input-content-held-out diagnostic removed **zero** memories. Mean-squared
prediction error was 0.0879 versus 0.0826 for always predicting zero on
these bounded training labels. The trained selector therefore supplies no
demonstrated improvement over native retrieval. The smaller provisional
first-wave model also removed zero memories; its apparent error reduction
did not survive the additional outcome-blind ALFWorld cases.

The original 78-example policy was subsequently tested without refitting on
six new content-hash-selected Poker calibration-prefix inputs, each in repeat
303 and 404 with three actor seeds. All 36 source-chain/seed cells and 25
memory examples passed source, context, actor-seed and official-reward audits.
Using the original training reward scale, its held-out MSE was 0.130437
versus 0.151372 for predicting zero, but the frozen confidence rule still
removed **zero** entries. The two repeats share six public inputs and source
lineage; these are neither 12 independent tasks nor an online-policy gain.
See `CREDIT_ASSIGNMENT_NEXT.md` for the full audit and interaction results.

A simple historical-success filter is also unsupported by these labels.
Among the 60 bound ALFWorld memory examples, 47 came from unsuccessful
source episodes and had mixed marginal signs (eight positive, seven
negative); 13 came from successful sources and had no positive marginal
labels and one negative label. These small, selected samples do not show
that failed memories should always be kept or successful ones removed;
they rule out using the source outcome as a reliable credit label by itself.

`probe_full_context.py` separately audits the set-level intervention
`R(full retrieved text) - R(empty context)` on the same official ALFWorld
train games, fixed snapshots, and actor seeds. It reuses the already frozen
full-context branch of the index-0 probe and samples only an empty-context
branch. This avoids the confounding in the older contextual-utility pilot,
which compared independently evolving native and no-memory chains at each
game and then treated those differences as if they measured a single memory
decision. All 66/66 branches passed the input and reward audit, but a
later service audit found an important limitation: the 36 first-wave
pairs used the same actor service for full and empty context, whereas
the 30 second-wave pairs reused a full branch from port 18557 and
sampled the empty branch on port 18559. `repair_full_context_service.py`
replayed those 30 unchanged full contexts on the empty branch's port
18559, with identical train game and actor seed. All 30 rewards matched
the archived branch; two complete action trajectories differed, but
their first actions and step counts did not. The repaired 66/66
same-service pairs yield 17 positive, 10 negative, and 39 tied
seed-level effects, mean +0.1061. A 10,000-draw input-clustered
bootstrap interval is [-0.1212,+0.3333], so the mean is uncertain.
The same-service first wave had 3 positive, 4 negative and 29 ties
(mean -0.0278); the repaired second wave had 14 positive, 6 negative
and 10 ties (mean +0.2667). Across both waves, the look family had
one positive, eight negative, and six ties on five selected inputs
(mean -0.4667). These heterogeneous selected cases do not support
a general set-level memory gate.

The previously frozen contextual gate regressed on disjoint `valid_seen`
look games (4/13 versus native 5/13 first-attempt successes), so its
task-pattern claim remains rejected. The repaired set-level probe has
27/66 nonzero seed-level labels, whereas the older writer run had
14/96 nonzero future-task advantages. Their selections and
interventions differ, so this is a signal-density comparison, not
evidence that a new writer would improve online reward.

This deletion learner acts only on memories that native MemRL already
retrieved. It cannot improve tasks whose retrieval is empty. In the selected
Spectrum calibration cases, all paired memory effects were zero despite
the full `v15` Spectrum chains improving strongly through their grounded
public-evidence action path. Thus individual-memory credit is one component
of a general method, not a replacement for evidence capture and use.

## Writer-training extension under consideration

The new CPU-only `counterfactual_writer_credit.py` now implements the
full-horizon branch scorer for this extension. It requires a complete
candidate/previous/empty grid for every predeclared target and at least two
actor seeds. Every branch must bind the completed source trace, source and
target input content, exact document hash, and common pre-target snapshot.
The scorer also requires an exact map from target-content hashes to new
reviewed annotation hashes; a future collector must verify and freeze those
review files before any rollout. Every branch row must carry the matching
reviewed annotation hash; a stale or differently reviewed target is rejected
instead of silently inheriting a label from an old history ID.
It rejects missing runs, nonfinite official outcomes, duplicate branches,
changed snapshots, and a source reused as its own target. Candidate versus
previous is the writer gradient label; candidate versus empty is reported
separately. A fixed, training-derived reward scale and optional predeclared
actor-call penalty put different official reward units on a comparable scale.
Identical candidate and previous text receives zero credit even if stochastic
actor calls produce different outcomes. The report retains individual
target/seed differences and conflicting seed signs; it does not fabricate a
positive label from a tie. This is code-only infrastructure: it has not
collected new targets, updated the writer, or produced a benchmark result.

The code now also has `score_writer_online_branches` for the harder causal
question exposed by v24 and v29. It starts candidate, previous and empty
arms from one frozen root bank, then lets each arm update its own memory
through an identical ordered sequence of reviewed future tasks. Its credit
is the paired, optionally discounted **whole-chain** return, including
later effects of changed trajectories, writer output, retrieval and Q values.
It checks within-arm snapshot continuity at every task; later branch
snapshots are expected to differ across arms. This differs from the
fixed-snapshot scorer, which estimates immediate target effects only. Both
methods require complete official outcomes and at least two paired chain
seeds. The online report separates the first-target contribution from
discounted later-target contributions, while the writer gradient uses
their combined whole-chain estimate. Neither is trained or empirically
validated yet, and a collector
must freeze candidate documents and reviewed targets before rollout.
`bind_online_writer_sample` checks the generated text, exact writer input,
source trace and writer freeze against this online report before attaching
an advantage to the existing PPO-style writer sample. A consistent
nonzero effect across chain seeds can drive the policy term. Opposed or
single-seed effects, ties and unchanged documents receive zero policy
advantage while retaining the diagnostic estimate and the ordinary KL
term. This prevents stochastic actor differences between identical
documents from becoming writer-improvement labels.

A later writer LoRA experiment can use the *same paired principle* while
rewarding changes to the previous experience document. Each candidate must
be generated from a completed source trajectory before any target task is
seen; compare candidate, previous document, and empty memory on the same
training task and actor seeds. Candidate-versus-previous measures what the
new extraction added; candidate-versus-empty reports absolute usefulness.
Only newly generated, reviewed, content-bound trajectories can enter that
training run. Prior `experience_design` already implemented these two
baselines separately; a new run should first show more nonzero, replicable
signal before spending GPU time on another LoRA update. No new 4B writer
training has started while the available cards are occupied by existing
experiments and diagnostic actor services.

The full success criterion remains one frozen policy with an independent,
paired ALFWorld evaluation across all six families and three seeds, and a
CLBench evaluation across all four domains and two repeats. Static deletion
labels and cross-fitting cannot substitute for those online results.

## Next train-only interventions (2026-10-03)

`probe_evidence_frame.py` freezes 66 actor-seed pairs on the same 22
outcome-blind selected ALFWorld `train` inputs used by the audited
full-versus-empty probe. It adds one interface-neutral rule to the
historical text: treat it as evidence from another instance, verify
objects, locations, numbers and conclusions against current observations,
and ignore conflicting details. The current task and historical text are
unchanged; the longest framed context is 1,992 of the allowed 2,048
tokens. The probe compares the new branch with the frozen full-context
branch at the same game and actor seed. The archived full branches were
generated on different actor services, so
`probe_evidence_frame_control.py` replayed all 66 unchanged full
contexts on the framed branch's actor service. All 66/66 pairs passed
the source, game, seed and reward audit. Relative to this same-service
control, framing won 2, lost 10 and tied 54 actor-seed pairs, mean
-0.1212. The 10,000-draw input-clustered bootstrap interval is
[-0.2576,0.0000], and five of the six selected families had a negative
mean; the remaining one was positive. Eight same-seed full
replays differed in reward from their archived branches, confirming
why the direct archived comparison should not be used. The general
framing rule is **rejected** for promotion to the online policy.

`probe_retry_context.py` separately freezes every official `train` game
in the existing native source that reached a third attempt with nonempty
retrieval: 22 games across five families. It pairs full versus empty
third-attempt memory on the same game and actor seed for three new actor
seeds, with no online Q update. All 66/66 pairs passed the official-train
source and reward audit: using the third-attempt memory won eight, lost
nine and tied 49, mean -0.0152, with an input-clustered bootstrap
interval [-0.1970,+0.1818]. The look family had nine losses and no
wins in 15 pairs, whereas the cool-and-place family had eight wins
and no losses in 18 pairs. Thus fixed third-attempt dropout protects
some inputs but harms others; its net effect is uncertain. It does not
identify which individual memory helped or establish an online-chain
effect. Both probes bind the input content, retrieval, source trajectory
and actor seed before generating new trajectories. Neither starts LoRA
training or alters other GPU experiments.

`grounded_evidence_v23.py` removes v21's unconditional third-attempt
dropout for the general `retryable_text_command` interface and uses
native MemRL retrieval on every text attempt. The structured JSON
projection is unchanged. This is a conservative code correction, not a
demonstrated ALFWorld improvement: the paired retry data show why the
old fixed schedule is unjustified, but do not identify a better
conditional gate. A small official-train online wiring check is separate
from the required six-family, three-seed evaluation. Its two selected
look-family train games completed all four native/candidate cells.
Both arms had identical retrieval IDs and context SHA-256 on all six
attempts, identical 0/2 rewards and 150 actor calls per game; the
controlled actor client reported 306 response-cache hits. This verifies
the text retry connection for those two train games, not an online
improvement.

`probe_second_context.py` freezes all 27 official-train source games
that reached a second attempt with nonempty retrieval, across all six
families. It compares second-attempt full and empty context on the same
game and actor seed for three actor seeds, with input and source hashes
bound before replay. All 81/81 pairs passed the audit. Full context
won 14, lost 10 and tied 57 versus empty context, mean +0.0494;
the 10,000-draw input-clustered interval is [-0.1481,+0.2346].
The selected look games favored empty context (one full-context win,
nine losses), while simple and clean games favored full context
(four and six wins, respectively, no losses). A fixed second-attempt
dropout therefore also has mixed effects; its total is uncertain.

`grounded_evidence_v24.py` is a separate, budget-neutral online
candidate: native retrieval on text attempt one, empty context on
attempt two, and native retrieval on attempt three. The second attempt
receives no Q credit for memories not shown; the native writer still
extracts from its new trajectory. The native similarity list still
indexes that new memory, so this is an actor-context intervention,
not a complete removal of memory-system state. Every retryable text-command task
uses this schedule, while JSON-schema tasks keep the v21 typed
projection. In the first two official-train look games, the candidate
succeeded on attempt two in both cases and native MemRL failed all
three attempts in both. These are tiny development results, and the
online histories already diverged by game two. The frozen six-family,
36-game official-train pilot has now completed and passed input/reward
audits: nine candidate wins, two losses and 25 ties, mean official
success difference +0.1944, with 636 fewer actor calls in total. A
game bootstrap interval is [+0.0278,+0.3611], but it does not measure
variation across actor seeds or online memory chains. Look had five
wins and cool had three; simple and clean tied throughout; heat had
one loss; pick-two had one win and one loss. In the heat loss, native
retrieved the just-written first-attempt memory and succeeded on
attempt two while the dropout candidate failed all three attempts.
In the pick-two loss, native succeeded on attempt three after both
policies had failed attempt two; the dropout changed the intermediate
trajectory, writer output and later retrieval. Those cases reject a
per-family guarantee even on train. The rule and six-family official
`valid_unseen` evaluation were frozen before inspecting any new
valid-unseen outcome. The first complete family of that repeat-92601
evaluation, simple pick-and-place, then regressed: 2 wins, 4 losses
and 18 ties across all 24 paired unseen games, mean success difference
-0.0833. The rule's predeclared all-family objective is therefore
already false. The remaining run was stopped after that whole family;
one subsequent look pair had completed. The 25/134 completed pairs,
109 missing pairs, frozen source hashes and stop reason are saved in
`stopped_early.json` and `analysis_partial.json`. Missing pairs are
unscored, never zeros. The v24 rule is rejected as a general online
ALFWorld improvement despite its positive train aggregate.

Among the nine train wins, four happened directly on the candidate's
second attempt, four had candidate first-attempt success after the
online memory chains diverged, and one arose only on a later attempt.
Of the two losses, one was direct on attempt two and one appeared only
on attempt three. Thus most paired final-task differences cannot be
attributed to the immediately suppressed retrieval alone.

The pick-two counterexample also narrows the proposed RL target. An
attempt-local difference `R_t(with memory) - R_t(without memory)` would
label both second attempts equally unsuccessful, while their subsequent
writer states and third-attempt outcomes differed. A future writer or
retriever policy needs a *branch-level* target over the remaining retry
budget, such as the paired difference in final task reward minus a fixed
actor-call cost. Both branches must start from the same memory snapshot,
public input and actor seed, then keep their own generated memories and
Q updates. This is an algorithm design consequence of the audited
trajectory, not a trained or validated new RL policy.

A concrete next writer objective is to generate a candidate experience
`e_new` from a completed, content-bound source trajectory before the
target task is sampled. On each training-only target and matched actor
seed, freeze the starting memory state and run three branches to task
completion: `e_new`, the prior writer's `e_old`, and empty context.
Define the writer advantage as the mean across seeds of
`(R_new - R_old) - lambda * (calls_new - calls_old)`, where `lambda`
is fixed before evaluation. Report `R_new - R_empty` separately to
detect a candidate that merely beats a worse writer. If the branches
continue into later tasks, they must retain separate memory and Q
states, and the horizon must be frozen in advance. Only source/input
bindings with reviewed targets may enter a future writer update;
zero or sign-unstable differences remain recorded but provide no
invented positive label. No writer weights are being trained by this
design while the diagnostic signal remains sparse.
