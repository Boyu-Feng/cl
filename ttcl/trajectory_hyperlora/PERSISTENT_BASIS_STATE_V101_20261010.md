## Superseding implementation audit — basis growth and trajectory credit

This audit supersedes the mechanism conclusions below for runs through v120.
`append_basis` extended the bank and coefficient vector but did not update `k`.
Mounted mixtures still iterated only over the original bases. New coefficients
could dilute the original mixture, but newly stored success/failure matrices
were not mounted. Thus v109–v116 scores cannot establish that added residual
content was used, nor rule out dilution or collinearity as limitations.

The last-four-action loss also paired its first selected action with the initial
observation instead of the preceding transition, and divided by weights from
the full trajectory. This affects direct-B and mounted-gradient comparisons.
Both bugs are fixed, with regression tests that check numerical contribution
of a newly appended basis and all selected observation/action pairs plus loss
normalization. All 14 controller/runner tests passed.

The v120 24-game run was interrupted intentionally after discovery; preserve
its partial output, and do not treat it as a completed or corrected comparison.
Earlier 2/12 versus 1/12 raw counts remain historical observations, not proof
of continual learning. A frozen identical-initialization adapter control is
still required to distinguish static adapter benefit from experience updates.
The repeatedly used 12-game slice is now development data. Earlier metadata-key
runs read game paths; these must not be presented as observation-only agents.
Further evaluation must bind the revised protocol and trajectory inputs anew.

# Persistent basis-state experience v101

## Audit correction (supersedes the earlier smoke-test interpretation)

Code inspection found that both XLand v101 smoke runners instantiate a new
`v94.Runner` without loading trained hypernetwork weights. Its B heads are
zero initialized, making the purported generated basis zero. The `direct`
LoRA factors are also never updated. v101c recreates memory per ruleset, so it
does not test cross-ruleset retention. A finite update loss or a new memory slot
therefore did not demonstrate any change to the effective policy parameters.
The reported 1/4 and 0/8 are environment outcomes, NOT valid comparisons of
online LoRA versus basis-state learning. These runs are ineligible for method
ranking. They also wrote v94 runner artifacts into its default output directory;
the original v94 initial.pt/source_bindings.json provenance requires re-audit.

The ALFWorld factor-bank run uses the older `ParameterMemory`, not the learned
coefficient controller. It reports 6/18 vs 4/18, but its script does not perform
independent environment replay or verify the target file hashes. Earlier claims
of replay verification for this run and the contextual-history baseline were
incorrect. Raw results remain preserved; these are provisional diagnostics.

The synthetic 24/24 measures repeated post-label fitting to a constant target,
not held-out next-task learning. The original coefficient update mixes basis
logits, which is not equivalent to the nonlinear actor with mixed B matrices.
`update_mounted` now differentiates the actor's actual loss through mounted B
and records the effective B change; this is still successful-action imitation,
not proof of improved future reward. Tests cover nonzero effective changes,
immutable actor/bases, degenerate zero bases, and rejection of nonfinite losses.

## Purpose

Keep the trained hypernetwork and its generated basis matrices frozen.  At
inference time, retain only a bounded key-addressed state of basis
coefficients and a write gate.  A successful action updates the coefficients;
a failure decays the gate and does not fabricate an action target.

## Implementation

- `basis_state_controller_v101.py` contains `BasisStateController` and
  `PersistentBasisMemory`.
- `PersistentBasisMemory` keeps separate coefficient states per normalized
  task key, interpolates states by cosine attention, and bounds the number of
  slots.
- `test_basis_state_controller_v101.py` verifies immutable bases and separate
  task slots.

## Checks

The controller unit suite passes (2 tests).  The small frozen-basis mechanism
check records 24 successful action updates: post-update accuracy is 24/24;
the output is in
`results/trajectory_hyperlora/basis_state_synthetic_v101_20261010/result.json`.

This is a mechanism check, not a benchmark score.  The fair external frozen
transfer v100 is complete and remains the current transfer evidence: HLE
3/24 for base, direct LoRA, and frozen hypernetwork; ALFWorld 2/12 base and
3/12 for the other arms; XLand 0/8 for every arm.  The next benchmark run
must compare direct online LoRA, frozen hypernetwork without state, and this
key-addressed persistent state from empty memory under the same task order.

## Real XLand smoke test

The one-ruleset pilot in
`results/trajectory_hyperlora/basis_state_xland_smoke_v101b_20261010/result.json`
used identical official environment seeds for all three arms and replayed
every trajectory. All arms received one positive episode out of four (1/4
each). The persistent arm created exactly one slot after the positive
transition, recorded a finite update loss, and performed no write on the
failure episodes. This verifies the reward-gated write path in the real
environment; the sample is too small to claim a performance gain.

A follow-up four-ruleset pilot (two shared-seed episodes per arm) completed
with official replay and zero runtime failures. All three arms obtained
0/8 positive episodes in that slice, so the memory remained empty and no
failure-only write occurred. This is a valid negative signal about the
available reward density, not evidence that the state mechanism improves
accuracy; the one-ruleset v101b run remains the positive-write path check.

## ALFWorld persistent factor-basis pilot

The 18-game reviewed train sequence completed with the frozen contextual
checkpoint and a capacity-eight success-only convex factor memory. The
paired base arm won 6/18; the persistent factor-basis arm won 4/18, with four
successful writes, one persistent-only success, and three base-only successes.
The result exactly matches the cumulative contextual-history pilot's 4/18
in aggregate, so there is no demonstrated gain from this first persistent
basis implementation. This arm stores generated factor bases and retrieves
them by task key; it does not yet learn coefficient credit from later rewards.

## Corrected mounted-B preflight

Using the existing trained external-hyper v100 checkpoint, four distinct
nonzero basis matrices were generated. On one held-out training row, the
actual Qwen loss through the weighted mounted B decreased from `0.0245313` to
`0.0244868`; the effective B changed by norm `0.2400`. A matched direct-B SGD
step also changed B. Hypernetwork/model parameters and all immutable bases
remained unchanged. This is valid evidence that coefficient learning reaches
the actual LoRA computation. It is a preflight on a training row, not a task
success or generalization result.

A 24-row real-Qwen sequence with the same frozen external checkpoint then
updated only the coefficient/gate state on each answer-confirmed row. The
actual mounted-B loss improved on 15/24 rows; mean loss changed from
`0.128711` to `0.124950`. This is evidence of a functioning cumulative
optimization path, not an environment success rate: rows are training-corpus
answers and the update uses their labels. It motivates, but does not replace,
the pending ALFWorld reward-gated evaluation.

## Corrected ALFWorld mounted-B pilot

The first six reviewed games were rerun with the trained external hypernetwork
loaded, four fixed nonzero bases, and actual reward-gated mounted-B updates.
Base, direct-B, and persistent-coefficient arms each scored **3/6**. The
persistent arm performed three real coefficient updates; each changed the
effective B (norm changes `0.2884`, `0.0683`, `0.0070`). Failures produced no
updates. No arm separated on this six-game slice, so this is a valid mechanism
and fairness check, not evidence of a gain.

The same corrected runner was expanded to the full reviewed 18-game sequence.
Base, direct online B, and persistent coefficients each scored **6/18**;
the persistent controller received six reward-gated mounted-B updates. There
were no persistent-only or direct-only successes in this run. This is a clean
null comparison: the corrected learning path is active, but it has not yet
improved ALFWorld success over direct B updates.

The full-trajectory-credit variant was expanded to the same 18-game sequence.
Base, direct B, and persistent coefficients each scored **6/18**, with six
reward-gated persistent updates. There were no arm-exclusive successes. Thus
using all successful actions improves the amount of parameter learning, but
still does not create a measurable ALFWorld success advantage in this slice.

## Full-trajectory credit variant

To address sparse credit, v103 uses every action in a reward-confirmed
successful episode with discount `0.9`, rather than only the first action.
On the first six reviewed games, base, direct B, and persistent coefficients
again scored **3/6**. The persistent arm made three updates with effective B
changes `1.8232`, `1.4008`, and `1.4101`; failures still made no updates.
This confirms that the denser credit signal reaches the actual B, but it does
not yet produce a success-rate gain.

## Next-task utility gate pilot

The next variant evaluates a candidate coefficient update on the following
reviewed task before committing it, and rolls back on a lower official reward.
On six games, all three arms scored 3/6. Two lookahead gates were evaluated:
one tie was accepted and one candidate improved the lookahead reward from 0
to 1. This demonstrates that state selection can use future-task utility, but
the pilot reuses the next task for selection and is therefore a development
diagnostic rather than an independent benchmark.

## Family-specific persistent states

To reduce cross-task interference, v106 uses one coefficient/gate controller
per ALFWorld task family while sharing the frozen basis implementation. Across
the full 18-game sequence, base, direct B, and family-specific persistent
coefficients each scored **6/18**. Six successful own trajectories expanded
their family basis banks; no arm-exclusive success appeared. Separating state
by family did not yet produce a measurable gain.

## Own-success basis expansion pilot

The controller now appends each newly generated successful own trajectory as
an immutable basis vector while retaining the coefficient state. In six
ALFWorld games, the basis bank grew from 4 external bases to 7 after three
successes. Base, direct B, and expanded persistent basis each scored **3/6**.
The added bases changed the actual mounted B, but did not yet change the
success count.

The gate was expanded to all 18 reviewed games. Base, direct B, and gated
persistent coefficients each scored **6/18**. Five lookahead decisions were
available; all five candidates were accepted (one improved the next-task
reward from 0 to 1, four were ties). No persistent-only or direct-only success
appeared. The utility gate prevents a measured regression in this protocol,
but has not yet produced a net success-rate gain.

An additional six-game check initialized each newly confirmed own basis with a
bounded positive coefficient logit (`0.5`) so it was not diluted by older
bases. Base, direct B, and this success-prioritized basis arm remained **3/6**.
The change fixes a real representation issue, but this sample provides no
performance improvement.

## Failure negative-evidence pilot

The failure branch was then changed to differentiate the actual mounted-B loss
of actions the agent executed on a failed episode, using bounded gradient
ascent on the persistent coefficients and a small gate decay. It does not
invent a correct action or reuse a future label. On the first six reviewed
valid-unseen games, base, direct B, and persistent state each scored **1/6**;
the persistent controller made one successful-trajectory update and five
failure negative-evidence updates. The mechanism is active, but this slice
shows no gain, so failure ascent alone is not sufficient evidence that the
state has learned a transferable correction.

The corrected negative-evidence run was extended to all 12 valid-unseen games.
Base scored **1/12**, direct online B **1/12**, and the persistent controller
**2/12**. This is a one-success lead over both controls, but the sample is too
small to establish a general improvement; two persistent coefficient updates
were committed.

## Failure residual-basis pilot

A new branch now converts the gradient of the observed failed-action loss into
a bounded residual basis (maximum 25% of the current effective-B norm), then
lets the persistent coefficient state decide its contribution. This preserves
the no-fabricated-label rule and makes a failure leave reusable parameter
content rather than only shrinking a gate. On the first six valid-unseen games,
base, direct B, and residual-basis persistence each scored **1/6**. Five
failure residual bases were added and one success update was committed, but no
exclusive success appeared. The representation path is active; its benefit is
not yet demonstrated.

The residual-basis branch was expanded to the full 12 valid-unseen games. It
scored **1/12** for base, **1/12** for direct online B, and **2/12** for
persistent residual-basis state. Ten failure residual bases were added and two
successful-trajectory updates were committed. The score matches the simpler
failure-negative-evidence branch on this split; the residual representation
therefore has not yet shown an additional gain, although it provides reusable
parameter content after failure.

A weight-sensitivity check initialized each failure residual basis with logit
`1.5`, so a newly observed failure direction was immediately favored over the
older bank. On the same 12 games the result remained **1/12** base, **1/12**
direct B, and **2/12** persistent residual state. The persistent-only success
was the same eleventh game as the zero-logit run. This rules out simple
initial-weight dilution as the main cause of the remaining gap.

## Key-addressed persistent memory pilot

A bounded key-addressed memory was added so each task can own coefficient and
failure state while sharing immutable bases. The first implementation exposed
that the initial observations collapse to one cosine slot; adding task metadata
to the key and allowing failures to allocate new slots fixes the routing logic.
On the six-game pilot with the corrected routing, base, direct B, and keyed
persistent state were all **1/6** and the embeddings still selected one slot.
Thus the memory API is now correct and tested, but this observation-derived key
is not discriminative enough to provide a benchmark gain. The family-keyed
controller and residual-basis results remain the stronger current evidence.

## Orthogonal failure-residual pilot

Failure residuals were projected away from the existing basis bank before
normalization and insertion, so each stored failure direction carries only
novel parameter content. The 12-game result was unchanged: base **1/12**,
direct B **1/12**, persistent state **2/12**. Ten orthogonal residual bases
were added and two success updates committed. The same persistent-only success
appeared as in the non-orthogonal run, so basis collinearity is not the main
remaining limitation on this split.

## Feature-conditioned basis router pilot

A small frozen-basis router was added. It maps the current task embedding to
mixture coefficients and updates only that map through the actual mounted-B
loss; failed episodes use bounded ascent on the actions actually executed.
On the complete 12-game valid-unseen sequence, base scored **1/12**, direct B
**1/12**, and the feature router **2/12**. It made two reward updates and the
persistent-only success was the same eleventh game seen in the simpler
controllers. Thus routing adds a clean parameterization of experience but does
not yet improve transfer on this sequence.

## Structured task-key router pilot

The router key was augmented with deterministic hashed metadata tokens from the
public task path (family, object, receptacle, and action type), alongside the
text embedding. On the same 12-game sequence, base scored **1/12**, direct B
**1/12**, and the structured-key router **2/12**. Two online router updates
were committed; the persistent-only success remained the same eleventh game.
The richer key therefore did not yet reveal a transferable routing advantage.

## Success-prototype retrieval pilot

A task-keyed bank now stores the actual B generated from each confirmed own
success. Later tasks retrieve the closest prototype instead of mixing all
successful bases globally; failures do not create labels or prototypes. On the
12-game valid-unseen sequence, base scored **1/12**, direct B **1/12**, and
prototype retrieval **2/12**. Two success prototypes were stored, but no
prototype-only success appeared. The result confirms the bank is active while
showing that this task order contains too few repeated successful prototypes to
establish transfer.

## v121 corrected growth preflight and frozen-control preparation

The real-Qwen external training-row preflight now isolates appended basis
content from coefficient dilution by comparing an actual appended basis with a
zero-content basis at identical coefficients and gate. The mounted matrix
content difference norm was 13.13205; actual-content loss was 0.0248802 versus
0.1433917 with zero content. This is a one-row mechanism check, not evidence of
cross-task learning. Actor/hyper parameters and source bases stayed unchanged.
The expanded regression suite passed 14/14.

The environment runner now includes a frozen-initial-B arm, preserves runner,
controller and review hashes plus CLI arguments, and removes game paths from
routing keys. Family parsing was also corrected for valid_unseen paths.
Lookahead selection is disabled pending rollback audit. No corrected environment
scores are available yet. Next: bind the revised protocol and new own-trajectory
inputs, then compare frozen B with online B on development tasks before using
an independent sequence. Direct B still starts at zero and learns only from
success, so it is not yet an equal-initialization/equal-feedback causal control.

## Corrected frozen-control pilot (v122)

After the growth and credit fixes, the same 12-game development sequence was
rerun with a frozen initial effective B, persistent family state, direct online
B, and base. Results were base **1/12**, frozen initial B **2/12**, persistent
state **2/12**, direct B **1/12**. The persistent-only success disappeared when
compared with the frozen arm: on this split its apparent gain was fully
explained by the initial generated adapter, not by online trajectory updates.
This is the first fair causal control and means no continual-learning gain is
currently established. The 12-game sequence must remain development-only.

## Corrected independent 36-game pilot (v123)

The revised runner was evaluated on the separately reviewed 36-game
valid-unseen slice with base, frozen initial B, family persistent state, and
direct online B. Scores were base **7/36**, frozen initial B **10/36**,
persistent **9/36**, and direct B **9/36**. Frozen initial B was best; the
persistent update and direct LoRA tied and both trailed frozen by one success.
The pairwise differences show that the earlier apparent persistent gain is not
a reliable continual-learning advantage. This is still a development slice, but
it is large enough to reject the current update rule as superior to its
identical-initialization frozen control.

## Initial-B anchoring pilot (v124)

The persistent coefficient update was regularized toward each family's frozen
initial effective B (`anchor_strength=10`). On the 12-game development slice,
base scored **1/12**, persistent anchored state **2/12**, direct B **1/12**,
and the frozen initial-B arm **2/12**. Anchoring removes the observed
regression relative to frozen B but does not create a gain over it. This is a
safety improvement, not evidence of learned transferable experience.

## Anchored success-delta pilot (v125)

A residual-only controller kept each family's frozen initial B fixed and added
only bounded `(B_success - B_anchor)` deltas from confirmed own successes. On
the 12-game development sequence, base **1/12**, direct B **1/12**, anchored
delta persistence **2/12**, and frozen initial B **2/12**. The residual design
prevents replacement of the initial adapter but still does not exceed it.

## Equal-initialization direct-B control (v126)

The direct LoRA control was corrected to start from the same frozen initial
effective B as the hypernetwork arms, then update only after its own confirmed
successes. On the 12-game development sequence, base **1/12**, persistent
state **2/12**, equal-initial direct B **2/12**, and frozen initial B **2/12**.
This confirms the earlier zero-initial direct LoRA comparison was not causal: all
three adapter arms tie the frozen start on this slice.

## Corrected failure-residual/equal-init pilot (v127)

After fixing basis growth and trajectory credit, failure gradients were
projected into ten orthogonal bounded residual bases. Direct B also started
from the frozen initial B. On 12 development games: base **1/12**, frozen
initial B **2/12**, persistent failure-residual state **2/12**, equal-init direct
B **2/12**. Failure residuals are now genuinely mounted, but they still do not
exceed the frozen adapter.

## Success-only update control (v128)

Failure updates were disabled completely; failed episodes only logged evidence.
Direct B started from the same frozen initial B. On the independent 36-game
slice, base scored **7/36**, frozen initial B **10/36**, success-only
persistent state **9/36**, and equal-init direct B **10/36**. Removing negative
updates did not recover the persistent arm; direct B matched frozen while the
hypernetwork state remained one success lower. The loss is therefore not
caused only by failure-driven drift.

## Signed basis composition pilot (v129)

Replacing softmax convex composition with bounded signed coefficients gives
the same frozen initial B at time zero while allowing positive and negative
combinations after trajectory feedback. On the 12-game development sequence:
base **1/12**, frozen initial B **2/12**, equal-init direct B **2/12**, and
signed persistent composition **3/12**. The last double-object task became a
persistent-only success after the preceding same-family success. This is the
first corrected small-sample signal beyond frozen initialization, but it must
be replicated on a separate slice before claiming continual-learning gain.

## v130 attribution audit (supersedes independent-slice and signed-direction claims)

The 36-task slice overlaps the 12-task development slice in **4 exact game
paths**. Neither slice is an independent confirmation set after these repeated
experiments. The earlier description of an independent 36-game result is
withdrawn; the raw scores are preserved.

Immediately before v129's final exclusive success, the four coefficient values
were between 0.623659 and 0.627146, versus 0.125 at initialization. Thus each
was about 4.99–5.02 times its initial value. This is consistent with common
adapter scaling, not evidence that signed cancellation or experience-specific
basis selection caused the extra success. Full coefficient vectors were not
saved in v129; the revised controller now records them.

A tied-coefficient ablation is implemented via --signed-composition
--tied-signed. It projects gradients onto the common coefficient direction
using the mean gradient, preserving the comparable common-mode learning rate.
Its actual matrix stays a scalar multiple of the initial adapter. Regression
tests verify this against an unconstrained update with unequal gradients;
18 tests pass. The new ablation has not yet been evaluated in the environment.
Prepare reviewed bindings for revised trajectory inputs before starting it.

Also, v124's 2/12 does not prove anchoring repaired the v123 regression:
v122 already scored 2/12 on that same slice without anchoring. The claimed
safety improvement is unproven, and v123's single paired loss is descriptive,
not a statistical rejection of the method.

## Tied-scale attribution (v130)

The tied-coefficient common-mode ablation also scored **3/12**, exactly
matching the unconstrained signed controller. This shows the apparent v129
extra success is explained by common scaling, not signed cancellation or
experience-specific basis selection. The v129 score must not be described as
a signed-composition gain. A norm-preserving signed-direction ablation is now
needed; its coefficients will be projected back to the initial L1 magnitude
after every update.

## Norm-preserving signed-direction pilot (v131)

The signed coefficients were projected back to the initial L1 magnitude after
every update, eliminating common scaling. On the 12-game development slice,
base **1/12**, frozen initial B **2/12**, equal-init direct B **2/12**, and
norm-preserving signed persistence **2/12**. The v129/v130 extra success was
therefore entirely a scale effect; changing the composition direction alone did
not improve transfer.

## Positive tied-scale pilot (v134)

A positive tied coefficient was allowed to scale the frozen mixture, with success-only updates and equal-initial direct B. On the 12-game development slice: base **1/12**, frozen initial B **2/12**, equal-init direct B **2/12**, and positive persistent scale **3/12**. The coefficient can grow substantially, so this result is not evidence of trajectory-specific experience; it is an unbounded gain/scaling effect and is retained only as an attribution control.

## Bounded positive tied-scale pilot (v135)

The same positive tied scale was bounded to `[0, 0.1875]` around the initial coefficient `0.125`, limiting the common adapter scale to at most 1.5x. Success-only updates were used. On the 12-game development slice: base **1/12**, frozen initial B **2/12**, equal-init direct B **2/12**, and bounded persistent scale **2/12**. The extra v134 success disappears under a small scale bound, so the current evidence does not show a gain from persistent trajectory-derived basis composition.

## Verified-base hindsight residual pilot (v136)

The runner now supports an explicit online write source. Each task first produces a base-model trajectory; when that trajectory is verified successful, the frozen hypernetwork compiles it into a bounded residual around the frozen initial adapter. This residual is written to the persistent state even if the current persistent arm failed. The direct LoRA control starts from the same frozen adapter and receives the same success-only update rule for its own successful trajectories. On the 12-game development sequence, base **1/12**, frozen initial B **2/12**, persistent hindsight residual **2/12**, and equal-init direct LoRA **2/12**. Only one base success was available for writing. The result ties the frozen control, so using a verified base trajectory as a write source does not yet demonstrate a cross-task gain.


## v136/v137 interpretation correction

v136 has **zero post-write reads**: its only base success is the last
look-at-object task, and the state is keyed by family. No subsequent task
uses that updated state. Its 2/12 therefore cannot assess transfer at all.
The base-sourced persistent arm and own-sourced direct arm have different
supervision and collection budgets; equal initialization alone does not make
this a fair online-learning comparison. v137 is a descriptive shared-base
source diagnostic on already exposed valid_seen games, not an independent
confirmation or a same-source LoRA comparison.

The name `hindsight_basis` means compiling an observed successful trajectory
through the existing frozen hypernetwork. It does **not** implement a
hindsight-conditioned teacher distribution or ASCENT. Likewise,
`credit_steps=0` implements full-trajectory action cross-entropy, not soft
teacher distillation. Local failure contrast is not implemented yet.
These runs reuse old target manifests and have not received new protocol
bindings or independent environment replay; treat scores as provisional
engineering diagnostics pending those audits. No benchmark superiority claim
is justified by them.

## Shared-source control implementation (v138, not yet evaluated)

`--shared-write-source` now directs direct-B training to the exact same
verified base/frozen episode as the persistent arm. It requires success-only
updates and identical initial B. A failed common source yields no update even
if the direct arm itself succeeded. The common episode content hash is logged.
This matches supervision selection and action-loss settings, not trainable
parameter count, family routing, gradient cost, or hypernetwork encoding cost.
Twenty unit tests pass, including the failed-shared-source/own-success case.
No new environment run with this option has started; new protocol bindings
are required before launch.

`audit_basis_read_opportunities_v138.py` reports past-write read opportunities,
action changes and paired reward differences. Its v136 audit confirms zero
post-write reads. This is a log audit, not independent environment replay.

## v137 verified-base hindsight residual on exposed valid_seen-36

The same base-source hindsight residual was run on the already exposed 36-game
`valid_seen` sequence. Base, frozen initial B, persistent hindsight residual,
and equal-init direct LoRA each scored **8/36**. Eight successful base episodes
were written; the log audit found **16** subsequent same-family reads, only one
persistent action sequence different from frozen, and zero net success
advantage over frozen. This is a descriptive engineering diagnostic on a
reused sequence, not an independent confirmation. It also used a base success
as the shared persistent source while direct LoRA retained its own rollout
source, so it is not yet the final same-source causal comparison.

### v138 launch and interpretation correction

The shared-source run was launched before generating new reviewed protocol
bindings, contrary to the launch prerequisite above. Preserve its output as
an engineering diagnostic; do not claim it is an audited benchmark. The old
manifest does not establish review of this changed supervision protocol.
At 33 completed tasks, all arms score 8, and all recorded shared-source hashes
and update eligibility agree with the corresponding base episodes. Persistent
versus frozen action lists differ once; direct versus persistent differ twice.
Equal success totals do not imply identical behavior. Matching source alone
does not isolate parameter capacity: learning rate, routing and update
parameterization still differ. These limitations supersede broader commentary
claims about strict fairness or capacity attribution.

## v138 shared-source engineering diagnostic (complete)

Using the same verified base-model success episode for both update arms,
base, frozen initial B, persistent hindsight residual, and direct LoRA all
scored **8/36** on the reused valid_seen sequence. The shared source hash and
update eligibility matched for all rows. There were 16 post-write same-family
reads and one persistent-vs-frozen action-list change; the post-write net
success difference was **0**. This rules out the earlier source-selection
confound for this diagnostic, but not capacity, learning-rate, routing, or
manifest-review confounds. Because the run reused an existing reviewed target
manifest before the changed protocol received new bindings, it is not an
independent benchmark result.

## v140 full-trajectory hindsight diagnostic (complete)

After v139 hit a CUDA memory limit while retaining all trajectory graphs, the
runner was changed to use `torch.autograd.graph.save_on_cpu` for
`credit_steps=0`; the 20 unit tests still pass. The new content-bound v139
manifest was used without changing its target games or using outcome labels.
The complete 36-game run scored base **8/36**, frozen initial B **8/36**,
shared-source persistent full-trajectory residual **8/36**, and shared-source
direct LoRA **8/36**. The final audit records all rows complete, the source
eligibility and shared source hashes consistent, and post-write reward
advantage still zero. This is an engineering diagnostic on the exposed
valid_seen sequence; it does not establish independent benchmark performance.

### v140 completion correction

The first audit invocation occurred while v140 was still live and reported
only 32 rows; that intermediate audit is invalid and must not be used. The
process subsequently exited after all **36/36** rows. The final audit is
`results/trajectory_hyperlora/alfworld_full_credit_v140_final_audit.json`:
base **8/36**, frozen **8/36**, persistent full-trajectory residual **8/36**,
and shared-source direct LoRA **8/36**; 16 post-write reads, zero
persistent-vs-frozen action changes, two direct-vs-persistent changes, and
zero post-write net success advantage. The earlier section's statement that
the run was already complete is superseded by this final audit.

## v141 half-residual amplitude diagnostic

A new content-bound manifest increased the residual cap from 25% to 50% while
keeping shared base success sources, equal frozen initialization, and the
historical four-step credit window. On 12 development tasks, base **4/12**,
frozen **4/12**, persistent **4/12**, and direct LoRA **4/12**; there was no
post-write net success advantage. Increasing residual amplitude alone did not
recover the missing transfer signal.

## v142 context-keyed prototype memory diagnostic

A new protocol binds each verified base success to a persistent prototype
adapter and reads the nearest prototype by contextual cosine similarity on the
next task. On 12 development tasks, base **4/12**, frozen **4/12**, prototype
persistent state **4/12**, and shared-source direct LoRA **4/12**. Four
prototypes remained in the bank and there was no post-write net advantage.
Context-keyed full-adapter retrieval did not improve this development slice.

## v143 low-weight successful-trajectory basis composition

Each verified base success was compiled into a new immutable basis vector with
initial logit **−4**, then composed with the frozen basis bank and updated from
the same source trajectory. On 12 development tasks, base **4/12**, frozen
**4/12**, persistent basis composition **4/12**, and shared-source direct LoRA
**4/12**. The direct append-composition mechanism did not improve the slice.

## v145 protocol enforcement audit (supersedes binding claims for v141–v144)

The previous freezer changed the protocol name without changing all settings.
Inspection of actual saved run arguments finds `credit_steps=4` in v141–v144,
whereas their new manifests declare `credit_steps=0`. Thus the earlier claims
of correctly bound new protocols for these runs are withdrawn. Their raw
observations remain engineering diagnostics only. v140's declared fields
match, but the old schema omits controller choice, learning rates and several
other settings, so this is not proof of complete protocol coverage.

The runner now compares every declared protocol setting to its runtime
arguments, verifies target order, per-target content digest, review flag,
unique relative game paths and current game bytes before loading a model.
A regression test rejects the exact credit-window mismatch and altered game
bytes; 21 tests pass. This does not retroactively validate earlier runs or
replace independent annotation review. No further run should launch until the
manifest schema includes all behavior-affecting settings.

Additional interpretation corrections: v142 uses global prototype retrieval;
the prior family-based `post_write_reads=2` audit is inapplicable to that
controller. Actual retrieval selections were not saved, so actual prototype
reads cannot be counted from that audit. v141 and v143 each have only two
post-write family reads; their null scores cannot establish that amplitude or
basis composition is generally ineffective. `credit_steps=0` is weighted
action imitation, not teacher soft distillation. Residual bases are normalized
to the configured fraction individually; that fraction is neither an upper
bound on the accumulated residual nor a bound in effective BA space.

## v144 deferred target-conditioned composition diagnostic

The controller was changed to defer a successful base trajectory until the
next same-family task. At that next task, the frozen hypernetwork receives the
previous trajectory as source and the current task observation as target;
composition and direct LoRA both receive the deferred source update before
acting. On 12 development tasks, base **4/12**, frozen **4/12**, deferred
persistent composition **4/12**, and direct LoRA **4/12**. There were four
source writes and two deferred reads/updates, with zero post-read net success
advantage. The run completed before the later strict manifest validator was
added, so it remains an engineering diagnostic.

## v145 strictly bound deferred target-conditioned composition

The full behavior configuration was included in a new manifest and validated
before model loading: base source, shared source, success-only updates, frozen
initialization, four-step credit, learning rates, family routing, low-weight
basis append, and deferred target conditioning. On 12 development tasks, base
**4/12**, frozen **4/12**, deferred persistent composition **4/12**, and
shared-source direct LoRA **4/12**. There were four deferred source writes,
two deferred reads/updates, and zero post-read net success advantage. Unlike
v144, the runtime configuration passed the strict binding validator.

## v146 deferred composition with base-teacher KL

Completed 12 reused development tasks: base **4/12**, frozen adapter **4/12**,
persistent composition **4/12**, direct LoRA **4/12**. Four successful source
writes yielded only two deferred reads. Both coefficient updates changed the
mounted matrices (reported delta norms 0.449 and 0.576), but **zero** tasks
changed their persistent-versus-frozen action sequence. Direct LoRA also had
zero action-sequence changes relative to frozen. Post-read net success was zero.
Audit: `results/trajectory_hyperlora/alfworld_deferred_kl_v146_audit.json`.

The teacher is the zero-adapter base, evaluated on observations and action
prefixes from its successful trajectory. It has no privileged future-success
context. Thus this is base-behavior distillation on selected successful states,
not a teacher improved by hindsight reasoning. It cannot establish the value
of privileged hindsight distillation. The KL implementation uses model-logit
precision and batchmean reduction (a sum across action tokens), so its scale
also varies with action length. These numerical details must be addressed in
a separately identified follow-up, not silently changed in the completed run.
The manifest validator passed at launch; this is not independent annotation
review or proof that every behavior-affecting dependency has been bound.

### Next diagnostic decision

Do not expand the hypernetwork or claim that memory generally fails from two
reads. Before another performance sweep, test whether trajectory-derived basis
directions can change the ranking of useful actions at all. Separate the
questions: (1) usable basis direction, (2) coefficient update magnitude,
(3) transfer to a later task. A bounded coefficient diagnostic on development
source trajectories can test (1) and (2); its selected coefficients must be
fixed before scoring a later task and must not use that task's reward. Compare
correct source versus shuffled source under the same norm and update budget.
Only proceed to fresh held-out evaluation after this mechanism check succeeds.

## v147 deferred composition with compiled-trajectory KL teacher

The v146 teacher bug was corrected: v146 distilled the zero-adapter base,
whereas v147 uses the LoRA basis compiled from the verified source trajectory
under the current target observation as the KL teacher. The strict v147 run
completed 12 tasks with the same bound source and budget. See
`results/trajectory_hyperlora/alfworld_deferred_compiled_kl_v147_audit.json`
for the audited counts and action comparisons.

## v148 direct deferred basis read diagnostic

v148 bypassed the low-weight persistent composition on a later same-family task:
the LoRA basis compiled from the prior verified trajectory under the current
target was mounted directly. It still produced base/frozen/persistent/direct
**4/12**. The audit records whether direct mounting changed actions; this is a
mechanism diagnostic, not a claim of persistent-memory gain.

## v149 scaled direct deferred basis diagnostic

The direct deferred basis read was multiplied by 4 to test whether v148 was
only too small. On the strict 12-task sequence, base and frozen scored 4/12,
direct LoRA 4/12, and persistent direct-basis read **3/12**. The first
post-write read already regressed from frozen success to persistent failure;
the later post-read tasks did not recover. The audit records action changes and
source writes. This rejects unconditional norm amplification: the generated
trajectory direction can affect behavior, but the current direction is not a
reliable positive action direction.

## v150 source-loss safety gate

v150 added a reward-free write/read gate: a generated deferred basis is read
only when its action loss on the verified source trajectory is no worse than
the frozen initial adapter. On the strict 12-task sequence, the audit records
accepted/rejected source-loss gates and the resulting persistent score. This
is a source-fit safety diagnostic; source loss is not evidence of target-task
success and the gate is not allowed to inspect target reward.

## v151 target-action margin gate

v151 scored the legal first-action candidates under the frozen and direct
trajectory-basis adapters, accepting the generated basis only when its top-1
margin was larger. The gate used only the current initial observation and legal
commands, never target reward. On 12 strict tasks, base/frozen/persistent/direct
scores and gate acceptance counts are in
`results/trajectory_hyperlora/alfworld_deferred_action_gate_v151_audit.json`.
This is the first target-state action-selection gate; it remains a diagnostic
because confidence margin is not a verified action utility label.

## v151 utility-objective hypernetwork smoke training

Added `train_external_utility_hyper_v151.py`, which keeps the target-disjoint
CLBench corpus and matched direct-LoRA arm but replaces token imitation with a
pairwise sequence-score objective: the recorded next action must outrank a
deterministically permuted answer. A 100-update smoke run completed at
`results/trajectory_hyperlora/external_utility_hyper_v151_20261011_steps100`.
Both arms received finite gradients and the hypernetwork sometimes produced a
larger positive margin than direct LoRA (for example update 70: 1.31 vs 0.66),
but the hypernetwork gradient norm became unstable (22.3, 15.5, 18.4 at
updates 80, 90, 100). This is evidence that the utility objective is wired,
not a transferable performance result. The deterministic answer permutation
is only a smoke negative; a full run must use reviewed future-action pairs,
clip/update-normalize the hypernetwork gradients, and then undergo frozen
cross-domain evaluation.


## Utility norm/anchor follow-up (2026-10-11)

Unregularized utility transfer completed: base 4/12; frozen, persistent, and direct each 3/12. These are development tasks, not unseen confirmation. Code audit: `--direct-init-frozen` initializes online B from the same external hypernetwork fallback; A is also shared. Frozen fallback averages four external generated bases. Therefore this pilot does not compare an independently trained direct LoRA with dynamic per-target hyper generation.

Added generated-B mean-square penalty (0.01) and one-sided positive log-likelihood anchor (0.1), equally to training arms. This anchor is not distribution KL and does not guarantee behavior preservation; B-only penalty does not bound effective BA if A changes. Deterministic permuted negatives remain unverified and may coincide with positives: this is a regularization diagnostic, not a validated future-utility dataset. 100-step smoke and 600-step training completed; late gradient spikes persist (step 440 preclip norm ~80). 21 existing controller/binding tests passed; trainer syntax and diff checks passed. Evaluation live session 16783, output `alfworld_utility_hyper_v151_norm_anchor_dev12.json`; 8/12 observed with all arms at 4 successes. Full outcome pending.


## v152 corpus audit

Reproducible audit script `audit_external_experience_v152.py` found 244/826 rows with empty trajectory sources, including 174/600 scheduled training updates. The v151 deterministic negative-selection fallback still produces an identical positive/negative answer for 12 rows (8 scheduled updates). Unequal answers are not thereby verified negatives. This changes the next priority from more regularizer sweeps to source-use diagnostics and reviewed utility supervision. No new labels were created. Artifact: `results/trajectory_hyperlora/external_experience_audit_v152.json`.


Final v152 transfer result: norm/anchor utility hypernetwork completed 12/12. Base, frozen, persistent, and direct each achieved 4/12; no post-read success gain and no separation from the controls. Audit `results/trajectory_hyperlora/alfworld_utility_hyper_v151_norm_anchor_audit.json`.


## v153 source-dependence diagnostic

On 24 non-empty external-corpus rows, the trained hypernetwork responded to trajectory content: true-source vs shuffled-source generated-B RMSE is recorded per row; mean true-source score minus empty-source score was -0.114, while shuffled-source minus empty was -0.239. However true B norm (0.0686) and shuffled B norm (0.0687) were nearly identical, and both reduced the correct-answer score on average. This shows the encoder reads source content but does not yet convert it into reliably useful executable experience. Artifact: `results/trajectory_hyperlora/external_source_dependence_v153b.json`.


## v155 correction of source diagnostic and v154 scope

v153 mislabeled disabled adapters as empty-source hypernetwork and did not actually filter empty sources (nonzero embeddings are insufficient). Its scores were recorded-action mean token log probabilities, not success probabilities; actions are not verified correct labels. Corrected v155 filters the exact empty-source digest, uses float32 log-softmax and compares base/empty-source/true/shuffled on 24 nonempty external corpus rows. True minus base = -0.124895; shuffled minus base = -0.137846; true minus empty-source = +1.721175; true minus shuffled = +0.012951. Thus the large empty-source contrast is not strong evidence of content-specific useful experience: true-vs-shuffled difference is small and both remain below base. This diagnostic is not held-out task success. Artifact `external_source_dependence_v155_corrected.json`.

v154 running session 59100 uses historical-source loss against frozen_initial, not zero B, and only gates append/update in the composition branch. Direct read overrides controller matrices and is still independently controlled by the action-margin gate. Therefore v154 cannot establish the effectiveness of a comprehensive write/read protection mechanism. No guarantee of future-task safety follows from source loss.


## v154/v156 follow-up

The trajectory-write-gate pilot completed 12/12 with base, persistent, and direct each 4/12 (frozen also 4/12 in row summaries); no measurable gain. The gate compares deferred candidate vs frozen-initial historical-source loss, so it is not a future-task utility oracle. Public-feedback audit v156 found only 2/826 explicit correct and 28/826 explicit incorrect rows; scheduled updates contain 2 correct, 18 incorrect, 49 no-results and 531 unverified. Current external corpus therefore lacks valid positive utility supervision and includes known incorrect actions. Artifact `results/trajectory_hyperlora/external_feedback_audit_v156.json`.


## v157 success-corpus candidate

Prepared audit-bound candidate `results/trajectory_hyperlora/external_success_candidate_v157_20261011`: 22 episodes/101 rows from CLBench typed-grounded runs, each episode with explicit positive feedback and no explicit failure. The manifest binds source hashes and input hashes and marks `candidate_requires_review`; no training has used this corpus yet because per-action correctness is not explicitly established. Existing external v100 training remains invalid as utility supervision.


## v157 success-filtered transfer

The reviewed episode-level success candidate (22 CLBench episodes, 101 rows) trained for 600 steps with norm/anchor objective and transferred to the same ALFWorld dev12 protocol. Base, frozen, persistent and direct each scored 4/12; the candidate did not improve transfer. The corpus has better outcome provenance than v100 but labels intermediate actions only as episode-utility candidates, so this is not proof of action-level utility learning. Audit `results/trajectory_hyperlora/alfworld_success_hyper_v157_audit.json`.


## v158 exact-query success/failure pair transfer

Built 35 exact-query action pairs from official success/failure episode markers. Pair training drove training margins high but generated-B mean-square norm to about 0.27 by step 600. On ALFWorld dev12, base/frozen 4/12, persistent/direct 3/12. This is direct evidence that true failure supervision can change behavior but naïve pairwise ranking overfits and causes negative transfer. Next control: stronger effective-matrix norm/trust-region penalty and early stopping selected without target-task labels. Audit `results/trajectory_hyperlora/alfworld_paired_return_v158_audit.json`.


## v159 strong trust-region pair control

Pair-trained hypernetwork with B norm penalty 1.0 and base-positive anchor 0.5 completed 600 steps. Final sampled B mean-square norm ~0.019 (vs ~0.27 unregularized pair training). ALFWorld dev12: base/frozen/persistent/direct all 4/12. Strong trust region removes the observed 3/12 negative transfer but has no gain yet. Audit `results/trajectory_hyperlora/alfworld_paired_return_v159_strong_audit.json`.


## v159 step100 early-stop control

The strong-constraint checkpoint at step100 was frozen and evaluated on the same ALFWorld dev12. Base/frozen/persistent/direct all 4/12, matching step600. Early stopping alone does not yield transfer gain. Audit `results/trajectory_hyperlora/alfworld_paired_return_v159_step100_audit.json`.


## v160 executable-action audit

v158 has 35 serialized response pairs, but 6 pairs have identical executable actions after stripping thinking/thought; 24 have empty source prefixes. Thus training margin can reflect rationale/style discrimination and query-only learning. Exact public query is not an identical history or hidden-state checkpoint; poker win/loss is stochastic and does not prove causal action value. Prior descriptions of these as verified step-level utility pairs were too strong. Next corpus must bind full causal histories, exclude same executable actions, preserve outcome uncertainty, and split by source task before validation. Audit `results/trajectory_hyperlora/paired_executable_action_audit_v160.json`.


## v161 full public-history audit (supersedes v158 supervision claims)

Revalidated frozen source hashes and uniquely bound each action to its source step. Of 35 v158 pairs, 24 share full histories only at empty prefixes; 10 have both prefixes nonempty; zero have nonempty identical full public histories and distinct executable actions. Ignoring rationale raises history matches to 28 but still does not establish identical hidden state or causal utility. v158 is outcome-associated observational data, not verified step-return supervision. Old artifacts preserved; builder now labels candidates as requiring causal review; trainer rejects candidate status and the historical v158 protocol. Three focused tests cover prefix leakage, differing feedback, and executable-action identity. Next required direction: collect branching rollouts from reproducible external environment states with multiple outcome samples, or construct explicitly verified external support/query tasks; no further target-dev12 parameter sweep justified by these labels.

## v163 same-state branch-return utility training (2026-10-11)

To replace the unreviewed v158 query-pair corpus, Poker was run from 100 fixed initial states with each legal first action branched from the same state and a fixed deterministic continuation. All 100 states had action-varying returns; the mean best-minus-worst return spread was 2.08 big-blind units. The resulting reviewed corpus has 100 positive-best versus negative-worst action pairs and forbids XLand, ALFWorld, and HLE target data.

The external utility hypernetwork was trained for 600 updates with a 2e-6 learning rate, B-norm penalty 1.0, and base-behavior anchor 0.5. Training completed with final generated-B mean-square norm approximately 3.1e-5. A 2e-5 learning-rate control became unstable after about step 390 and was stopped; it was not used for transfer.

On the fixed ALFWorld dev12 transfer protocol, the low-learning-rate frozen hypernetwork, persistent branch, direct online LoRA, and base each scored **4/12**. The persistent arm made one recorded update, but no score gain. This is a valid negative transfer result: same-state branch-return supervision is executable and stable, but one Poker source domain is not yet evidence of a general cross-domain utility hypernetwork.

### v163 publication correction

The dev12 artifact records one infrastructure failure in each of persistent and direct arms, versus zero in base/frozen. All four arms record four successes, but the former two have only 11 completed episodes. Missing rewards were summed as zero by the reporting fallback; this is not an all-complete paired evaluation. The earlier statement calling this a valid negative transfer result is superseded: no gain was observed, with unresolved infrastructure failures. v164 database branches remain unreviewed draft outputs, not training labels.
