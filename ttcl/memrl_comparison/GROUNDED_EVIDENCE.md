# Grounded evidence: one cross-benchmark MemRL candidate

This is a new research candidate, separate from the completed native MemRL run,
the task-aware CLBench public-state extension, and the earlier notebook and
structured-evidence pilots. It is not yet a demonstrated improvement.

## Latest frozen v21 outcome (2026-10-02 UTC)

The requested all-domain improvement has **not** been achieved. The full
ALFWorld clean-family repeat-92602 regressed from 16/31 native successes
to 10/31 candidate successes within three attempts, while using 804 more
actor calls. Look-family repeat-92602 also regressed, from 11/18 native
successes to 9/18 candidate successes, with 423 extra actor calls. Four
ALF families remain unmeasured for v21, but these two complete losses
already reject the universal claim. The four CLBench domains have two
complete online repeats each, scored on the
official last 80%:

| Domain | Repeat 303 candidate − native | Repeat 404 candidate − native | Paired suffix status |
| --- | ---: | ---: | --- |
| Spectrum | +0.157879 | +0.161507 | 142 wins, 0 losses, 2 ties; 180/180 final actions independently re-scored |
| Cohort | +0.075249 | +0.090002 | 30 wins, 2 losses; one shared prefix schema failure retained |
| Database | -0.025000 | +0.049996 | Mixed directions; sparse non-tied outcomes |
| Poker | +0.109375 | +3.078125 | Heavy chip tails; pooled descriptive interval crosses zero |

These are within-run paired differences, not cross-version policy effects.
The detailed source bindings, official-score audits, costs, and caveats are
recorded below and in each ignored run's `design.json` and `analysis.json`.

### Why the retained experience did not protect ALFWorld

The ALFWorld text-command path in v21 inherits v15's retrieval schedule:
attempts one and two call native MemRL retrieval against that arm's *own*
online memory, while attempt three removes the retrieved actor context and
withholds Q credit from suppressed IDs. The public-event ledger and typed
JSON action projection do not enter the ALFWorld actor path. Thus retaining
the native memory implementation does not mean that both arms see identical
experience after their online trajectories diverge.

A read-only audit of the complete `valid_unseen` repeat-92602 chains found
that first-attempt context hashes matched on only 2/31 clean games and 2/18
look games. Every candidate-only first-attempt loss (8 clean, 5 look) had a
different context from native, although the candidate retrieved nonempty
native MemRL experience; the paired game, initial observation, and actor seed
matched. The saved first-attempt actions diverged by steps 1–8 in those cases.
For example, clean game 14 retrieved three entries in each arm, but different
IDs and text: native succeeded on its first attempt and v21 failed. The
candidate's first command was `look`, while native began with
`go to handtowelholder 1`.

The earliest observed state split is concrete. In clean game 1, the first
two attempts had identical context, actions, and reward; the third attempt
omitted memory only in v21 and diverged at action 2. Both still scored zero,
but their subsequent stored trajectory text and Q values differed (for
`memory_000000`, Q was -0.51 native versus -0.30 candidate). In look, game 1
was a common first-attempt success; game 2's first two attempts matched, then
the third-attempt dropout changed the action trajectory despite both arms
eventually succeeding. Their Q values and later retrievals diverged. These
observations explain how first-attempt context can differ even though native
experience storage remains enabled. They do **not** by themselves prove that
the dropout uniquely caused every later loss: only three games across both
families reached a third attempt after fully matched first two trajectories,
and all three third-attempt rewards tied. The full online result establishes
negative transfer, while a frozen-state intervention would be needed to
isolate the contribution of a particular history change.

The follow-up, official-train fixed-snapshot probes in
`PAIRED_CREDIT_RL.md` also reject a simple prompt-level repair.
On 22 selected ALFWorld train games and three actor seeds, unchanged
retrieved text versus empty context won 17, lost 10 and tied 39
same-service first-attempt pairs (mean +0.1061, input-clustered 95%
bootstrap interval [-0.1212,+0.3333]); the look family lost eight
of its 15 pairs. Adding a general instruction to verify historical
details against current observations then lost to unchanged text on
the same actor service: 2 wins, 10 losses, 54 ties (mean -0.1212,
interval [-0.2576,0.0000]). Neither selected-train fixed-snapshot
result establishes a new online ALFWorld policy; the framing rule is
rejected.

A separate same-service official-train third-retry probe paired the
same frozen retrieval with empty context on 22 games × three actor
seeds. Full context won eight, lost nine and tied 49 (mean -0.0152;
input-clustered interval [-0.1970,+0.1818]). The opposed effects
within the selected look and cool families reject v21's blanket
third-retry dropout as a general rule. `grounded_evidence_v23.py`
therefore restores native retrieval for every retryable text-command
attempt while retaining the structured JSON projection. This code
correction is not yet a demonstrated online ALFWorld improvement.
The first two official-train look games in a v23 online wiring check
matched native retrieval IDs and context hashes on all six attempts,
with identical rewards and actor calls. This is a connectivity check,
not a scored family result.

The newer `v24` candidate tests a budget-neutral native/empty/native
schedule for every retryable text-command task. Its complete six-family,
36-game official-train online pilot yielded nine wins, two losses and
25 ties (mean success difference +0.1944); heat regressed on one game
and pick-two had both a win and a loss. The frozen `valid_unseen`
repeat-92601 evaluation completed all 24 simple pick-and-place pairs
at 2 wins, 4 losses and 18 ties (mean -0.0833), then stopped because
the predeclared all-family objective was already false. One look pair
had completed; all remaining pairs are recorded as missing, not zero.
This rejects v24's text schedule as a general online ALFWorld rule.
`v25` keeps that text
schedule and changes the JSON-schema path: the actor receives only
native MemRL context, while the source-bound public ledger remains
available to the same general type-directed action operator. It also
removes v21's post-first-action context withdrawal. This addresses the
Database/Poker first-action divergence without disabling the projection
that improved Spectrum/Cohort. The rule branches on the action
interface, never a domain or task name. Its CLBench calibration-prefix
checks are still development evidence; no v25 all-domain claim is made.
The complete repeat-404 Spectrum calibration prefix (first 18 tasks)
yielded 10 candidate wins, no losses, and eight ties, mean official
reward difference +0.074750. Native and candidate retrieved identical
memory IDs and actor context on all 18 tasks. Independent official
re-scoring of all 18 final actions matched saved rewards: the raw
actor actions totaled 3.9773, and ten type-projected final actions
raised the total to 5.3228. The Database repeat-404 first-six prefix
was six exact reward/context ties. The Cohort repeat-404 first-four
calibration prefix showed why unverified numerical history needs a burn-in:
two early numeric projections changed actor submissions. Their official
scores totaled -0.117496, versus -0.102313 for those same raw actor actions.
The candidate still beat the independent native chain's -0.137990, so a
paired-policy comparison alone would have hidden this projection loss.
`grounded_evidence_v26.py` therefore abstains from numerical consensus
until four distinct prior source episodes support it; the same source-count
condition applies to any numeric JSON action, with no domain-specific switch.
Neither prefix is a scored last-80% CLBench result. Poker has not yet
completed a v25 check, and v26 needs its own online validation.

The v26 Cohort-404 first-four calibration prefix completed with two
early numeric candidates rejected, zero final-action changes, and four
independently re-scored rewards. Candidate minus native averaged
+0.006155, with one loss, one win, and two ties; this is a prefix
diagnostic, not a suffix result. The v26 Poker-404 first-eight prefix
had eight exact reward and actor-call ties and matched native retrieval
context hashes/IDs in all eight hands. Full v26 Cohort-404 and
Database-404 online chains were then started to test the scored suffix.

The v26 Database-404 full chain has completed all 30 paired episodes
without failures. All 24 official suffix rewards tied: candidate minus
native mean was zero. Native and candidate retrieval context hashes and
IDs matched on all 30 tasks, no typed action was changed, and actor-call
totals were equal. This avoids a negative difference on this new chain,
but supplies no Database improvement.

The v26 Cohort-404 full chain completed all 20 pairs. Independent
official re-scoring matched all 20 final actions. On the 16 scored
last-80% tasks, candidate reward averaged 0.068341 versus native
-0.003495: mean difference +0.071836, 15 wins and one loss. All
20 retrieval context hashes matched across policies. Sixteen numeric
projections increased the candidate raw-action total by 1.149368;
the scored suffix had equal actor-call totals across arms. This is
one online repeat, not a four-domain claim. Poker-404 has started
on the released actor service.

The v26 Spectrum-404 full chain completed all 90 pairs, and independent
official re-scoring matched all 90 candidate final actions. On the 72
scored last-80% tasks, candidate reward averaged 0.379572 versus native
0.218065: mean difference +0.161507, 72 wins and no losses or ties,
with equal actor-call totals. Across all 90 tasks, 82 recurring-record
projections changed the final action. Candidate and native retrieval
context/IDs matched on 54/90 tasks; their online memory states can
diverge after different rewards, so the full-chain gain is not an
isolated per-action intervention estimate.

The v26 Poker-404 full chain completed all 120 pairs without failures.
All 96 scored suffix rewards tied (both means -1.796875), with equal
actor-call totals and matching retrieval context hashes on every hand.
All 351 typed-action audits were KEEP. Hand 115 produced the same
52-action, -100 return in both arms: 50 successive RAISE actions
followed by CALL. Thus v26 prevents candidate-specific Poker drift
on this chain but does not improve Poker or cure its action loop.
Together with Database's full tie, v26 is not a four-domain gain.

`grounded_evidence_v27.py` removes the failed v24 text schedule from
the combined policy: retryable text commands use v23's native retrieval
on all attempts, while structured JSON actions retain v26's native
actor context and four-source numeric gate. The interface routing is
unit-tested. This is a regression correction, not evidence of an
ALFWorld gain; no universal-improvement claim follows from v27.
An independent `evaluate_typed_grounded_v27.py` runner now freezes source
hashes and paired input bindings for that policy. A two-game `valid_unseen`
pilot in each of clean and look (repeat 92602, current frozen actor service)
completed all eight arm cells. Within every game, native and v27 had identical
retrieval IDs, context hashes, public action trajectories, rewards, and actor
call counts on every attempted retry. Clean scored 0/2 in each arm (300 actor
calls each); look scored 2/2 in each arm (70 actor calls each). Clean game 1
had a third attempt with two nonempty retrieved IDs in both arms, directly
checking that v27 removed v21's dropout at the earlier split. These four
paired ties are a wiring check, not a full-family or cross-service score.
The first attempted run is retained separately as a failed cell: its actor
URL accidentally included `/v1`, which the client adds automatically, and
received HTTP 404 before any model result. The completed pilot used a new
output directory and the corrected base URL.

The later independent repeat-92602 `valid_unseen` clean-family run completed
all 31 paired games without failed cells. Native and v27 both succeeded on
12/31 first attempts and 18/31 within three attempts: 31 ties, zero wins,
zero losses. All 63 paired attempts matched on retrieval context hashes and
IDs, public trajectories and rewards; each arm used 2,567 actor calls and
17,028,659 input tokens. This establishes same-run parity for the full clean
family, rather than a v27 success gain. The look-family run was stopped at
the user's request after 13 of 18 paired games. In the completed prefix,
both arms succeeded on 2/13 first attempts and 6/13 within three attempts,
with 13 ties and no wins or losses; actor calls and input tokens were equal
at 1,354 and 8,702,080 per arm. The five uncompleted games are recorded as
missing, and the interrupted native game has no scored row. These same-run
results do not establish a full look-family score. The structured CLBench
branch inherits v26, whose repeat-404 four-domain results are reported above;
v27 has not had an independent full CLBench evaluation.

`grounded_evidence_v28.py` is a new training-only writer candidate.
On any failed public trajectory, it uses the existing single MemRL
writer call and token cap to ask for an evidence-bound retry reflection:
separate observations from guesses, identify a contradicted assumption,
and give at most four checkable next steps. Successful writes, Q updates,
retrieval budgets and action projection remain v27 behavior. The same
failure rule applies across tasks and domains. Its two-game official
ALFWorld train wiring check completed: candidate succeeded twice while
native failed twice, with the intended failure/success writer routes
verified. The second candidate game benefited from a changed online
history, so this is not a per-reflection causal estimate. Its frozen
six-family, 36-game official-train pilot later stopped early after
negative complete families; v28 has no independent evaluation claim.
Its first complete look-family train block yielded three wins, two
losses and one tie (mean +0.1667), with 168 fewer actor calls than
native. Subsequent complete families overturned that early signal.
The v28 Spectrum-404 first-18 calibration prefix independently
re-scored all 18 final actions: ten wins, no losses, eight ties and
mean paired difference +0.074750, matching v26's prefix scores.
Candidate retrieval context matched v26 on only nine of the 18 tasks,
so identical prefix scores do not prove the new writer is behaviorally
irrelevant. This prefix supplies no scored-suffix validation for v28.
The v28 ALFWorld clean-family train block completed at zero wins, one
loss and five ties (mean -0.1667). Its Cohort-404 first-four prefix
also completed: four independently re-scored final actions, no numeric
projection changes, and candidate minus same-service native mean
-0.005668 (one loss, three ties). Native Cohort absolute scores differed
between the v26 and v28 actor services despite fixed seeds, so
cross-run score subtraction is not an algorithm-only comparison.
The cool-family train block also completed at zero wins, one loss and
five ties; simple tied all six games. After four complete train
families (24/36 pairs), the v28 run stopped: look was 3 wins, 2 losses
and one tie, simple 6 ties, clean 1 loss/5 ties, cool 1 loss/5 ties.
The 12 unrun heat/pick-two pairs are preserved as missing in the
frozen `stopped_early.json` and `analysis_partial.json`. These train
and Cohort-prefix failures reject v28 as a universal candidate.

`grounded_evidence_v29.py` instead leaves the native writer untouched.
For any retryable text task, it stores the existing failed-attempt
reflection only for retries on that exact task, presenting it in place
of cross-task retrieved text on attempts two and three. The new note
is cleared at the next task; suppressed native IDs receive no Q credit,
while the normal MemRL memory write and Q update still run. First
attempts and structured actions are v27 behavior. The rule uses no
extra model calls or task-family names. A two-game official-train
wiring check completed: one candidate-only win and one tie, with the
intended first/native and retry/local context hashes, source bindings,
and Q-credit suppression recorded. The win arose on the next game's
first attempt after online memory divergence, so it is not an isolated
effect of that game's retry reflection. Its frozen six-family,
36-game official-train pilot later stopped after a negative clean
family; no independent ALFWorld improvement is claimed.
The two-game smoke win did not replicate in the fresh full train chain.
On its first game, the candidate's second attempt first diverged at
action six even though both runs used the same actor service, rendered
prompt SHA-256 and sampling seed at that step. This observed sampling
variation reinforces the need for paired, repeated online evidence;
the smoke outcome is only a wiring check.
The first complete v29 look-family train block then yielded two wins,
no losses and four ties (mean +0.3333), with 232 fewer actor calls.
The simple-family block tied all six pairs. The clean-family block then
had zero wins, one loss and five ties: on game five, native succeeded on
its first attempt in ten actor calls, while v29 failed all three attempts
and used 150 calls. The loss happened before the local-reflection retry
on that game. Its first-attempt actor context already differed because
earlier online trajectories and memory states had diverged after game
two; the result cannot be assigned solely to game five's retry rule.
The pilot stopped after these three complete families. Its 18/36 audited
pairs, 18 missing pairs, policy/design hashes and stop reason are saved
in `analysis_partial.json` and `stopped_early.json`. The unrun cool,
heat and pick-two families are not ties. This negative train family
rejects v29 for promotion to an unseen all-family evaluation. There is
still no demonstrated ALFWorld plus four-domain improvement.

`grounded_evidence_v30.py` adds a general readout of *observed* feedback:
the same public action must have produced the exact same substantive
environment response in at least two prior episodes. At most three such
source-bound responses are ranked against the current task and appended
only if native MemRL text plus the readout fits the original 2,048-token
budget. The rule has no benchmark or domain names and preserves native
retrieval IDs and Q credit. It applies to structured-action tasks; the
ALFWorld retryable-text path remains native. A saved-data audit found
16 qualifying action/feedback groups in the Database-404 trajectory,
none in Spectrum, Cohort or Poker under the same exact rule. This is a
candidate component, not an all-benchmark method.

The complete v30 Database-404 online chain passed 30/30 input, snapshot,
budget and prior-source readout audits. On the official final 24 tasks,
candidate minus native averaged +0.011113, with two wins, one loss and
21 ties. The candidate used 61 fewer actor calls overall, and 27/30
tasks received at least one qualifying readout. The three nonzero
paired outcomes were episode 14 (+0.2000), episode 24 (-0.7333) and
episode 25 (+0.8000). These effects largely cancel; one repeat does not
establish a stable Database gain. In three post-hoc, single-seed probes
restored from the candidate's exact prior online snapshot, the readout
raised episode 14 from 0.6667 to 0.8667 by reducing actor calls 6→3,
left episode 24 incorrect in both arms while reducing calls 10→5, and
raised episode 25 from 0.5333 to 0.8000 by reducing calls 8→4. The
probes diagnose immediate readout effects only; the full online chains
also differ through previous trajectories, writer output and memory
updates. A second frozen repeat-303 Database chain tested the direction;
v30 has no ALFWorld text-policy improvement.

That repeat-303 chain has now completed and passed the same 30/30 audit.
Its official final-24 mean candidate-minus-native difference was
**-0.050000**, with one win, two losses and 21 ties, despite 15 fewer
actor calls overall. Episode 25 lost 0.8000 after the candidate queried
only one category before answering; episode 28 lost 0.5333 after it
skipped the counts needed for a percentage, even though three repeated
feedback groups had been shown. The positive episode-14 efficiency
effect recurred (+0.1333). Across the two repeats, the mixed directions
reject v30 as a reliable Database improvement. `v32` is a development
hypothesis that abstains from injecting a single stable feedback group;
it does not address the episode-28 three-group failure. Its complete
Database-303 repeat passed 30/30 source, input and snapshot audits. On
the 24 official suffix tasks the mean paired difference was exactly
zero (three wins, two losses, 19 ties); candidate actor calls fell by
45 overall. Episode 25 lost 0.8000 and episode 30 won 0.7333 even though
the gate abstained on both current tasks. Earlier online memory-state
divergence can therefore change outcomes when the current readout is
absent. Episode 28 still lost 0.5333 with three readouts. These mixed
effects reject v32 as a reliable Database gain. A post-hoc fixed-snapshot
probe on episode 28 replayed the same candidate bank and target with and
without the readout: the readout branch scored 0.0 in 12 actor calls,
versus 0.2 in 13 calls without it. The replayed readout branch and
original online cell had the same first prompt hash and final reward,
but different later call counts. This one-seed diagnosis supports an
immediate harmful readout on that case without establishing a stable
causal effect across actor samples. No v30/v32 all-domain claim is made.

The next `v31` candidate composes v30's structured-action path with a
general public-action constraint. When an environment advertises a finite
list of legal text commands, the actor's single existing completion is
decoded using exactly that list (`guided_choice` on the installed vLLM
server). The same prompt, actor weights, sampling seed, 50-command limit,
three-attempt limit and maximum completion tokens are retained. It adds
no model call, task-family rule or hidden expert action. Structured JSON
tasks retain v30 behavior. A two-game official-train look-family wiring
check completed: native failed both after 150 actor commands per game;
the guided actor succeeded in six and eight commands, respectively.
Every guided command was present in the public admissible list. This is
one seed and two games, with online memory divergence after game one.
The frozen six-family, 36-game official-train pilot stopped after two
complete families because the second family regressed.
Its first complete look-family block passed source, input, command and
budget audits: four candidate wins, zero losses and two ties over six
paired games, with 564 fewer actor commands. The two failures on both
sides are retained as ties. The complete simple-family block had zero
wins, one loss and five ties; the candidate failed game two after 150
valid commands while native succeeded in 58 commands across two
attempts. All 12 scored pairs passed the source, input, budget and
command audits. The 24 unrun train pairs remain missing in the frozen
`analysis_partial.json` and `stopped_early.json`; they are not ties.
This rejects full-time guided decoding as the universal candidate.

`v33` is a more conservative general decoder: use native generation
while the last proposed command was public and valid; after an invalid
command, use one guided-choice completion from the current public action
list, then return to native generation. The rule uses no task family,
reward or additional actor call. Unit checks verify the mode transition.
Its two-game paired official-train simple-family pilot passed source,
input, step-mode and one-completion-per-step audits. Game one tied at
success but used 30 candidate actions versus 25 native; game two lost,
with native succeeding in 55 actions while the candidate failed after
150. The candidate used guided choice on only 3 and 25 steps,
respectively, yet this did not prevent the same simple-family negative
transfer as v31. Online memory had already diverged after game one, so
the game-two loss is not an isolated decoder effect. This small result
rejects promotion of v33 to the full all-family pilot.

`v34` moves the public-command constraint to the *next attempt* only
after a complete failed attempt has more than 40% invalid commands.
The first attempt is always native, the rule reads only public action
validity, and both modes retain one completion per environment step.
Its structured-action branch is v27, preserving the established typed
projection without v30's unreliable recurrent-feedback augmentation.
The first two look-family official-train games produced one paired win
and one tie: game one succeeded in both arms without triggering the
rule; in game two, 21/50 invalid commands on the shared native first
attempt triggered guided decoding and the candidate succeeded in seven
second-attempt commands while native failed all three attempts. The
first-attempt public trajectory and generated responses matched exactly
between arms on that game; the second-attempt retrieval IDs, context
hash and actor seed also matched. The decoder intervention therefore
explains the immediate second-attempt contrast within this one sampled
pair, although later online memory changes remain coupled to it. The
first two simple-family games tied in success and exact actor-call
counts (25 and 55), with no guided attempts. Both are small development
checks. Independent unseen ALFWorld and all-domain CLBench evidence is
still absent.
The first complete look-family train block now has one candidate win,
zero losses and five ties over six paired games; three games actually
used guided retries. The complete simple-family block tied all six
games, including one game where a guided second attempt matched native
success in the same number of calls. Later complete-family results are
reported below.

A retrospective trigger audit of the older 36-game native train source
found two cases where v34 would guide the second attempt even though
that native second attempt succeeded: one simple and one clean game.
`v35` therefore keeps both first and second attempts native, guiding
only the final attempt if both failed and either earlier attempt had
more than 40% invalid public commands. In that historical source, the
rule would reach 14 final attempts across five families and none of
those native final attempts succeeded. This retrospective observation
motivates the safer timing but does not predict a new online-chain
result. v35 is implemented and unit-tested; no model training has started
for it. A two-game official-train look smoke on a separate existing actor
service completed with two candidate wins and zero losses: the baseline
failed both games and the candidate succeeded both. In the first game,
both native attempts matched the baseline's public trajectories and
actor responses exactly, and the third-attempt retrieval context also
matched; guided decoding then succeeded in seven commands while native
failed after 50. These are same-seed local intervention results, not a
cross-family validation.

The independently started six-family v35 train chain did not replicate
the first smoke win: its first look pair tied at failure after 150 actor
calls per arm. Across runs, the first attempt was identical, but the
second attempt diverged at action 11 despite the same initial prompt
and seed. Its different public trace then produced a different writer
prompt, failed-attempt reflection, and third-attempt context. Within
each run, native and candidate received the same context at the point
of intervention. The smoke win is sensitive to the preceding online
action-and-memory chain; the writer alone cannot be assigned the
between-run difference.

At the user's request, both new full-train pilots have stopped. The
frozen v34 pilot audited 18/36 pairs: look 1 win/0 losses/5 ties,
simple 6 ties, and clean 0 wins/1 loss/5 ties. Its clean loss happened
on an unexposed current game after earlier guided retries changed the
online memory chain. Eighteen games are missing, not ties. The v35
pilot audited 3/36 pairs, all from look: 1 win/0 losses/2 ties;
33 games are missing. Their source hashes, completed rows and operator
stop reasons are in their `analysis_partial.json` and
`stopped_early.json` files. Neither is a complete six-family result;
the consolidated status is in `RESULTS_20261003.md`.

`v36` is a no-training ALFWorld candidate. It keeps v27 retrieval
on every text attempt. If two or more memories were shown, it records their
IDs as deferred for direct Q credit instead of broadcasting the terminal
reward to each of them; singleton updates and the source-bound native writer
remain unchanged. At each text step, it first asks the same native actor for
one command. A command outside the currently advertised public admissible
set is not executed: one extra guided-choice completion on that same public
observation supplies the command. Both completions, tokens, seeds, source
hashes and the repaired command are recorded and charged against the original
per-attempt actor-call limit. If the final available call produces an invalid
command, the attempt ends as a recorded failure without stepping the
environment. A valid native command is passed through. Structured JSON
actions retain v27/v26 behavior.
This design does not estimate individual causal memory value. Independent
`both`, `credit_only`, and `repair_only` runs froze each candidate mode in a new
output directory. On six official ALFWorld **train** look games (repeat 92721),
the paired three-attempt results were: `both` 5 wins / 0 losses / 1 tie
(candidate 5/6, native 0/6); `repair_only` 5 / 0 / 1 (also 5/6 vs 0/6);
`credit_only` 0 / 1 / 5 (candidate 0/6, native 1/6). These are separate online
chains, so their score differences are not a factorial estimate. Native actor
completions are stochastic across runs even with the same nominal seed.

The predeclared simple-family guardrail caught a negative result. `both`
completed only the first three of six train pairs: 0 wins / 1 loss / 2 ties,
candidate 2/3 vs native 3/3. We stopped that run; the other three games are
missing, and an interrupted fourth native game is unscored. In a separate
two-game simple pilot, `repair_only` was 0 / 1 / 1 (candidate 1/2 vs native
2/2); `credit_only` was 0 / 0 / 2, with no Q deferral on either game. On the
simple loss in the combined run, the first retrieved IDs and context were
identical and empty, and the commands matched through step 17. At step 18,
the native actor emitted invalid `go to drawer 8`, while the candidate
repaired it to admissible `go to drawer 9`. The candidate later exhausted all
three 50-call attempts and failed; native succeeded on its second attempt.
The repair-only loss confirms that Q deferral is not necessary for this
negative case. Validity repair can alter the trajectory without improving
task success.

The credit-only look loss is downstream of online memory divergence: on game
two, both arms retrieved the same IDs and context in attempts one and two,
but the candidate deferred the two-ID failed-attempt Q update whereas native
set both Q values to -0.3/-0.51 as applicable. Attempt three then had a
different retrieval order/context; by game five, the first retrieval IDs and
prompt differed, and native won while candidate failed. This does not identify
a single harmful memory or prove that Q deferral alone caused the loss.

All completed rows retained official scoring, per-attempt actor-call limits,
input hashes, frozen source hashes, and failed-attempt records. `both` used
208 candidate actor calls vs 900 native on look; `repair_only` used 318 vs
900. In the scored simple prefix, `both` used 184 vs 135, and in the
two-game `repair_only` pilot 176 vs 81. Candidate repair runs executed zero
invalid commands, but the extra guided completions count toward those totals.
No new model training, `valid_unseen` confirmation, or independent CLBench
run was performed. Structured JSON actions retain the v27/v26 branch. The
train look gain is local evidence; the simple regression rules out v36 as a
general ALFWorld improvement.

The ignored local result roots are
`results/memrl_credit_training/20261003_v36_{both,repair_only,credit_only}_alf_look_train_full6_v1/`,
`20261003_v36_both_alf_simple_train_full6_v1/`, and
`20261003_v36_{repair_only,credit_only}_alf_simple_train_pilot2_v1/`
under the same parent. Each completed run has `analysis.json`; the stopped
simple run has `analysis_partial.json` and `stopped_early.json`. Raw trajectories
and model assets are intentionally absent from Git.

A post-hoc CPU audit compared the current Database question directly
with each stable action/feedback group, instead of matching it only to
prior questions. The helpful episode-14 readout's three selected groups
had BGE-M3 similarities about 0.45–0.51, while the harmful episode-28
readout's groups were about 0.48–0.55. A simple direct-similarity
threshold cannot separate these cases; no threshold has been selected
from the official scored suffix.

## Rule

Every completed episode contributes exact public events. Each event keeps the
public input, the submitted action, and the environment's observed response as
different fields, bound to the full public trace by SHA-256. Private reasoning
fields and the official scalar reward are excluded. The complete event values
remain in the memory snapshot; prompt excerpts are separately bounded. In
particular, an acknowledgment of a submitted report is never labeled as
verification of the report.

For every new task, native MemRL still retrieves and Q-updates its memories.
The same BGE-M3 embedding ranks public events from all prior episodes, without
an absolute similarity cutoff. At most one event per past episode and five
events in total enter the actor context. Native task/lesson text uses at most
half of the existing 2,048-token memory budget when events exist; the event
view and native text together must fit the full budget. Only native memories
actually injected receive the official reward through the native Q updater.
No task-name, action-key, or domain-specific parser chooses the evidence.

Where the environment exposes a JSON action schema, a general type-directed
projection supplies two possible views of prior *submissions*: a per-field
mean for repeated numerical
maps, and recurring records in a list after learning a stable numerical
identity field from cross-episode repetition. The projection inspects JSON
shape and repetition, never a benchmark or task name. Both views remain
explicitly unverified. In `v4`, one short model call selects an applicable
operator from the current task; it does not write a large replacement JSON
action. The operator computes the action deterministically from source-bound
submissions. Tool calls are passed through. The result must validate against
the current response schema; rejection retains the original action.
ALFWorld has a text-command actor rather than that schema interface, so its
actions receive the same evidence context without the additional selector
call. The selector prompt distinguishes historical submissions from
observations. Source hashes validate provenance, not semantic truth; this
remains a limitation, and projected actions require outcome audit.

This method does not train a causal per-memory utility estimator. The native Q
update still assigns the episode reward to all injected native memories. A
source-grounded view and a review call alone cannot guarantee that a model
will use historical evidence correctly. Extra review tokens and calls must be
reported separately from actor and writer costs.

The first `v1` two-instance ALFWorld and Spectrum integration checks are
frozen in ignored `results/memrl_grounded_evidence/20261002_smoke_*`. They
completed with exact ties. Spectrum's second instance received one source and
an extra review call but retained the original action. Those prefixes have no
CLBench suffix metric. `v2` added the general type-directed view. Its Spectrum
development prefix was stopped after episode 3: the reviewer read a generated
MemRL failure reflection, removed every reported record, and reduced reward
from 0.2117 to 0.1964. The frozen partial run and operator-stop record remain
under `20261002_v2_spectrum_303_first30`. `v3` gives the actor
native MemRL text plus raw evidence, but gives the independent reviewer only
raw public events and typed candidates. If that extra review call fails, the
schema-valid original action is retained and the failure is logged. A new
output directory is required for `v3`.

The `v2` Cohort prefix was also stopped and kept under
`20261002_v2_cohort_303_first10`: two development episodes had invalid final
actions after a large response schema was combined with long injected memory.
`v4`, in `grounded_evidence_v4.py`, removes actor memory at turns
whose response schema exceeds 10,000 JSON characters, without task-name
checks. It avoids Q credit if native memory was never shown. When a compatible
typed candidate exists, one short model call chooses KEEP,
NUMERIC_CONSENSUS, or RECURRING_RECORDS; a deterministic operator builds and
schema-validates the action. Numeric consensus averages prior and current
submissions. Record recurrence appends historical clusters only when the
selector determines the task asks for persistent entities. The selector,
operator, sources, costs, and failures are recorded. `v4` has no full paired
result yet.

## Evaluation

Use the frozen original MemRL plan and a compatible actor server. The paired
entrypoint starts native MemRL and this candidate from independent empty
memory chains, with the same task order, seeds, and actor decoding:

```bash
python -m ttcl.memrl_comparison.evaluate_typed_grounded_v10 \
  --origin "$TTCL_WORKSPACE/results/memrl_comparison/20260928_budgeted" \
  --output "$TTCL_WORKSPACE/results/memrl_grounded_evidence/new_spectrum_303" \
  --benchmark clbench --task blind_spectrum_monitoring --repeat 303 \
  --url http://127.0.0.1:18557
```

Use a fresh output directory per task and repeat. For ALFWorld, pass
`--benchmark alfworld` and a family/repeat from the frozen plan. Each output
freezes the candidate and runner sources, input bindings, memory snapshots,
responses, action reviews, and failures. A short prefix is an integration
check, not a last-80% CLBench result. The full gate is all six ALFWorld
families and three seeds, plus all four available CLBench domains and two
repeats, on common valid pairs. Report first and within-three ALFWorld success,
each CLBench domain's official suffix reward, failures, added calls/tokens,
and negative transfer. Do not average CLBench rewards across domains.

## Current development status (2026-10-02)

The completed `v3` Spectrum repeat-303 pilot covered the first 30 episodes.
On its 12 scored suffix pairs, native MemRL and the candidate both averaged
0.2162417. The reviewer kept every original action, despite having a
recurring-record candidate on each scored episode, so this is a tie rather
than an improvement. Its additional review calls and input tokens are recorded
in the pilot report.

The `v4` Cohort pilot stopped after 6 of 20 cells: the selector rejected a
numeric-consensus candidate because it mistook averaging complete predictions
for filling missing values. The `v4` Spectrum pilot stopped after 24 of 60
cells: its selector repeatedly treated the public persistent-record task as
independent single scans. Both partial runs have operator-stop records. They
are development diagnostics, not official comparative results.

The separately frozen `v5` selector added the current initial public task and
source-bound prior public task context to its prompt. Its Spectrum pilot was
stopped after 22 of 60 cells: on episodes 9 and 10, the selector described a
persistent-record need but selected NUMERIC_CONSENSUS when the only available
candidate was RECURRING_RECORDS. The schema-safe fallback kept the original
action. `v6` constrained selection to labels of the currently available generic
typed candidates, with a unit test for label binding. Its Spectrum pilot was
stopped after 24 of 60 cells because episodes 9–11 still chose KEEP: the
selector saw how many records recur, but not their concrete values or support.
`v7` showed a bounded, source-bound preview of those records to the same
generic selector. Its independent paired Spectrum first-30 pilot completed:
all 12 official suffix pairs were valid, candidate 0.292833 versus native
MemRL 0.216242, with 5 wins and 7 ties. The independent official scorer
matched all 30 saved final actions. The scored pairs required 24 candidate
actor/selector calls and 101,386 input tokens, versus 12 calls and 24,208
input tokens for native MemRL. This is a development prefix, not a full-domain
result. `v8`
additionally removes unverified public-event context from tool-call schemas
while retaining native MemRL lesson text; it is being checked in a fresh
Cohort pilot. Neither pilot
nor an offline replay establishes a cross-domain improvement. The full gate remains
the six ALFWorld families across three seeds and all four CLBench domains
across two repeats, with official scoring and costs reported separately.

The completed `v5` Cohort first-10 prefix has 5 jointly scorable suffix
pairs, all candidate wins: 0.012084 versus -0.013087 for native MemRL.
One of the six planned suffix pairs failed on the candidate arm: its tool
action was schema-invalid after three format retries while native MemRL
completed. An independent scorer replayed all eight completed candidate
final actions; six changed, and every recorded final reward matched. For two
early numeric-consensus uses, the same actor's raw/final scores were
-0.011009/-0.006352 and -0.046513/-0.007027. The failed tool turn motivates
the generic `v8` context rule; `v5` is not the final candidate. `v9` instead
classifies the stable public task brief once into persistent-set and
shared-numeric-target semantics, freezes that contract in the online memory
snapshot, and applies shape-compatible operators deterministically. It keeps
the `v8` tool-schema context rule. Its independent Spectrum first-30 pilot
completed all 12 official suffix pairs: candidate 0.401433 versus native
MemRL 0.217167, with 12 wins, no losses or ties. Both arms used 12 actor
calls on the scored pairs; candidate actor input was 36,338 tokens versus
24,210 native. An independent official scorer matched all 30 saved final
actions. This remains a development prefix, not a full-domain result.

The `v8` Cohort prefix was stopped after 16 of 20 cells. Its seventh episode,
which failed in `v5`, completed after unverified event context was removed
from tool-call turns. However, the per-instance selector chose KEEP on four
scored pairs with numerical candidates, so the partial paired result was a
tie. `v9` Cohort now tests the same tool-context rule with a frozen task
contract instead of repeated per-instance selection.

The completed `v9` Cohort first-10 pilot had all six suffix pairs valid:
candidate 0.015923 versus native MemRL -0.006634, with six wins and no losses.
Scored-pair calls and input tokens were identical (38 calls and 311,814 input
tokens per arm) because the one-time task classifier ran before the suffix
and the large final schema suppressed actor memory context. The independent
official scorer matched all ten saved candidate final actions. This is still
a short development prefix.

The completed `v9` Database first-10 prefix had four valid suffix pairs,
all tied at reward 0.0. Candidate actor usage was 33 calls and 245,624 input
tokens versus 25 calls and 85,967 input tokens for native MemRL. No typed
operator applied. This is a cost regression and does not meet the all-domain
goal. The `v9` Poker first-30 pilot is still running.

The `v8` ALFWorld `look_at_obj_in_light` seed-92601 guardrail was stopped
after 5 of 36 cells. On the second fully paired game, native MemRL succeeded
on the first attempt, while the candidate failed all three attempts and used
150 actor actions. Its first retrieval included a prior desklamp event at a
different desk; the candidate visited that desk first, while native MemRL
visited another and eventually succeeded. This is an association in a
diverged rollout, not proof that a specific event caused the failure. The
partial run and operator-stop record are preserved. Cross-benchmark success
remains unproven even if CLBench pilots improve.

Two fixed-seed, single-game ALF ablations kept the second game's input and
actor seed unchanged. Removing only the event from the candidate prompt
changed the first action back to the native route, but still failed at the
50-step cap. Adding the event to native MemRL's exact original context also
failed at 50 steps. Both are saved under ignored
`20261002_alf_event_ablation_game2` and
`20261002_alf_native_plus_event_game2`; neither isolates a single cause of
the full online-chain regression. `v10` preserves native MemRL context exactly
for retryable text commands and uses the existing three-attempt budget as
native context, memory dropout after a failed first attempt, then native
context again. Structured JSON actions retain the `v9` frozen task contract.
The `v10` ALF look-family seed-92601 guardrail is running; this policy has
not yet shown an ALF benefit.

The `v11` retrieval experiment retains exact native MemRL context for
structured actions as well as text commands. It ranks prior public events
using the submitted action and the observed environment response, deduplicates
identical action-response pairs, and spends remaining memory tokens on a
short input plus a response-first excerpt. Source bindings and the same
2,048-token cap remain. Its completed first-10 Database development pilot had
four valid suffix pairs, all tied at 0.0. Candidate usage rose from 27 to 29
actor calls and from 93,166 to 125,600 actor input tokens. Its early retrieval
selected prior SQL queries with `(no results)`, so `v11` is rejected as a
cross-benchmark candidate. The longer `v10` runs remain frozen and separate.

The completed `v9` Poker first-30 prefix had six valid suffix pairs. Five
tied, while one candidate hand lost 0.5 chips instead of native MemRL's
3.5-chip gain; mean reward was 0.8333 candidate versus 1.5 native. The
candidate's first action in the losing hand was FOLD; native continued for
three actions. Candidate retrieval added five prior public events, including
a previous FOLD/loss. The different context and divergent rollout are
associated with the regression, without isolating a single causal event.

`v12` filters public feedback with little substantive content and focuses
embedding similarity on the current question rather than repeated action
instructions. Its independent Database pilot was stopped after 4/20 cells:
the second candidate task retrieved the prior task's terminal incorrect-answer
feedback. The partial run has an operator-stop record and no scored suffix
claim. `v13` carries a generic `instance_complete` marker from the public
trajectory into each source-bound event and excludes terminal events from
actor-context retrieval; they remain in the event ledger for typed projection.
It retains the `v12` content filter and focused query. Its completed first-10
Database pilot had four suffix ties at 0.0, but used 35 actor calls and
171,408 input tokens versus 26 and 84,017 native. It is rejected on cost.
`v14` also excludes explicit tool-error feedback and allows up to two
distinct events from the same prior episode, still within the 2,048-token
cap. Its independent Database pilot has completed. These are development
diagnostics, not evidence of an all-domain improvement.
In its second development question, native submitted 4.15 and passed while
the candidate submitted 4.151729267385646 and failed. The official ground
truth was 4.11 for both arms; the accepted value fell just inside the
benchmark's 1% relative tolerance and the longer value just outside. This
is a scoring boundary, not a hidden-database mismatch, and the official
outcomes are left unchanged.

At the first 4 paired ALF `look_at_obj_in_light` games in the `v10` guardrail,
within-three success was 3/4 for both policies, with one win and one loss;
first-attempt success was 2/4 native versus 1/4 candidate. The first 13
scored Database pairs in the separate full `v10` run were 1 win, 1 loss, and
11 ties (candidate mean 0.0359 versus native 0.03077). These interim numbers
cannot establish improvement or safety over a full domain.

The full 30-episode `v10` Database repeat-303 run later completed: all 24
official suffix pairs were valid, with 2 wins, 2 losses, and 20 ties;
candidate mean 0.052779 versus native 0.050000. Candidate actor calls were
174 versus 240, and input tokens 905,784 versus 1,075,304. This small mean
gain needs an independent repeat; repeat 404 is reported below. The `v9` Poker
first-30 loss and `v10` ALF first-attempt regression remain cross-domain
counterexamples.
For the 24 paired Database suffix differences, a fixed-seed 20,000-sample
paired bootstrap gave a 95% percentile interval of about -0.1000 to
0.1056 for the mean difference. It crosses zero widely; the tiny positive
mean is not evidence of a stable Database improvement.

The full `v10` Database repeat-404 run then completed with all 24 suffix
pairs valid: candidate mean 0.083333 versus native 0.119450, with two wins,
four losses, and 18 ties. Candidate actor calls (172 versus 221) and input
tokens (803,158 versus 932,454) were lower, but reward regressed. The
20,000-sample paired bootstrap interval for the mean difference was about
-0.1222 to 0.0444. Taken together, the two repeats reject a reliable
Database reward improvement for `v10`; the separately frozen `v15`
structured-retrieval run is still in progress.

`v15` changes only the generic retry schedule for text-command tasks: keep
the exact native context for the first two attempts, then temporarily drop it
on the third attempt if both failed. It shares `v14` structured-event logic
and typed projection. A fresh look-family first-six paired pilot is running;
the first three games had native and candidate first-attempt success 2/3,
and within-three success 2/3 versus 3/3. On the third game, both native
attempts failed, and the candidate's third no-memory attempt succeeded in
seven steps. This is an early guardrail, not a full-family result.

The full 20-episode `v10` Cohort repeat-303 run completed with all 16
official suffix pairs valid and improved: candidate mean 0.084028 versus
native -0.022656, no losses. Both arms used 99 actor calls and 756,770
input tokens on scored pairs. Independent replay with the official scorer
matched all 20 saved final actions; 18 final actions differed from their
unprojected actor outputs. This confirms the typed projection's effect on
this repeat, while repeat 404 and other domains remain open.

The completed `v14` Database first-10 pilot had four official suffix ties at
0.0, no failures. Candidate actor calls were 27 versus 29 native, but input
tokens were 119,507 versus 107,248. The retrieval refinements did not show
reward improvement on this prefix.

The completed `v15` look-family first-six guardrail ended with within-three
success 4/6 candidate versus 3/6 native (2 wins, 1 loss, 3 ties), equal
first-attempt success 3/6, and fewer actor calls (452 versus 505) and input
tokens (2,458,223 versus 2,987,649). One later loss prevents a no-regression
claim. A separate `v15` clean-family first-six pilot and Poker first-30 pilot
are running to test whether the same version carries across interfaces.

`v16` uses the same structured-action rule as `v15`, but chooses the generic
text retry order from the first retrieval. If the first attempt had no prior
memory, a newly written failed lesson is withheld on attempt two and native
memory returns on attempt three. If prior memory was already available, the
first two attempts preserve native context and the third drops memory after
both fail. This uses retrieval state, not task names. Ten focused tests pass;
a fresh clean-family first-six pilot is running. It must be compared as an
independent online chain, not spliced into either `v10` or `v15` results.
The look-family first-six pilot has also started independently.

The completed `v15` clean-family first-six pilot regressed: within-three
success 3/6 candidate versus 4/6 native (0 wins, 1 loss, 5 ties), equal
first-attempt success 1/6, and more actor calls (596 versus 499). This
contradicts any claim that the fixed late-dropout schedule helps both ALF
families; `v16` specifically tests the initial-context condition.

The separate full 18-game `v10` look-family seed-92601 run completed with
within-three success 10/18 candidate versus 9/18 native (4 wins, 3 losses,
11 ties). First-attempt success fell to 4/18 from native 6/18, while actor
calls rose to 1,728 from 1,575. That version does not pass a broad
first-attempt and efficiency guardrail, despite its one-game net gain under
the three-attempt metric.

`v17` adds a first-attempt provenance gate to the `v16` retry rule: if every
retrieved native memory was written from a failed trajectory, the actor sees
no memory on that first attempt and those IDs receive no Q credit for being
shown. A mixed or success-only retrieval is unchanged; later retries retain
the initial-context schedule. This uses the stored official success flag,
not a task name or a guessed answer. Eleven focused tests pass. A fresh
clean-family first-six pilot is running; the observed correlation between
failure-only retrieval and negative transfer does not yet prove benefit.
The earlier disjoint-game evidence in `GENERAL_METHOD.md` found native
successes after retrieving only failure-tagged memories; suppressing such
memories regressed on unseen games. Therefore a success/failure-only gate
is not a credible general solution even if a short same-game pilot ties or
wins. `v17` is a diagnostic, not a selected final method.

The `v16` clean-family first-six run completed and regressed further:
within-three success 2/6 candidate versus 4/6 native (0 wins, 2 losses,
4 ties); first-attempt success 2/6 versus 3/6, with 626 versus 409 actor
calls. Every candidate first retrieval after the first game contained only
failed-trajectory memories, including both losing pairs. This is the
specific guardrail that `v17` tests; it does not validate the gate in advance.

The `v16` look-family first-six run completed as six exact outcome ties:
first-attempt and within-three success were 3/6 on both arms, with 504
actor calls each. Candidate input tokens were lower (2,722,016 versus
3,008,557). It provides no reward gain on that independent chain. A fresh
`v17` look-family first-six pilot has started to test the provenance gate
outside the clean family.

`v17` clean-family first-six then completed with within-three success 5/6
for both arms, but candidate first-attempt success 3/6 versus native 4/6
and actor calls 365 versus 288. Its failure-only gate never fired in that
online chain, so this result does not test the gate's effect. `v18` retains
the same general rules but isolates unassisted retries: when a deliberate
no-memory retry fails, its exact public trajectory and official failure
record remain, while the MemRL writer does not convert it into a future
retrievable lesson. Successful unassisted retries still write normally.
Twelve focused tests pass; a fresh clean-family first-six pilot is running.

The `v18` clean-family pilot was stopped after four cells (two complete
pairs): native within-three success 2/2, candidate 0/2. The skipped writer
appears to remove reflections needed by later retries in this chain, so
`v18` fails the cross-domain guardrail. Its look-family pilot was also
stopped, after eight cells, without a full-family claim. Both ignored
outputs retain their partial trajectories and operator-stop records.

`v19` replaces the skipped writer with a same-task quarantine. A failed
unassisted retry still writes its source-bound MemRL lesson, so a later
attempt on the same input can use it. The lesson remains in the audit store
but is removed from actor context and Q credit on different input bindings.
The rule is based on whether memory was shown, observed failure, and exact
input binding; it contains no family names. Twelve focused tests pass, and
a fresh clean-family first-six guardrail is running. This is unvalidated
until paired outcomes show that the quarantine helps without new losses.

The `v19` clean-family guardrail was stopped after three cells (one complete
pair): native succeeded within three attempts, candidate failed all three
with 150 actor actions. No old memory existed before that first game, so
the quarantine rule had not acted; the loss is a retry-schedule regression,
not evidence about the quarantine's causal effect. The partial output and
operator-stop record are preserved. `v19` is not selected.

The full `v15` Database repeat-303 run completed with all 24 official suffix
pairs valid: candidate mean 0.0750 versus native 0.016667, three wins,
21 ties, and no losses. Candidate actor calls were 181 versus 247 and
input tokens 986,229 versus 1,175,831. The paired bootstrap 95% interval
for the mean difference was 0.0 to 0.1417, reflecting the small number of
nonzero pairs. The separately frozen repeat-404 run has now completed; its
result is reported below.

The `v17` look-family first-six pilot completed as six outcome ties, with
first-attempt and within-three success 3/6 on both arms and 500 actor calls
each. The failure-only first-attempt gate fired once, on the sixth game;
both arms failed that game. The gate's observed use therefore did not create
a success in this prefix. The `v18` look-family pilot was later stopped after
eight cells because the same frozen method had already failed the clean-family
guardrail; its partial record and operator-stop marker remain preserved.

The completed `v15` Spectrum first-30 pilot had 12 valid official suffix
pairs: candidate mean 0.383133 versus native 0.215783, with 11 wins, one
tie, and no losses. Scored actor calls and input tokens were equal (12 and
24,210 per arm). Independent official re-scoring matched all 30 saved final
actions; 19 actions changed after typed projection. This still needs the
complete 90-episode result and an independent repeat. A full `v15` Database
repeat-303 run has started to test whether the structured readout transfers.

The completed `v15` Poker first-30 pilot had six valid official suffix
pairs: three wins and three ties, candidate mean 1.0833 versus native
-16.1667. The difference is dominated by one hand in which native lost
1,000 chips after 52 actor calls while the candidate folded for a 5-chip
loss after one call. The saved terminal public feedback matches both
official rewards. This small, heavy-tailed prefix is insufficient to claim
domain improvement; a full 120-hand repeat-303 run is underway.

The completed `v15` Cohort first-10 pilot had five jointly valid suffix
pairs, all wins (candidate 0.015749 versus native -0.011638). The sixth
planned suffix pair failed on both arms because neither raw actor output
could be repaired into a schema-valid JSON action. This shared failure
remains in the coverage record; it is not scored or silently dropped.
Scored actor calls and input tokens were identical. Independent official
re-scoring matched all nine completed candidate final actions, seven of
which changed after typed projection. A full-domain claim needs complete
coverage and another repeat.

The separate full 90-episode `v10` Spectrum repeat-303 run completed with
all 72 official suffix pairs valid and improved: candidate mean 0.379424
versus native 0.219029, 72 wins and no losses. Scored actor calls were equal
at 72, while candidate input tokens were 223,025 versus native 171,324.
Independent official re-scoring matched all 90 final actions; 82 changed
from the actor's raw action after typed projection. A 20,000-sample paired
bootstrap interval for the mean difference was approximately 0.1481 to
0.1731. Repeat 404 is still required before a cross-repeat claim.

The full `v4` queue entrypoint is `run_typed_grounded_full.py`, with
`report_typed_grounded_full.py` and `analyze_typed_grounded.py` for terminal
and per-job reports. It binds the base method, type projection, executor,
evaluator, and origin hashes. The separate `run_grounded_full.py` route remains
for `v3`. Short pilot outputs must state incomplete coverage and added actor
cost.

## Per-memory causal diagnostic, 2026-10-02

`credit_probe.py` now binds each read-only source episode to its input content,
original rows, retrieval, update, memory snapshot, individual retrieved texts,
and every candidate context by SHA-256. It records pre-action per-memory
retrieval scores, Q statistics, historical feedback, and provenance. It rejects
an existing case directory if any source binding changes. A fixed design has
passed reconstruction checks
for one case from each of the six ALFWorld families, with three paired actor
seeds and a leave-one-out comparison for the first retrieved memory. The
design and source bindings are saved under ignored
`results/memrl_credit_diagnostic/20261002_multifamily_per_memory_development_v3/`.
The earlier validation-only directories (`...development/` and
`...development_v2/`) were superseded before any rollout. An independent
GPU-6 actor service on port 18559 ran the paired ALFWorld replays independently
of the Database and Poker evaluation services. The read-only
`analyze_credit_probe.py` checks the frozen source, actor seed, actual
per-arm context and terminal reward before calculating each leave-one-out
difference. The selected source games belong to the canonical `valid_unseen` run and
include outcome-stratified cases, so they are **development diagnostics only**.
They cannot train a utility policy that is subsequently claimed as an
untouched `valid_unseen` evaluation. The completed probes do not by themselves
establish a memory-selection rule.

All six first-ranked memories in this diagnostic have `success=False` in their
prior snapshots. The first-index run completed all 18 paired replays and passed
the source, context, seed, and reward audit. Its overall mean leave-one-out
reward difference was 0: one positive pair for the pick-and-place case, one
negative pair for the pick-two case, and 16 ties. Each of these two nonzero
effects occurred in only one of three actor seeds for its case. This small,
outcome-stratified sample shows that failure-tagged memory can be helpful or
harmful in context; it does not estimate the rate of either effect.
Before the first-index run completed, `followup_design.json`
fixed leave-one-out probes for all eleven remaining retrieved memories, using
the same six games and three actor seeds. Those include both success- and
failure-tagged entries. This is a selected mechanism check, not an estimate
of how frequently each memory type helps in the benchmark.

The full `v15` Database repeat-404 run completed all 30 paired episodes and
24 official suffix pairs: candidate mean 0.1111125 versus native 0.0833375,
with four wins, two losses and 18 ties. Candidate actor calls were 196 versus
233, and input tokens 987,950 versus 1,036,418. The paired bootstrap 95%
interval for its mean difference is -0.0611 to 0.1194. Averaging repeats
303 and 404 gives +0.04305; resampling the 24 canonical task indices with
both actor repeats together gives a 95% interval of -0.0208 to 0.1222.
Thus the two observed repeat means are positive, but the evidence does not
establish reliable Database reward improvement. Across both repeats the
candidate used 377 versus 480 actor calls and 1,974,179 versus 2,212,249
input tokens on the common scored suffix, an observed cost reduction. A
separate full `v15` Spectrum repeat-303 run has started on the freed actor
service.
At its first 30 complete episodes, the full run's 12 scored suffix pairs
averaged 0.400808 versus 0.216242 (12 wins), whereas the separate first-30
pilot averaged 0.383133 versus 0.215783 (11 wins, one tie). The frozen method,
temperature, and first-30 input bindings match, but saved actor completions
and subsequent memory chains diverged between the independent runs. The pilot
is therefore not an exact prefix replay; the full run and a separate repeat
must carry the final inference.

The second-index development probe completed all 18 audited pairs. Its mean
leave-one-out reward difference was +0.2222: the pick-and-place failure-tagged
memory and the cool success-tagged memory each helped in all three actor
seeds, while look and pick-two each had one harmful seed; the other ten
pairs tied. The cool memory's native post-update Q value was -0.51 even though
its fixed-snapshot marginal effect was positive in 3/3 replays. This is
evidence of misassigned shared-trajectory credit in selected cases, not a
trained policy or a representative domain estimate. The predeclared third-index
probe later completed all 15 audited pairs: the pick-and-place success-tagged
memory helped in 3/3 seeds, the look success-tagged memory harmed in 1/3,
and eleven pairs tied. Across all three memory positions there were 51 audited
paired replays, ten positive differences, four negative differences, and 37
ties. Because the six source games were selected using earlier outcome signs,
these counts describe mechanism heterogeneity in selected cases, not a
general improvement rate or a training set.

`collect_credit_train.py` now prepares a separate native MemRL source chain
from 36 official ALFWorld `train` games, six per family, selected as the first
two historical training groups per family without using their rewards. The
prepared design under ignored
`results/memrl_credit_training/20261002_alf_train_native_source_v3/`
binds every game file, the source plan, and eight runtime source files. A
CPU-only six-family environment preflight passed. The collector is running
from this train-source design; the earlier `...native_source/` and
`...native_source_v2/` directories are superseded preparation-only artifacts.
The collector keeps ordinary failures,
interrupted partial cells, and explicit failure records rather than replaying
them silently. Later utility labels must be derived from newly bound
counterfactual rollouts, not from old episode IDs or unreviewed text targets.

A separate four-arm Poker probe completed on three first-20%-prefix
instances from the frozen CLBench run. Each source has exactly two retrieved
memories; three paired actor seeds execute full, none, drop-first, and
drop-second contexts. These are training-prefix development labels, not
last-80% evaluation results. `analyze_cl_credit_probe.py` checks the
actual first actor context, seed, saved official reward, and source binding
before reporting either memory's marginal effect and their interaction. All
nine four-arm comparisons passed that audit. Episode 4 showed positive
leave-one-out differences for both memories in all three actor seeds;
episode 3 had mixed signs and ties; episode 6 had one +60 reward difference
amid ties and small losses. The pooled means (+11.89 and +13.00 for the
two memory positions) are therefore heavily influenced by selected hands
and are not a robust Poker improvement estimate.

The full `v15` Poker repeat-303 run then completed all 120 episodes and 96
official suffix pairs, with no unscored pair. Candidate mean reward was
-0.682292 versus native -1.015625, a +0.333333 paired mean difference,
but pair counts were 22 wins, 23 losses, and 51 ties. A 20,000-sample paired
bootstrap interval was -1.1823 to +1.9167; the largest three positive
differences totaled +83.5 while the entire 96-pair net difference was only
+32. The 10%-trimmed mean difference was +0.1603 and the median difference
zero. Candidate actor calls were 224 versus 222, and input tokens 599,416
versus 511,585. This is a positive observed mean but not a reliable Poker
improvement or cost win. A full `v15` Cohort repeat-303 run has started to
cover the fourth CLBench domain.

The separate ALFWorld train-source collector has started on its prepared
36-game design. It produces native MemRL and no-memory trajectories from
empty family-specific banks, preserving the official action and retry budgets.
These new trajectories are bound to their actual train-game content and have
no reused supervision targets. Their later counterfactual utility labels
must be reviewed against the new input bindings before fitting a selector.
Before collection completed, `probe_selection_rule.json` and the SHA-256 of
`select_credit_train_probes.py` froze an outcome-blind choice of the earliest
and latest eligible episode with at least two memories per family, with actor
seeds 92741–92743. The selector requires a complete source chain and rejects
changed plans, designs, or code. The paired label audit checks that each
probed case and actor seed matches this frozen choice, then verifies the
train-game bytes, retrieved context, memory snapshot, actor seed, and terminal
reward.

The old CLBench MemRL calibration prefixes reveal a coverage gap in the
initial two-memory probe: Spectrum, Database, and Cohort have no completed
prefix case with two retrieved memories. `credit_probe.py` and
`analyze_cl_credit_probe.py` now handle a singleton by comparing full context
with no memory under the same actor seed; multi-memory probes still remove
one item at a time. An outcome-blind, source-bound selection from repeat 404
is frozen in ignored
`results/memrl_credit_training/20261002_cl_prefix_probe_selection.json`:
earliest and latest eligible memory-bearing prefix case per domain, plus the
first multi-memory case where available, nine cases across all four domains.
Three actor seeds and exact source/query/snapshot hashes were fixed before any
new CL probe rollout. The nine cases passed reconstruction and binding checks.

The independent full `v15` Spectrum repeat-303 run has now completed all 90
paired episodes and 72 official scored suffix pairs. Candidate reward was
0.377665 versus native MemRL 0.218482, with 70 wins, no losses, and two ties.
Actor calls and actor input tokens were identical on scored pairs (72 calls
and 171,222 input tokens per arm). An independent official scorer reproduced
the saved final reward for all 90 candidate actions; 78 final actions differed
from the actor's raw submission. A 20,000-draw paired episode bootstrap gave
a descriptive 95% interval of +0.14497 to +0.17375; a circular 10-episode
block bootstrap gave +0.13139 to +0.18371. Neither resampling method creates
an independent online memory chain. A separate full repeat-404 run was started
on the freed actor service to test repeatability. These results establish a
strong Spectrum gain for this policy, not a four-domain or ALFWorld gain.

The repeat-404 full Spectrum run has also finished all 90 paired episodes and
72 scored suffix pairs. Candidate reward was 0.379990 versus native 0.218881,
a +0.161110 difference with 72 wins, no losses, and no ties. All 90 saved
candidate final actions were independently re-scored; 82 differed from the
raw actor submission. Pooling the two repeat descriptions gives 144 scored
pairs, mean difference +0.160147, and 142 wins / 0 losses / 2 ties. The two
online chains are the independent repeat units; the 144 task pairs should not
be treated as 144 independent memory-evolution runs. A 20,000-draw bootstrap
of the 72 canonical suffix indices with both repeats together gave a
descriptive 95% interval of +0.14667 to +0.17366; it does not add online
chains.

The repeat-303 full Cohort run completed all 20 episodes. Its 16 official
suffix pairs had candidate mean 0.073464 versus native 0.025128, difference
+0.048336, with 13 wins and 3 losses. One prefix episode failed under both
policies because neither produced a schema-valid action; this failure is now
reported separately as `prefix_failures` by `analyze_typed_grounded.py`.
Nineteen completed candidate final actions were independently re-scored and
matched the recorded rewards. The repeat-404 full run also completed all
20 episodes; its 16 scored suffix pairs averaged 0.047219 for candidate and
-0.003495 for native, a +0.050713 difference with 12 wins and 4 losses.
All 20 candidate final actions were independently re-scored. Across two
online chains, the 32 scored pairs have mean difference +0.049524 and
25 wins / 7 losses. A 20,000-draw bootstrap resampling the 16 canonical
suffix indices *with both repeats together* gave a descriptive 95% interval
of +0.02227 to +0.08121. The two online chains remain the independent repeat
units; this interval does not add more chains.

The frozen CLBench repeat-404 calibration-prefix credit probe completed all
27 case-by-actor-seed comparisons. Its 60 branch rewards match the official
runner outcome records; 24 final Spectrum/Cohort actions were independently
re-scored. Spectrum's two selected single-memory cases had six zero paired
differences. Poker's three cases had four positive, three negative and five
zero per-memory seed labels, with heavy-tailed returns. Database had one
negative and five zero labels; Cohort had four positive and two negative.
These counts are deliberately selected *training-prefix* diagnostics, not
domain-level improvement rates. A second outcome-blind prefix selection from
repeat 303 has seven cases in three domains; Cohort had no completed
memory-bearing source in its first 20%, so no Cohort label was fabricated.

The separate ALFWorld train source completed 72/72 native/no-memory cells,
and `audit_credit_train_source.py` verified all 36 official-train game pairs.
Native first-attempt success was 9/36 versus no-memory 7/36, with mixed
directions by family. The outcome-blind selector chose 12 cases spanning all
six families, with 31 retrieved memory positions. First-position deletion
probes completed all 36 paired actor seeds: 3 positive, 1 negative and 32
ties. All selected positions from both outcome-blind train waves later
completed and passed their source audits. The CPU-only paired-credit model
fitted to all ALFWorld positions plus both CLBench calibration-prefix probes
has 78 bound memory examples and removed zero memories in input-content-held-
out cross-fitting; its error exceeded a zero predictor. That is insufficient
evidence to claim a useful credit-aware retrieval policy. The separate
full-context-versus-empty fixed-snapshot probe completed 66/66 ALF train
pairs: 17 positive, 10 negative, 39 tied. A later audit found that 30
of those pairs crossed actor services, so this aggregate is **not** a
valid causal estimate of memory utility. The 36 same-service pairs had
3 positive, 4 negative and 29 ties. See `PAIRED_CREDIT_RL.md` for the
source audit and repair plan.

The later `v15` Poker repeat-404 full chain exposed a severe development
failure at hand 46. Native MemRL completed the hand with -5 chips; the
candidate raised repeatedly from its second action through a 52-action hand
and received the official -100 reward. All 52 type-directed audits recorded
`KEEP` because no type-compatible historical candidate existed. The chains
have different prior memory histories, so this comparison alone does not
isolate the injected event text as the cause. `v20` adds an interface-general
rule: after the first structured action has no type-compatible candidate,
remove only the extra unverified public-event excerpt from later actor calls;
keep native MemRL text and the source-bound event ledger for future projection.
In a frozen-snapshot continuation from the `v15` candidate's state just before
hand 46, the first action remained CALL; later actions changed from repeated
RAISE to CALL, and the official reward was -5 in five calls. This is one
mechanism replay, not an independent online-chain gain. A full independent
`v20` Poker repeat-303 chain started while `v15` repeat-404 was still running.

That `v20` Poker-303 online chain later reached its own hand 46. The first
action had no type-compatible projection and the v20 withdrawal fired, yet
the candidate raised repeatedly and received -99.5 in 38 actor calls; native
MemRL received -5 in five calls. Runner outcome records, row rewards, and
post-task memory hashes agree. In the older `v15` repeat-303 chain, both
arms also raised repeatedly on hand 46 and both lost heavily, so this is a
different online history, not proof that one text source alone caused the
loss. It is a concrete counterexample to treating the v20 fixed-snapshot
rescue of Poker-404 hand 46 as a general Poker solution. `v21` therefore
tests a stronger interface-general continuation: if the first structured
action has no compatible typed candidate, later turns omit all historical
actor text while preserving the source-bound event ledger. Only mechanism
tests have passed; its paired online outcome remains unmeasured.

The `v15` Poker repeat-404 full run has now completed all 120 paired hands
without failures. On the 96 official suffix pairs, candidate reward was
-2.343750 versus native -1.109375, a -1.234375 difference with 23 wins,
24 losses, and 49 ties. Candidate actor calls were 324 versus 262 native;
input tokens were 2,279,531 versus 1,371,821. Repeat 303 had been +0.333333,
so the two online chains pooled to -0.450521 over 192 scored pairs. A
canonical-hand-index bootstrap interval of [-1.815104, +0.622396] crosses
zero and cannot account for uncertainty from only two online memory chains.
The worst candidate-specific hand in repeat 404 was hand 46, -95 relative
to native. The full result rejects `v15` as a reliable Poker improvement.

The `v20` Cohort repeat-404 full run also completed all 20 pairs, with 16/16
official suffix pairs valid: candidate 0.072928 versus native -0.020234,
15 wins and one loss, no failures. Independent official re-scoring matched
all 20 final candidate actions, 18 of which changed from the raw actor action.
Actor calls and input tokens were equal across arms on scored pairs (112
calls and 1,093,277 input tokens each). The v20 context-withdrawal rule did
not fire in this run, so the gain is attributable to the typed action path.

In a fixed-snapshot continuation from the `v20` Poker-303 state just before
hand 46, `v21` kept the same first CALL, withdrew all historical actor text,
then used three CALLs and a FOLD. Official reward was -4 in five calls,
versus -99.5 in 38 calls for the original v20 online hand and -5 for its
native arm. This is still a single selected development hand, not evidence
that v21 improves all Poker hands. Full v21 Poker-404 and Cohort-404 online
chains have started to test reward and negative transfer.

The `v20` Poker repeat-303 full chain then completed 120/120 hands without
failed cells. On all 96 official suffix pairs it regressed by -1.192708:
candidate -1.729167 versus native -0.536458. It won 22 hands, lost 16 and
tied 58, but three largest negative differences summed to -157 chips; the
official mean is sensitive to those large losses. Candidate calls increased
from 204 to 256 and input tokens from 484,986 to 959,739. The event-only
withdrawal is rejected as a Poker solution despite the single-hand rescue.
`v21` now runs independent full Poker chains for both repeats 303 and 404.

An additional `v22` diagnostic tested a generic loop recheck when three
consecutive structured actions had the same kind and a strictly increasing
numeric argument. On a frozen continuation of `v21` Poker-404 hand 40, the
recheck changed the third RAISE to CALL and ended the hand in five actor calls
with official reward +6. Crucially, the original `v21` online hand ultimately
earned +100 despite 51 calls, while native earned +2. The recheck therefore
reduced calls but lost 94 reward relative to the candidate on this hand.
Long escalating action sequences cannot be treated as failures from shape
alone. `v22` remains a mechanism diagnostic and was not promoted to a full
benchmark candidate.

The independent `v21` Cohort repeat-404 chain completed all 20 pairs. Its
16 official suffix pairs scored 0.069768 for the candidate versus -0.020234
for native MemRL, a +0.090002 difference (14 wins, two losses), with no
failed cells. Independent official re-scoring matched all 20 saved final
actions; 18 changed from the raw actor action. Both arms used 112 actor calls
and 1,093,277 input tokens on the scored suffix. This supports the typed
projection path on this repeat, not the continuation-withdrawal rule alone.

The independent `v21` Database repeat-404 chain completed all 30 pairs.
All 24 official suffix pairs were valid: candidate 0.108329 versus native
0.058333, a +0.049996 difference (three wins, one loss, 20 ties), with no
failed cells. Actor calls increased from 211 to 234 on the scored suffix;
input tokens were 869,571 candidate versus 887,328 native. This is a
positive repeat with sparse non-tied outcomes, so robustness is still
uncertain. Full `v21` Poker repeats 303/404 and Spectrum repeat 404 are
running on their previously allocated actor services; they have no final
outcome claim yet.

The `v15` ALFWorld clean-family first-six guardrail is a concrete warning for
the shared text branch inherited by `v21`: native MemRL solved four of six
games within three attempts, while the candidate solved three. On game four,
native succeeded on retry two after ten actions; the candidate used the
full 50-action cap on each of three attempts and never succeeded. The arms
had different online memory histories and retrieved different memories, so
this comparison establishes negative transfer in the observed chain, not
a causal effect of any one event. A full six-family ALFWorld benefit remains
unshown.

Cross-version scores must not be read as a direct policy A/B test. For
example, the native MemRL control arm's saved rewards differed on all 20
Cohort-404 episodes between the independent `v15` and `v21` runs; the actor
services also differed. The task plan and repeat number matched, but
stochastic generation and online memory evolution did not yield an identical
control trajectory. The supported comparisons are each run's within-run
paired rewards, failures, and costs.

`v21` Poker repeat 404 completed all 120 paired hands without failures. On
the 96 official suffix hands, candidate mean reward was +1.765625 chips
versus -1.312500 native, a paired difference of +3.078125. It won 25 hands,
lost 23, and tied 48. Actor calls were 275 candidate versus 268 native;
input tokens were 1,170,823 versus 1,327,503. The median paired difference
was zero and the three largest wins summed to +285.5 chips, versus -88.0
for the three largest losses. A descriptive shared-hand-index bootstrap
interval was [-0.234375, +7.062500], crossing zero; it does not account
for variation between independent online chains. One large favorable hand
was index 115: candidate folded for -0.5 in one call while native ran 50
calls and received -100. A second repeat is required before a robust Poker
claim. The first independent `v21` ALFWorld clean-family run, repeat 92602,
started on the released actor service; no ALF result is claimed yet.

`v21` Poker repeat 303 also completed all 120 pairs without failed cells.
Its 96 official suffix hands scored +0.312500 candidate versus +0.203125
native, a +0.109375 paired difference (25 wins, 18 losses, 53 ties).
Candidate actor calls were 208 versus 195 native; input tokens were
333,766 versus 446,834. Pooling the two same-implementation repeats gives
192 scored hands with mean difference +1.593750, median zero, 50 wins,
41 losses and 101 ties. The three largest positive differences total
+285.5 chips and the three largest negative differences -97.0. A
descriptive shared-hand-index bootstrap interval of [-0.276042,+3.729167]
crosses zero, and only two independent online chains are available. This
is a promising direction on Poker, not a robust all-domain conclusion.
`v21` Database repeat 303 has started on the released GPU3 actor service.

`v21` Spectrum repeat 404 completed all 90 pairs without failed cells. On
the 72 official suffix pairs, candidate score was 0.379572 versus 0.218065
native, a +0.161507 difference: 72 wins, no losses or ties. Both arms used
72 actor calls; candidate input was 171,388 tokens versus 171,530 native.
An independent invocation of the official scorer matched all 90 saved
candidate final actions. On the scored suffix, all 72 final actions differed
from the raw actor action through `RECURRING_RECORDS`; the raw-action mean
was 0.219110 and the final-action mean was 0.379572. This directly verifies
the projection's per-action gain on this run: every one of the 72 scored
raw-to-final differences was positive (minimum +0.0723). Full `v21` Spectrum repeat 303
has started on the released GPU6 actor service.

`v21` Database repeat 303 completed all 30 pairs with 24/24 official
suffix pairs valid and no failed cells. Candidate reward was 0.058333
versus native 0.083333, a -0.025000 difference (two wins, three losses,
19 ties). Actor calls were equal at 215; candidate input tokens were
722,385 versus 954,447 native. Repeat 404 was +0.049996, so this domain
has mixed directions across its two v21 online chains. The pooled mean
difference is +0.012498. A descriptive common-index bootstrap interval
[-0.043069,+0.072223] crosses zero and does not represent extra online
chains, so it does not establish reliable improvement.
`v21` Cohort repeat 303 has started on the released GPU3 actor service.

In all nine non-tied Database suffix pairs across repeats 303 and 404,
the candidate's first saved actor action already differed from native
MemRL's. Every candidate audit on those cases was `KEEP` with no compatible
typed projection, and the v21 continuation withdrawal fired. These saved
traces locate the difference in the actor trajectory; they cannot identify
whether later context withdrawal helped or hurt because the arms' first
actions and online memory histories had already diverged. Database needs a
fixed-snapshot intervention before assigning causal credit to that rule.

The `v21` ALFWorld clean-family repeat-92602 full chain completed all 31
paired valid-unseen games without failed cells. Native MemRL succeeded
within three attempts on 16/31; candidate succeeded on 10/31, a -6/31
difference (one candidate-only win, seven candidate-only losses, 23 ties).
First-attempt success was 14/31 native versus 7/31 candidate (one win,
eight losses, 22 ties). Candidate actor calls increased from 2,686 to
3,490 and input tokens from 18,615,290 to 23,428,685. This full-family
negative transfer rejects `v21` as the requested universal ALFWorld plus
CLBench solution. The v21 structured-action gains remain separately valid
within their runs; the inherited ALF text branch needs a new train-grounded
credit and retrieval policy. Later successful CLBench repeats cannot erase
this ALF failure. Within this one online chain, a 20,000-draw game-level paired bootstrap
interval for the -6/31 difference is [-0.354839,-0.032258]; it cannot
replace a second independent chain. The later full look-family run also
regressed, as detailed below.

`v21` Cohort repeat 303 completed all 20 planned pairs. A shared prefix
failure occurred at episode 2: both arms exhausted format/packaging repair
without a schema-valid JSON action. It remains recorded and is outside the
official last-80% suffix. All 16 suffix pairs were valid: candidate mean
0.077494 versus native 0.002244, a +0.075249 difference (16 wins, no
losses or ties). Both arms used 102 actor calls and 792,189 input tokens
on scored pairs. Independent official re-scoring matched all 19 completed
candidate final actions; every scored action changed through
`NUMERIC_CONSENSUS` and improved on its raw actor action. The raw-action
mean on the suffix was 0.002244 and the final mean 0.077494. Together
with repeat 404 (+0.090002), this is a consistent Cohort projection gain,
subject to the explicitly retained shared prefix failure.

`v21` Spectrum repeat 303 then completed all 90 pairs with 72/72 official
suffix pairs valid: candidate 0.376361 versus native 0.218482, a
+0.157879 difference (70 wins, no losses, two ties), with no failures.
Both arms used 72 actor calls; candidate input was 171,093 tokens versus
171,222 native. Independent official re-scoring matched all 90 final
actions. On the scored suffix, 70 `RECURRING_RECORDS` changes raised the
official action score and two actions tied; the raw-action mean was
0.218482, exactly the native mean for this run, versus final 0.376361.
Pooling both Spectrum repeats gives 144 scored pairs, +0.159693 mean
difference, 142 wins, zero losses, two ties. A descriptive common-index
bootstrap interval is [+0.146535,+0.172960]. Both repeats and the
raw-to-final official scoring support a strong Spectrum projection gain.

Across both Cohort repeats, the pooled 32 scored pairs have +0.082626
mean difference, 30 wins and two losses. A descriptive common-index
bootstrap interval is [+0.055200,+0.112215]. The shared prefix schema
failure in repeat 303 remains part of the full failure record.

All four CLBench domains now have two complete `v21` online repeats:
Spectrum and Cohort show consistent projection gains; Poker has two
positive means but high-variance chip tails and a descriptive interval
crossing zero; Database has one positive and one negative repeat and a
descriptive interval crossing zero. No CLBench rewards are averaged across
domains. The full ALFWorld clean-family negative result independently
rules out the requested universal improvement.

The full `v21` ALFWorld look-family repeat-92602 chain also completed all
18 paired valid-unseen games without failed cells. Native MemRL succeeded
within three attempts on 11/18 and candidate on 9/18 (two candidate-only
wins, four candidate-only losses, 12 ties). First-attempt success was 8/18
native versus 4/18 candidate (one win, five losses, 12 ties). Candidate
actor calls increased from 1,468 to 1,891; input tokens from 9,100,800
to 11,200,249. Thus both independently completed ALF families regress,
while only two of six families have full v21 coverage. The early positive
look-family short pilot did not survive this complete repeat.
