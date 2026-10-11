# Parameterized experience methods relevant to trajectory learning

## Closest prior work

1. **Text-to-LoRA** maps a natural-language task description to a LoRA in one
   forward pass. It is the closest architectural reference, but its input is a
   task description rather than a reward-labelled failure trajectory.
2. **HyperLoRA** uses hypernetwork pretraining plus sparse-task
   generalization fine-tuning and constrained losses. The constrained objective
   and held-out task split are useful for stabilizing our generated `B`.
3. **SHINE** maps diverse context to LoRA and explicitly aims to turn
   in-context knowledge into parameters. Its context encoder and frozen
   generator setup are relevant to transfer, but it does not establish online
   credit assignment from an agent's own reward.
4. **LoraHub** composes a bank of LoRAs with task-dependent coefficients. This
   suggests generating coefficients over a fixed experience basis instead of
   predicting every `B` entry freely, which should make failure/success
   comparisons more identifiable.
5. **Profile-to-PEFT** maps a persistent profile/history to adapter parameters.
   Its history-conditioned interface is close to a persistent experience state,
   although the target is user preference rather than environment reward.
6. **Meta-UCF** uses task embeddings, a single LoRA generator and contrastive
   plus orthogonality objectives. The orthogonality idea is useful for keeping
   distinct experience directions from collapsing into the same adapter.

## Recommended design for this project

Do not update the hypernetwork online and do not directly learn a fresh full
`B` from every trajectory. Train one frozen generator on target-disjoint
external tasks, with an output of the form

```text
z_t = recurrent_experience_encoder(trajectory_t, reward_t)
alpha_t = coefficient_head(z_t, query_t)
B_t = sum_k alpha_t[k] * basis_B[k]
```

At inference, update only a small persistent experience state `z_memory` or the
coefficient vector `alpha`; the generator and basis remain frozen. The update
must be reward-confirmed and trained on a future-query objective, not on
reconstructing the old action. A failed trajectory should be allowed to produce
an explicit abstention/negative gate rather than an unconstrained residual.

## Required controls

- direct online LoRA with the same parameter and update budget;
- frozen generator with no state update;
- frozen generator with state/coefficient update;
- shuffled trajectory-to-reward bindings;
- random basis or random generator;
- success-only, failure-only and success-minus-failure state updates;
- first-attempt and cumulative scores separately.

The claim “trajectory experience was learned” is supported only if the stateful
frozen generator beats direct LoRA on at least two unseen domains and the gain
disappears under shuffled reward bindings. The current external CLBench transfer
run is a baseline for this claim; its HLE slice has not shown a gain so far.
