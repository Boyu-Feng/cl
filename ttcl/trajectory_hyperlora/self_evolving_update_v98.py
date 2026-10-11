"""Inference-time self-evolving trajectory-to-LoRA updater.

Unlike the v94 rule, this module has no fitted updater checkpoint.  The rule
starts randomly initialized.  During inference it accumulates a recurrent
trajectory state and updates its own parameters only from actions followed by
an environment reward.  Failed episodes still update the state, but never
create guessed action labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from .persistent_update_v95 import PersistentUpdateRule


@dataclass
class SelfEvolutionState:
    memory: torch.Tensor
    b: list[torch.Tensor]
    positive_rows: list[dict]
    episodes_seen: int = 0


class SelfEvolvingUpdater:
    """Online learner for a randomly initialized trajectory update rule.

    ``loss_fn`` must evaluate the base model on one or more reward-confirmed
    transitions and return a scalar action cross-entropy.  It is deliberately
    injected so the protocol can use the exact benchmark model without making
    the updater depend on an environment implementation.
    """

    def __init__(self, rule: PersistentUpdateRule, aa: Sequence[torch.Tensor],
                 base_norms: Sequence[float], lr: float = 1e-4,
                 replay_size: int = 8):
        self.rule = rule
        self.aa = list(aa)
        self.base_norms = list(base_norms)
        self.optimizer = torch.optim.AdamW(rule.parameters(), lr=lr, weight_decay=0.0)
        self.replay_size = replay_size

    def initial_state(self, b: Sequence[torch.Tensor]) -> SelfEvolutionState:
        return SelfEvolutionState(
            memory=torch.zeros(1, self.rule.memory.hidden_size, device=b[0].device),
            b=[x.detach().clone() for x in b], positive_rows=[])

    def propose(self, state: SelfEvolutionState, episode: Sequence[dict]):
        memory, b, gate, confidence = self.rule.update(
            state.memory, episode, state.b, self.aa, self.base_norms)
        return SelfEvolutionState(memory.detach(), [x.detach() for x in b],
                                  state.positive_rows, state.episodes_seen), gate, confidence

    def observe_and_learn(self, state, episode, loss_fn):
        """Consume one completed episode, then commit one residual update.

        Train only on transitions with observed positive reward. Recomputing
        from the pre-episode state after optimizer.step is not a second commit.
        A reward-free episode changes memory but supplies no action labels.
        """
        rows = [r for r in episode if float(r.get("reward", 0.0)) > 0]
        replay = (state.positive_rows + rows)[-self.replay_size:]
        info = {"updated": False, "reason": "no_reward"}
        if rows:
            self.optimizer.zero_grad(set_to_none=True)
            _, candidate, _, _ = self.rule.update(
                state.memory.detach(), episode, [b.detach() for b in state.b],
                self.aa, self.base_norms)
            loss = loss_fn(candidate, replay)
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError("loss_fn must return a finite scalar")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.rule.parameters(), 1.0,
                                           error_if_nonfinite=True)
            self.optimizer.step()
            info = {"updated": True, "loss": float(loss.detach()),
                    "replay_rows": len(replay)}
        with torch.no_grad():
            committed, gate, confidence = self.propose(state, episode)
        committed.positive_rows = replay
        committed.episodes_seen = state.episodes_seen + 1
        info.update(gate=float(gate), confidence=float(confidence))
        return committed, info
