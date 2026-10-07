"""Environment-independent paired policy gradient for trajectory-conditioned LoRA.

An actor adapter owns action sampling and recomputes the log probability of
the sampled action sequence with gradients through its trajectory-to-LoRA
generator. An environment adapter owns resets, rewards and budget accounting.
Neither action labels nor benchmark-specific trajectory slots occur here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Protocol

import torch


def trajectory_text(events: list[dict[str, Any]]) -> str:
    """Serialize observable event fields without interpreting task semantics."""
    if not events:
        raise ValueError("A source trajectory needs at least one event")
    return "\n".join(json.dumps(event, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")) for event in events)


def content_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class PairedCase:
    source_events: list[dict[str, Any]]
    target_id: str
    target_content_sha256: str
    reset_seed: int
    split: str

    def __post_init__(self) -> None:
        if self.split != "train":
            raise ValueError("Only reviewed training targets may update the generator")
        if not self.target_id or len(self.target_content_sha256) != 64:
            raise ValueError("Target needs an identifier and content binding")
        trajectory_text(self.source_events)


@dataclass(frozen=True)
class Episode:
    target_id: str
    target_content_sha256: str
    reset_seed: int
    reward: float
    status: str
    decisions: tuple[Any, ...]
    budget_used: int


class PairedActor(Protocol):
    def rollout(self, case: PairedCase, source_text: str | None,
                *, sample_seed: int) -> Episode: ...

    def action_log_probability(self, episode: Episode,
                               source_text: str) -> torch.Tensor: ...


def _validate(case: PairedCase, episode: Episode, max_budget: int) -> None:
    if (episode.target_id != case.target_id or
            episode.target_content_sha256 != case.target_content_sha256 or
            episode.reset_seed != case.reset_seed):
        raise ValueError("Rollout target or reset differs from reviewed pair")
    if episode.status != "complete":
        raise ValueError("Incomplete rollout must be preserved, not trained on")
    if not 0 <= episode.budget_used <= max_budget:
        raise ValueError("Rollout exceeded declared budget")
    if not episode.decisions:
        raise ValueError("Rollout has no sampled decisions")
    if not torch.isfinite(torch.tensor(episode.reward)):
        raise ValueError("Non-finite environment reward")


def paired_episode_policy_gradient(actor: PairedActor, case: PairedCase,
                                   *, sample_seed: int, max_budget: int,
                                   baseline_reward: float | None = None
                                   ) -> tuple[torch.Tensor, dict[str, Any]]:
    """One on-policy update sample with a same-reset no-LoRA reward baseline.

    The adapter must sample actions from its LoRA-conditioned policy, then
    recompute their joint log probability under that *same* sampling policy.
    Action constraints, temperature and stop decisions belong to the adapter.
    Environment calls are non-differentiable; gradients flow through the
    actor's token/action log probabilities into its LoRA generator.
    """
    if max_budget < 1:
        raise ValueError("A positive declared budget is required")
    source = trajectory_text(case.source_events)
    # Keep the environment reset paired, but use a separate action-sampling
    # stream. Identical streams can give zero advantage everywhere when the
    # initial adapter equals the base policy.
    baseline_seed = sample_seed ^ 0x5DEECE66D
    baseline = actor.rollout(case, None, sample_seed=baseline_seed)
    sampled = actor.rollout(case, source, sample_seed=sample_seed)
    _validate(case, baseline, max_budget)
    _validate(case, sampled, max_budget)
    log_probability = actor.action_log_probability(sampled, source)
    if log_probability.ndim != 0 or not torch.isfinite(log_probability):
        raise ValueError("Actor must return a finite scalar joint log probability")
    reference = baseline.reward if baseline_reward is None else baseline_reward
    advantage = sampled.reward - reference
    loss = -float(advantage) * log_probability
    record = {"target_id": case.target_id,
              "target_content_sha256": case.target_content_sha256,
              "source_content_sha256": content_sha256(case.source_events),
              "reset_seed": case.reset_seed, "sample_seed": sample_seed,
              "baseline_sample_seed": baseline_seed,
              "baseline_reward": baseline.reward,
              "adapter_reward": sampled.reward,
              "advantage": advantage, "baseline_budget": baseline.budget_used,
              "adapter_budget": sampled.budget_used,
              "status": "complete"}
    return loss, record
