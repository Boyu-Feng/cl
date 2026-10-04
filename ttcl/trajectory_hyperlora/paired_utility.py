"""Paired, trajectory-level utility for training a trajectory-to-LoRA generator.

The collector must execute the same future task with the same environment seed
and action budget for every arm. This module only validates and scores those
recorded outcomes; it does not turn a failed episode into bad-action labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import statistics


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PairedOutcome:
    source_hash: str
    query_hash: str
    environment_seed: int
    action_budget: int
    candidate_reward: float
    base_reward: float
    wrong_source_reward: float

    def validate(self) -> None:
        for name in ("source_hash", "query_hash"):
            value = getattr(self, name)
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError(f"Invalid {name}")
        if self.action_budget < 1 or self.environment_seed < 0:
            raise ValueError("Seed and action budget must be valid")
        for name in ("candidate_reward", "base_reward", "wrong_source_reward"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"Missing or nonfinite {name}; never substitute zero")


def trajectory_utility(
    outcomes: list[PairedOutcome], *, scale: float = 1.0,
    harm_weight: float = 0.5,
) -> dict[str, float]:
    """Score one generated adapter on a fixed, source-disjoint probe grid.

    Signed candidate-minus-base reward is the main learning signal. A downside
    term discourages occasional catastrophic transfer. Wrong-source utility is
    diagnostic only: beating a bad adapter is not a benefit over the base.
    All scales/weights must be frozen using train
    data before evaluation, never tuned on benchmark test rewards.
    """
    if not outcomes:
        raise ValueError("An empty probe grid is not evidence of utility")
    if not math.isfinite(scale) or scale < 1:
        raise ValueError("Training-frozen scale must be finite and >= 1")
    if not (math.isfinite(harm_weight) and harm_weight >= 0):
        raise ValueError("Penalty weights must be finite and nonnegative")
    keys = set()
    for row in outcomes:
        row.validate()
        key = (row.query_hash, row.environment_seed)
        if key in keys:
            raise ValueError("Duplicate future-task/seed pair")
        keys.add(key)
        if row.source_hash == row.query_hash:
            raise ValueError("Source and future query must have distinct content")
    source_hashes = {row.source_hash for row in outcomes}
    if len(source_hashes) != 1:
        raise ValueError("One utility estimate must refer to one source trajectory")
    budgets = {row.action_budget for row in outcomes}
    if len(budgets) != 1:
        raise ValueError("Paired arms must use the same declared action budget")

    deltas = [row.candidate_reward - row.base_reward for row in outcomes]
    wrong_deltas = [row.candidate_reward - row.wrong_source_reward for row in outcomes]
    raw_mean = statistics.mean(deltas)
    downside = statistics.mean(max(0.0, -value) for value in deltas)
    specificity = statistics.mean(wrong_deltas)
    return {
        "mean_delta_base": raw_mean,
        "mean_downside": downside,
        "mean_delta_wrong": specificity,
        "utility": (raw_mean - harm_weight * downside) / scale,
        "n": len(outcomes),
    }


def reinforce_loss(log_probability, utility: float, baseline: float = 0.0):
    """Policy-gradient loss for a sampled latent adapter, not actor tokens.

    ``log_probability`` is the log probability of the sampled latent code under
    the generator. The actor and the environment can remain non-differentiable.
    A predeclared baseline reduces variance but must not erase signed harm.
    """
    if not (math.isfinite(utility) and math.isfinite(baseline)):
        raise ValueError("Utility and baseline must be finite")
    return -(utility - baseline) * log_probability
