"""Bounded correction to a persistent LoRA B from same-task outcomes.

A failed episode alone has no signed update. If the same frozen actor solves
the same reset without the current LoRA, the successful and failed trajectories
form a contrast. Their generated factors propose a direction; the step is
limited relative to the current persistent B norm. This is a heuristic to test,
not a claim that subtracting a failure factor is generally beneficial.
"""

from __future__ import annotations

import math

import torch


def factor_norm(factors: list[torch.Tensor]) -> float:
    return math.sqrt(sum(float(x.detach().float().square().sum())
                         for x in factors))


def contrastive_failure_update(
    previous: list[torch.Tensor] | None,
    successful: list[torch.Tensor],
    failed: list[torch.Tensor],
    *, step_size: float = .25,
    relative_cap: float = .25,
) -> tuple[list[torch.Tensor], dict[str, float]]:
    """Take a norm-bounded step along B(success) minus B(failure).

    The caller must verify both outcomes came from the same task reset and
    that the successful episode won while the current-LoRA episode failed.
    """
    if not 0 < step_size <= 1 or not 0 < relative_cap <= 1:
        raise ValueError("Invalid contrastive step or cap")
    if not successful or len(successful) != len(failed) or (
        previous is not None and len(previous) != len(successful)
    ):
        raise ValueError("Changed LoRA layer count")
    values = []
    for i, (good, bad) in enumerate(zip(successful, failed, strict=True)):
        good = good.detach().cpu().float()
        bad = bad.detach().cpu().float()
        if good.shape != bad.shape or (
            previous is not None and previous[i].shape != good.shape
        ) or not bool(torch.isfinite(good).all() and torch.isfinite(bad).all()):
            raise ValueError("Changed or nonfinite LoRA factors")
        values.append((good, bad))
    old = ([torch.zeros_like(good) for good, _ in values] if previous is None
           else [x.detach().cpu().float() for x in previous])
    if not all(bool(torch.isfinite(x).all()) for x in old):
        raise ValueError("Nonfinite persistent LoRA factors")
    direction = [good - bad for good, bad in values]
    proposed = [step_size * x for x in direction]
    proposed_norm = factor_norm(proposed)
    reference_norm = (factor_norm(old) if previous is not None else
                      factor_norm([good for good, _ in values]))
    maximum = relative_cap * reference_norm
    scale = min(1.0, maximum / proposed_norm) if proposed_norm > 0 else 0.0
    updated = [a + scale * b for a, b in zip(old, proposed, strict=True)]
    if not all(bool(torch.isfinite(x).all()) for x in updated):
        raise ValueError("Nonfinite contrastive update")
    return updated, {"direction_norm": factor_norm(direction),
                     "proposed_norm": proposed_norm,
                     "reference_norm": reference_norm,
                     "applied_norm": proposed_norm * scale,
                     "scale": scale}
