"""Deterministic matched-norm LoRA perturbation without trajectory direction."""

from __future__ import annotations

import torch

from ttcl.trajectory_hyperlora.alfworld_failure_retry_residual_v31b import (
    effective_norm,
)


def matched_random_residual(agent, template: list[torch.Tensor],
                            target_norm: float, seed: int
                            ) -> list[torch.Tensor]:
    if target_norm < 0 or not torch.isfinite(torch.tensor(target_norm)):
        raise ValueError("Invalid target norm")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = [torch.randn(value.shape, generator=generator,
                         dtype=torch.float32)
             for value in template]
    norm = effective_norm(agent, noise)
    if norm <= 0 and target_norm > 0:
        raise ValueError("Random LoRA direction has zero effective norm")
    scale = target_norm / norm if norm > 0 else 0.
    return [value * scale for value in noise]
