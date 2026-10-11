"""Failure-conditioned bounded B residual on the frozen 68/134 actor.

The original expert-source hypernetwork remains active at every step. A
residual from the completed failed first attempt is added after each dynamic
readout. Opposite signs are development arms, not outcome labels inferred from
terminal failure alone.
"""

from __future__ import annotations

from contextlib import contextmanager
import math

import torch


def effective_norm(agent, factors: list[torch.Tensor]) -> float:
    if len(factors) != len(agent.adapters):
        raise ValueError("LoRA layer count changed")
    squared = 0.0
    for layer, factor in zip(agent.adapters, factors, strict=True):
        value = factor.detach().float().to(layer.a.device)
        expected = (1, layer.base.out_features, layer.rank)
        if value.shape != expected or not bool(torch.isfinite(value).all()):
            raise ValueError("Changed or nonfinite LoRA factor")
        delta_w = value @ layer.a.detach().float() / layer.rank
        squared += float(delta_w.square().sum())
    return math.sqrt(squared)


def bounded_residual(agent, expert: list[torch.Tensor],
                     failed: list[torch.Tensor], *, max_relative: float = .25
                     ) -> tuple[list[torch.Tensor], dict[str, float]]:
    if not 0 < max_relative <= 1 or len(expert) != len(failed):
        raise ValueError("Invalid residual bound or layer count")
    difference = [bad.detach().cpu().float() - good.detach().cpu().float()
                  for good, bad in zip(expert, failed, strict=True)]
    reference_norm = effective_norm(agent, expert)
    difference_norm = effective_norm(agent, difference)
    cap = max_relative * reference_norm
    scale = min(1.0, cap / difference_norm) if difference_norm > 0 else 0.0
    residual = [scale * value for value in difference]
    return residual, {"expert_effective_norm": reference_norm,
        "difference_effective_norm": difference_norm,
        "residual_effective_norm": difference_norm * scale,
        "relative_cap": max_relative, "scale": scale}


@contextmanager
def inject_dynamic_residual(agent, residual: list[torch.Tensor]):
    """Add a frozen B correction after each task-conditioned readout."""
    if len(residual) != len(agent.adapters):
        raise ValueError("LoRA layer count changed")
    original = agent.set_source

    def patched(source, *args, **kwargs):
        result = original(source, *args, **kwargs)
        if source is not None:
            for layer, value in zip(agent.adapters, residual, strict=True):
                if layer.b.shape != value.shape:
                    raise ValueError("Residual shape changed")
                layer.b = layer.b + value.to(layer.b.device, layer.b.dtype)
        return result

    agent.set_source = patched
    try:
        yield
    finally:
        agent.set_source = original
        original(None)
