"""Learn a bounded matrix-state transition from old B and a new trajectory.

The frozen trajectory hypernetwork encodes the new public trajectory once.
This module reads both actual old matrix factors and that encoding. Its
supervised target is the frozen encoder's full-prefix matrix on reviewed
training-domain histories; no valid_unseen reward enters training.
"""

from __future__ import annotations

import torch
from torch import nn


def features(old: list[torch.Tensor], new: list[torch.Tensor],
             reward: float, steps: int, updates: int) -> torch.Tensor:
    if len(old) != len(new) or not old:
        raise ValueError("Changed LoRA factor count")
    rows = []
    for previous, incoming in zip(old, new, strict=True):
        a = previous.detach().float().flatten()
        b = incoming.detach().float().flatten()
        if a.shape != b.shape or not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise ValueError("Changed or nonfinite LoRA B")
        na, nb = a.norm().clamp_min(1e-8), b.norm().clamp_min(1e-8)
        rows.extend((torch.log1p(na), torch.log1p(nb),
            (a @ b) / (na * nb), (a - b).norm() / na,
            a.mean(), b.mean(), a.std(), b.std()))
    rows.extend((torch.tensor(float(reward)),
                 torch.tensor(float(steps) / 50),
                 torch.tensor(float(updates) / 20)))
    return torch.stack(rows).float()


class JointBUpdater(nn.Module):
    def __init__(self, feature_dim: int, layers: int, hidden: int = 32):
        super().__init__()
        self.layers = layers
        self.net = nn.Sequential(nn.Linear(feature_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, layers))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


@torch.no_grad()
def update(model: JointBUpdater, old: list[torch.Tensor] | None,
           new: list[torch.Tensor], reward: float, steps: int,
           updates: int) -> tuple[list[torch.Tensor], list[float]]:
    if old is None:
        return [x.detach().cpu().float().clone() for x in new], [1.] * len(new)
    gate = model(features(old, new, reward, steps, updates)).cpu()
    factors = [a.detach().cpu().float() + float(g) *
               (b.detach().cpu().float() - a.detach().cpu().float())
               for a, b, g in zip(old, new, gate, strict=True)]
    if not all(bool(torch.isfinite(x).all()) for x in factors):
        raise ValueError("Joint update generated nonfinite B")
    return factors, gate.tolist()
