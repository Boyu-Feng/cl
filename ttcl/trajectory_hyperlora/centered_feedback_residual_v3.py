"""Low-parameter, source-centered trajectory evidence to LoRA latent code."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def normalized_source(source: torch.Tensor,
                      feedback_delta: torch.Tensor) -> torch.Tensor:
    return torch.cat((F.layer_norm(source.float(), source.shape[-1:]),
                      F.layer_norm(feedback_delta.float(),
                                   feedback_delta.shape[-1:])), dim=-1)


def normalized_target(target: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(target.float(), target.shape[-1:])


def centered_pca(rows: torch.Tensor, rank: int):
    rows = rows.float()
    if rank < 1 or rank >= len(rows):
        raise ValueError('PCA rank must be below reviewed row count')
    center = rows.mean(0)
    _, _, vh = torch.linalg.svd(rows - center, full_matrices=False)
    basis = vh[:rank].T.contiguous()
    coordinates = (rows - center) @ basis
    scale = coordinates.square().mean(0).sqrt().clamp_min(1e-6)
    return center, basis, scale


class CenteredEvidenceResidual(nn.Module):
    """Zero-mean source residual in a frozen training-bank PCA coordinate frame.

    For every fixed target, the eight reviewed training-source codes average
    to zero up to floating-point error. No task family or action slots exist.
    """

    def __init__(self, source_center, source_basis, source_scale,
                 target_center, target_basis, target_scale,
                 code_dim: int):
        super().__init__()
        if (source_basis.shape[1] != target_basis.shape[1] or
                source_scale.shape != target_scale.shape or
                code_dim < 1):
            raise ValueError('Source and target coordinate dimensions differ')
        self.register_buffer('source_center', source_center.float())
        self.register_buffer('source_basis', source_basis.float())
        self.register_buffer('source_scale', source_scale.float())
        self.register_buffer('target_center', target_center.float())
        self.register_buffer('target_basis', target_basis.float())
        self.register_buffer('target_scale', target_scale.float())
        rank = source_basis.shape[1]
        self.weight = nn.Parameter(torch.zeros(code_dim, rank * 2))

    @classmethod
    def from_state(cls, state: dict[str, torch.Tensor]):
        return cls(state['source_center'], state['source_basis'],
            state['source_scale'], state['target_center'],
            state['target_basis'], state['target_scale'],
            state['weight'].shape[0])

    def forward(self, source, feedback_delta, target):
        s = ((normalized_source(source, feedback_delta) -
              self.source_center) @ self.source_basis /
             self.source_scale)
        q = ((normalized_target(target) - self.target_center) @
             self.target_basis / self.target_scale)
        return F.linear(torch.cat((s, s * q), dim=-1), self.weight)


def from_encoded_features(encoded, rank: int, code_dim: int):
    global_source = encoded['global']
    delta = torch.stack([encoded['deltas'][i, :int(count)].mean(0)
        for i, count in enumerate(encoded['event_counts'])])
    source_rows = torch.stack([normalized_source(global_source[i], delta[i])
        for i in range(len(global_source))])
    target_rows = torch.stack([normalized_target(target)
        for target in encoded['targets']])
    source = centered_pca(source_rows, rank)
    target = centered_pca(target_rows, rank)
    return CenteredEvidenceResidual(*source, *target, code_dim)
