"""Trainable event memory, future-utility head, and LoRA latent residual.

Inputs are domain-neutral trajectory events and a current-task embedding.
The frozen text backbone supplies token representations; this module learns
which ordered feedback changes matter for later behavior.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class OnlineUpdateModel(nn.Module):
    def __init__(self, width: int, hidden: int = 48, code_dim: int = 8,
                 max_events: int = 16):
        super().__init__()
        self.width, self.hidden = width, hidden
        self.code_dim, self.max_events = code_dim, max_events
        self.global_proj = nn.Linear(width, hidden)
        self.event_proj = nn.Linear(width, hidden)
        self.delta_proj = nn.Linear(width, hidden)
        self.query_proj = nn.Linear(width, hidden)
        self.position = nn.Embedding(max_events, hidden)
        self.event_attention = nn.Linear(hidden, 1)
        self.source_head = nn.Sequential(nn.Linear(hidden * 2, hidden),
                                         nn.Tanh())
        self.pair = nn.Sequential(nn.Linear(hidden * 4, hidden),
                                  nn.Tanh(), nn.Linear(hidden, hidden),
                                  nn.Tanh())
        self.utility = nn.Linear(hidden, 1)
        self.residual = nn.Linear(hidden, code_dim)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    @staticmethod
    def norm(value: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(value.float(), value.shape[-1:])

    def encode_sources(self, global_vectors: torch.Tensor,
                       events: torch.Tensor, deltas: torch.Tensor,
                       counts: torch.Tensor) -> torch.Tensor:
        if events.shape != deltas.shape or events.shape[1] > self.max_events:
            raise ValueError("Changed event tensor size")
        times = torch.arange(events.shape[1], device=events.device)
        valid = times[None, :] < counts[:, None]
        if not bool(valid.any(dim=1).all()):
            raise ValueError("A trajectory has no reviewed events")
        value = torch.tanh(self.event_proj(self.norm(events)) +
            self.delta_proj(self.norm(deltas)) + self.position(times))
        logits = self.event_attention(value).squeeze(-1)
        weights = logits.masked_fill(~valid, -1e9).softmax(-1)
        event_summary = (weights[..., None] * value).sum(1)
        global_summary = torch.tanh(self.global_proj(
            self.norm(global_vectors)))
        return self.source_head(torch.cat((global_summary, event_summary), -1))

    def forward(self, global_vectors: torch.Tensor,
                events: torch.Tensor, deltas: torch.Tensor,
                counts: torch.Tensor, queries: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        sources = self.encode_sources(global_vectors, events, deltas, counts)
        query = torch.tanh(self.query_proj(self.norm(queries)))
        q = query[:, None, :].expand(-1, len(sources), -1)
        s = sources[None, :, :].expand(len(query), -1, -1)
        paired = self.pair(torch.cat((s, q, s * q, (s - q).abs()), -1))
        return self.utility(paired).squeeze(-1), self.residual(paired)
