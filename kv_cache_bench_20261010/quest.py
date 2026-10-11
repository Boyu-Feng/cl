"""Reference Quest page selection at decode time for Qwen3 attention.

Quest keeps full KV storage. The implementation below computes page extrema on
each decode step, so it reproduces page selection but not the optimized kernel
or the original paper's latency improvement.
"""

from __future__ import annotations

import math

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


class QuestAttentionBudget:
    def __init__(self, eviction_fraction: float, page_size: int = 16,
                 stable_prefix_length: int = 0):
        assert 0 <= eviction_fraction < 1
        self.retained_fraction = 1 - eviction_fraction
        self.page_size = page_size
        self.stable_prefix_length = stable_prefix_length
        self.original = ALL_ATTENTION_FUNCTIONS["sdpa"]
        self.selected_pages = []
        self.prefix_extrema = {}

    def __enter__(self):
        ALL_ATTENTION_FUNCTIONS["sdpa"] = self.forward
        return self

    def __exit__(self, *_):
        del ALL_ATTENTION_FUNCTIONS["sdpa"]

    def forward(self, module, query, key, value, attention_mask, **kwargs):
        if query.shape[-2] != 1 or key.shape[-2] <= self.page_size:
            return self.original(module, query, key, value, attention_mask, **kwargs)

        batch, heads, _, dim = query.shape
        groups = heads // key.shape[1]
        length = key.shape[-2]
        page = self.page_size
        pages = math.ceil(length / page)
        kept = max(1, math.ceil(pages * self.retained_fraction))
        stable_pages = min(self.stable_prefix_length // page, pages)
        layer_id = module.layer_idx
        if layer_id not in self.prefix_extrema:
            if stable_pages:
                prefix = key[:, :, :stable_pages * page].reshape(
                    batch, key.shape[1], stable_pages, page, dim)
                self.prefix_extrema[layer_id] = (prefix.amin(dim=-2), prefix.amax(dim=-2))
            else:
                self.prefix_extrema[layer_id] = (None, None)
        prefix_lo, prefix_hi = self.prefix_extrema[layer_id]
        tail = key[:, :, stable_pages * page:]
        tail_pages = math.ceil(tail.shape[-2] / page)
        missing = tail_pages * page - tail.shape[-2]
        if missing:
            tail = torch.nn.functional.pad(tail, (0, 0, 0, missing))
        if tail_pages:
            blocks = tail.reshape(batch, key.shape[1], tail_pages, page, dim)
            tail_lo, tail_hi = blocks.amin(dim=-2), blocks.amax(dim=-2)
            lo = torch.cat((prefix_lo, tail_lo), dim=-2) if stable_pages else tail_lo
            hi = torch.cat((prefix_hi, tail_hi), dim=-2) if stable_pages else tail_hi
        else:
            lo, hi = prefix_lo, prefix_hi
        lo = lo.repeat_interleave(groups, dim=1)
        hi = hi.repeat_interleave(groups, dim=1)
        q = query.squeeze(-2)
        bound = (q.clamp_min(0).unsqueeze(-2) * hi + q.clamp_max(0).unsqueeze(-2) * lo).sum(-1)
        # Always include the most recent page containing the current query.
        bound[..., -1] = torch.finfo(bound.dtype).max
        selected = bound.topk(kept, dim=-1).indices
        selected = selected.sort(dim=-1).values
        offsets = torch.arange(page, device=key.device)
        positions = (selected.unsqueeze(-1) * page + offsets).reshape(batch, heads, -1)
        valid = positions < length
        positions = positions.clamp_max(length - 1)
        batch_index = torch.arange(batch, device=key.device)[:, None, None]
        head_index = (torch.arange(heads, device=key.device) // groups)[None, :, None]
        chosen_key = key[batch_index, head_index, positions]
        chosen_value = value[batch_index, head_index, positions]
        scores = torch.matmul(query.float(), chosen_key.float().transpose(-1, -2)) / math.sqrt(dim)
        scores = scores.masked_fill(~valid.unsqueeze(-2), torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1).to(value.dtype)
        output = torch.matmul(probs, chosen_value)
        self.selected_pages.append((kept, pages))
        return output.transpose(1, 2), None
