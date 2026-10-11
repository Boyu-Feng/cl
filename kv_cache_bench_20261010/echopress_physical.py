"""EchoPress prefill scores followed by a rectangular, physical KV eviction.

The released EchoPress implementation masks scored entries during attention but
keeps its full cache tensor. This adaptation uses its unchanged scoring path and
keeps the highest-scored positions per KV head at the requested fixed budget.
"""

import math

import torch
from kvpress import EchoPress


class PhysicalEchoPress(EchoPress):
    def compress_post(self, model):
        if self.compression_ratio <= 0:
            return
        scores = self.score_val
        scores[..., : self.n_sink] = scores.amax() + 1.0
        length = scores.shape[-1]
        retained = max(1, math.ceil(length * (1 - self.compression_ratio)))
        cache = self._cache
        for layer_idx, layer in enumerate(cache.layers):
            selected = scores[layer_idx].topk(retained, dim=-1).indices.sort(dim=-1).values
            gather_index = selected.unsqueeze(-1).expand(-1, -1, -1, layer.keys.shape[-1])
            layer.keys = torch.gather(layer.keys[:, :, :length], 2, gather_index).contiguous()
            layer.values = torch.gather(layer.values[:, :, :length], 2, gather_index).contiguous()
            model.model.layers[layer_idx].self_attn.masked_key_indices = None
