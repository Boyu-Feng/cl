"""Bounded online memory of trajectory-generated LoRA factors.

Keys only select parameter updates; no source text is needed at read time.
All tensors are kept on CPU between episodes. The update is a convex average,
so storing more experiences cannot make factor scale grow by summation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import torch
from torch.nn import functional as F


@dataclass
class ParameterEntry:
    key: torch.Tensor
    factors: list[torch.Tensor]
    count: int = 1


class ParameterMemory:
    def __init__(self, *, capacity: int = 64, temperature: float = 0.05,
                 attention_mix: float = 0.75):
        if capacity < 1 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Invalid parameter memory capacity or temperature")
        if not math.isfinite(attention_mix) or not 0 <= attention_mix <= 1:
            raise ValueError("Attention mix must be in [0, 1]")
        self.capacity = capacity
        self.temperature = temperature
        self.attention_mix = attention_mix
        self.entries: list[ParameterEntry] = []

    @staticmethod
    def _key(value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 2 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 1 or not torch.isfinite(value).all():
            raise ValueError("Memory key must be a finite vector")
        value = value.detach().cpu().float().clone()
        if value.norm() == 0:
            raise ValueError("Memory key has zero norm")
        return F.normalize(value, dim=0)

    @staticmethod
    def _factors(values: list[torch.Tensor]) -> list[torch.Tensor]:
        if not values:
            raise ValueError("Parameter update has no LoRA factors")
        result = []
        for value in values:
            if value.ndim != 3 or value.shape[0] != 1 or not torch.isfinite(value).all():
                raise ValueError("Invalid LoRA factor")
            result.append(value.detach().cpu().float().clone())
        return result

    def write(self, key: torch.Tensor, factors: list[torch.Tensor]) -> int:
        key = self._key(key)
        factors = self._factors(factors)
        if self.entries:
            reference = self.entries[0]
            if (key.shape != reference.key.shape or
                    [x.shape for x in factors] != [x.shape for x in reference.factors]):
                raise ValueError("Parameter memory shape changed")
        if len(self.entries) < self.capacity:
            self.entries.append(ParameterEntry(key, factors))
            return len(self.entries) - 1
        # A full bank absorbs the new experience in the nearest latent state.
        # This keeps memory size fixed without task-family labels or source text.
        similarities = torch.stack([torch.dot(key, entry.key)
                                    for entry in self.entries])
        index = int(similarities.argmax())
        old = self.entries[index]
        count = old.count + 1
        candidate_key = old.key + (key - old.key) / count
        merged_key = (F.normalize(candidate_key, dim=0)
                      if candidate_key.norm() > 1e-8 else old.key.clone())
        merged_factors = [past + (fresh - past) / count
                          for past, fresh in zip(old.factors, factors, strict=True)]
        self.entries[index] = ParameterEntry(merged_key, merged_factors, count)
        return index

    def read(self, query: torch.Tensor) -> tuple[list[torch.Tensor] | None, dict]:
        if not self.entries:
            return None, {"entries": 0, "weights": [], "cosine": []}
        query = self._key(query)
        if query.shape != self.entries[0].key.shape:
            raise ValueError("Query key shape changed")
        similarity = torch.stack([torch.dot(query, item.key)
                                  for item in self.entries])
        attention = torch.softmax(similarity / self.temperature, dim=0)
        uniform = torch.full_like(attention, 1.0 / len(self.entries))
        weights = self.attention_mix * attention + (1 - self.attention_mix) * uniform
        result = [sum((weights[i] * entry.factors[layer]
                       for i, entry in enumerate(self.entries)),
                      torch.zeros_like(self.entries[0].factors[layer]))
                  for layer in range(len(self.entries[0].factors))]
        return result, {"entries": len(self.entries),
                        "weights": weights.tolist(), "cosine": similarity.tolist(),
                        "selected": int(attention.argmax())}

    def digest(self) -> str:
        state = hashlib.sha256()
        for entry in self.entries:
            state.update(str(entry.count).encode())
            for tensor in (entry.key, *entry.factors):
                state.update(str(tuple(tensor.shape)).encode())
                state.update(tensor.contiguous().numpy().tobytes())
        return state.hexdigest()
