"""One persistent LoRA factor state updated only by admitted trajectories.

The frozen hypernetwork produces candidate factors. The caller decides whether
the candidate is admitted. An admitted candidate updates the one carried state
by a running mean; a rejected candidate never calls write and leaves it intact.
"""

from __future__ import annotations

import hashlib

import torch
from torch.nn import functional as F

from ttcl.trajectory_hyperlora.online_parameter_memory_v2 import (
    ParameterEntry, ParameterMemory,
)


class ContinualLoRAState:
    def __init__(self):
        self.entries: list[ParameterEntry] = []

    def write(self, key: torch.Tensor, factors: list[torch.Tensor]) -> int:
        key = ParameterMemory._key(key)
        factors = ParameterMemory._factors(factors)
        if not self.entries:
            self.entries.append(ParameterEntry(key, factors))
            return 0
        old = self.entries[0]
        if (key.shape != old.key.shape or
                [x.shape for x in factors] != [x.shape for x in old.factors]):
            raise ValueError("Persistent LoRA shape changed")
        count = old.count + 1
        merged_key = old.key + (key - old.key) / count
        merged_key = (F.normalize(merged_key, dim=0)
                      if merged_key.norm() > 1e-8 else old.key.clone())
        merged_factors = [past + (fresh - past) / count
                          for past, fresh in zip(old.factors, factors, strict=True)]
        self.entries[0] = ParameterEntry(merged_key, merged_factors, count)
        return 0

    def read(self, query: torch.Tensor) -> tuple[list[torch.Tensor] | None, dict]:
        if not self.entries:
            return None, {"entries": 0, "weights": [], "cosine": []}
        query = ParameterMemory._key(query)
        state = self.entries[0]
        if query.shape != state.key.shape:
            raise ValueError("Persistent LoRA query shape changed")
        return [factor.clone() for factor in state.factors], {
            "entries": 1, "updates": state.count, "weights": [1.0],
            "cosine": [float(torch.dot(query, state.key))], "selected": 0}

    def digest(self) -> str:
        state = hashlib.sha256()
        state.update(b"continual_lora_v26")
        for entry in self.entries:
            state.update(str(entry.count).encode())
            for tensor in (entry.key, *entry.factors):
                state.update(str(tuple(tensor.shape)).encode())
                state.update(tensor.contiguous().numpy().tobytes())
        return state.hexdigest()
