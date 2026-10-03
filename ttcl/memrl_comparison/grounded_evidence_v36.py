"""No-training ALFWorld candidate: native retrieval with deferred ambiguous Q credit.

The structured-action path remains v27. A text attempt that exposes multiple
memories cannot identify each memory's contribution from one terminal reward.
Keep the retrieved context and writer provenance, but defer direct Q updates
for those IDs. Singleton retrievals retain the native update.
"""
from __future__ import annotations

from .grounded_evidence_v27 import TypedGroundedMemory as V27Memory
from .memory import digest


class TypedGroundedMemory(V27Memory):
    def __init__(self, plan, client, directory, calibration, embedder=None):
        super().__init__(plan, client, directory, calibration, embedder=embedder)
        self.signature = digest(dict(base=self.signature, variant='typed_grounded_v36',
                                     text_multi_memory_credit='defer_direct_q'))

    def update(self, query, public_trace, reward, success, retrieval, binding):
        ids = list(retrieval['ids'])
        if not self._text_retry or len(ids) < 2:
            result = super().update(query, public_trace, reward, success,
                                    retrieval, binding)
            return dict(result, deferred_q_ids=[])

        original = self.service.update_value
        expected = set(ids)
        if len(expected) != len(ids):
            raise ValueError('Duplicate retrieved memory ID')
        before_q = {mid: float(self.store.get(mid).metadata.q_value)
                    for mid in ids}
        seen = []

        def unchanged(memory_id, value):
            if memory_id not in expected or memory_id in seen:
                raise ValueError('Unexpected deferred Q update identity')
            seen.append(memory_id)
            # Memory.update still verifies a finite result and carries the
            # exact retrieved IDs into the native writer's provenance.
            return before_q[memory_id]

        self.service.update_value = unchanged
        try:
            result = super().update(query, public_trace, reward, success,
                                    retrieval, binding)
        finally:
            self.service.update_value = original
        if seen != ids:
            raise RuntimeError('Deferred Q update order changed')
        if any(float(self.store.get(mid).metadata.q_value) != before_q[mid]
               for mid in ids):
            raise RuntimeError('Deferred Q value changed')
        return dict(result, q_updates={}, deferred_q_ids=ids)
