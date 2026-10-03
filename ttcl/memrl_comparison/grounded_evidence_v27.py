"""Keep the v26 typed action path while restoring native text retries.

The v24 second-attempt dropout regressed on the first complete unseen
ALFWorld family. A common policy cannot retain that failed text schedule;
structured JSON actions keep v26's native actor context and typed operator.
"""
from __future__ import annotations

from .grounded_evidence_v23 import TypedGroundedMemory as V23Memory
from .grounded_evidence_v26 import TypedGroundedMemory as V26Memory
from .memory import digest


class TypedGroundedMemory(V26Memory):
    def __init__(self, plan, client, directory, calibration, embedder=None):
        super().__init__(plan, client, directory, calibration, embedder=embedder)
        self.signature = digest(dict(base=self.signature, variant='typed_grounded_v27',
                                     text_retry='native_exact_all_attempts'))

    def retrieve(self, query):
        if self._text_retry:
            return V23Memory.retrieve(self, query)
        return super().retrieve(query)
