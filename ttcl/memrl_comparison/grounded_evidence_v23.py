"""Retain native retrieval on every text-command retry.

Train-only paired retries found both positive and negative memory effects,
so an unconditional third-attempt dropout cannot be justified. Structured
JSON actions keep the source-bound v21 typed projection unchanged.
"""
from __future__ import annotations

from .grounded_evidence_v21 import TypedGroundedMemory as V21Memory
from .memory import Memory, digest


class TypedGroundedMemory(V21Memory):
    def __init__(self, plan, client, directory, calibration, embedder=None):
        super().__init__(plan, client, directory, calibration, embedder=embedder)
        self.signature = digest(dict(base=self.signature, variant='typed_grounded_v23',
                                     text_retry='native_retrieval_on_every_attempt'))

    def retrieve(self, query):
        if not self._text_retry:
            return super().retrieve(query)
        native = Memory.retrieve(self, query)
        self._text_attempt += 1
        self.selected_evidence = []
        self.selected_context = native['context']
        self.selected_public_context = ''
        return dict(native, evidence=[], retry_policy='native_exact_all_attempts',
                    action_interface='retryable_text_command')
