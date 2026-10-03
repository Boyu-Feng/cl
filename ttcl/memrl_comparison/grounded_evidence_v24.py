"""Test a budget-neutral memory/no-memory/memory retry schedule.

Every retryable text task uses the same schedule. The second attempt omits
historical context but still lets the native writer learn from the result;
the first and third attempts use native retrieval. Structured JSON actions
retain the v23 typed projection without a text-retry schedule.
"""
from __future__ import annotations

import hashlib

from .grounded_evidence_v23 import TypedGroundedMemory as V23Memory
from .memory import Memory, digest


class TypedGroundedMemory(V23Memory):
    def __init__(self, plan, client, directory, calibration, embedder=None):
        super().__init__(plan, client, directory, calibration, embedder=embedder)
        self.signature = digest(dict(base=self.signature, variant='typed_grounded_v24',
                                     text_retry='native_empty_native'))

    def retrieve(self, query):
        if not self._text_retry or self._text_attempt != 1:
            return super().retrieve(query)
        native = Memory.retrieve(self, query)
        self._text_attempt += 1
        self.selected_evidence = []
        self.selected_context = ''
        self.selected_public_context = ''
        return dict(native, context='', ids=[], tokens=0,
                    context_sha256=hashlib.sha256(b'').hexdigest(),
                    suppressed_ids=native['ids'], evidence=[],
                    retry_policy='memory_dropout_second_attempt',
                    action_interface='retryable_text_command')
