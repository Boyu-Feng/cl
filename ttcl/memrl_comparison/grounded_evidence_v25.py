"""Keep structured-action actors on native context; project public history.

The source-bound public event ledger remains available to the general typed
action operator. Extra historical event text is not injected into the actor
prompt, and the v21 post-first-action context withdrawal is bypassed. Text
commands retain the v24 retry schedule without task-family branches.
"""
from __future__ import annotations

from .grounded_evidence_v15 import TypedGroundedMemory as V15Memory
from .grounded_evidence_v24 import TypedGroundedMemory as V24Memory
from .memory import Memory, digest


class TypedGroundedMemory(V24Memory):
    def __init__(self, plan, client, directory, calibration, embedder=None):
        super().__init__(plan, client, directory, calibration, embedder=embedder)
        self.signature = digest(dict(base=self.signature, variant='typed_grounded_v25',
                                     structured_actor='native_context',
                                     public_history='typed_projection_only'))

    def retrieve(self, query):
        if self._text_retry:
            return super().retrieve(query)
        native = Memory.retrieve(self, query)
        self.selected_context = native['context']
        self.selected_public_context = ''
        self.selected_evidence = []
        return dict(native, evidence=[], projection='native_actor_typed_public_history')

    def decorate_system(self, system):
        # The typed operator uses self.public_events directly. Calling v15
        # skips v21's withdrawal of otherwise native actor context.
        V15Memory.decorate_system(self, system)
