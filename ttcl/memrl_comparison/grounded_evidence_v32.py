"""Require two distinct stable-feedback groups before contextual readout.

An isolated repeated tool result can anchor the actor on one past path.
This domain-agnostic evidence-diversity gate keeps native MemRL context
when fewer than two groups fit the original memory budget. ALFWorld text
commands and the v26 typed final-action operator are unchanged.
"""
from __future__ import annotations

from .grounded_evidence_v27 import TypedGroundedMemory as V27Memory
from .grounded_evidence_v30 import TypedGroundedMemory as V30Memory
from .memory import digest


MIN_DISTINCT_FEEDBACK_GROUPS = 2


class TypedGroundedMemory(V30Memory):
    def __init__(self, plan, client, directory, calibration, embedder=None):
        super().__init__(plan, client, directory, calibration, embedder=embedder)
        self.signature = digest(dict(base=self.signature, variant='typed_grounded_v32',
                                     min_distinct_feedback_groups=
                                     MIN_DISTINCT_FEEDBACK_GROUPS))

    def retrieve(self, query):
        proposed = super().retrieve(query)
        selected = proposed.get('recurrent_feedback', [])
        if self._text_retry or len(selected) != 1:
            return proposed
        native = V27Memory.retrieve(self, query)
        return dict(native, recurrent_feedback=[],
                    rejected_recurrent_feedback=selected,
                    readout='abstain_single_stable_feedback_group')
