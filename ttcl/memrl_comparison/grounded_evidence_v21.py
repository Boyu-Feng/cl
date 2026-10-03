"""Use a memory-free continuation when first structured action is unprojectable.

The first action retains v15 context. If it exposes no compatible typed
projection, later actor calls omit injected historical text. The source-bound
event ledger remains available to the deterministic action projection.
"""
from __future__ import annotations

import hashlib

from ttcl.icl_mem0_comparison.protocol import read, save
from .grounded_evidence_v15 import TypedGroundedMemory as V15Memory
from .memory import digest


def withdraw_unprojectable_memory(system, first_audit: dict) -> dict | None:
    if first_audit.get('typed_candidate_kinds'):
        return None
    marker = '\n\nPast experience:\n'
    before = system.messages[0]['content']
    if marker not in before:
        return None
    instruction, context = before.split(marker, 1)
    system.messages[0]['content'] = instruction
    return dict(decision='withdraw_unprojectable_memory_after_first_action',
                first_action_reason=first_audit.get('reason'),
                first_action_candidate_kinds=first_audit['typed_candidate_kinds'],
                removed_context_sha256=hashlib.sha256(context.encode()).hexdigest(),
                before_sha256=hashlib.sha256(before.encode()).hexdigest(),
                after_sha256=hashlib.sha256(instruction.encode()).hexdigest())


class TypedGroundedMemory(V15Memory):
    def __init__(self, plan, client, directory, calibration, embedder=None):
        super().__init__(plan, client, directory, calibration, embedder=embedder)
        self.signature = digest(dict(base=self.signature, variant='typed_grounded_v21',
                                     rule='withdraw_unprojectable_memory_after_first_action'))

    def decorate_system(self, system):
        super().decorate_system(system)
        previous = system.respond

        def respond(query):
            response = previous(query)
            if system.turn == 1:
                first = read(system.output / 'typed_action_001.json')
                change = withdraw_unprojectable_memory(system, first)
                if change:
                    save(system.output / 'typed_context_adaptation.json', change)
            return response

        system.respond = respond
