"""Withdraw unprojectable public-event text after the first structured action.

The event ledger remains available for later type-directed projection. Only
the extra actor-prompt excerpt is withdrawn when the first action shows no
compatible candidate; native MemRL text and its Q-credit behavior remain.
"""
from __future__ import annotations

import hashlib

from ttcl.icl_mem0_comparison.protocol import read, save
from .grounded_evidence_v15 import TypedGroundedMemory as V15Memory
from .memory import digest


def withdraw_unprojectable_evidence(system, public_context: str,
                                    first_audit: dict) -> dict | None:
    if first_audit.get('typed_candidate_kinds') or not public_context:
        return None
    marker = '\n\nPast experience:\n'
    before = system.messages[0]['content']
    if marker not in before:
        return None
    instruction, context = before.split(marker, 1)
    if not context.endswith(public_context):
        return None
    native = context[:-len(public_context)].rstrip()
    after = instruction + marker + native if native else instruction
    system.messages[0]['content'] = after
    return dict(decision='withdraw_unprojectable_public_event_text_after_first_action',
                first_action_reason=first_audit.get('reason'),
                first_action_candidate_kinds=first_audit['typed_candidate_kinds'],
                native_context_retained=bool(native),
                before_sha256=hashlib.sha256(before.encode()).hexdigest(),
                after_sha256=hashlib.sha256(after.encode()).hexdigest())


class TypedGroundedMemory(V15Memory):
    def __init__(self, plan, client, directory, calibration, embedder=None):
        super().__init__(plan, client, directory, calibration, embedder=embedder)
        self.signature = digest(dict(base=self.signature, variant='typed_grounded_v20',
                                     rule='withdraw_unprojectable_event_text_after_first_action'))

    def decorate_system(self, system):
        super().decorate_system(system)
        previous = system.respond

        def respond(query):
            response = previous(query)
            if system.turn == 1:
                first = read(system.output / 'typed_action_001.json')
                change = withdraw_unprojectable_evidence(
                    system, self.selected_public_context, first)
                if change:
                    save(system.output / 'typed_context_adaptation.json', change)
            return response

        system.respond = respond
