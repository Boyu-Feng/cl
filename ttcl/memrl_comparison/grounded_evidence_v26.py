"""Require four independent prior submissions for numeric projection.

Historical numerical submissions are unverified. The operator can still
consider recurrent records at its existing threshold; numerical consensus
abstains until at least four distinct source episodes have the same shape.
The text retry and native structured-actor context remain v25 behavior.
"""
from __future__ import annotations

from ttcl.icl_mem0_comparison.protocol import read, save
from .grounded_evidence_v25 import TypedGroundedMemory as V25Memory
from .memory import digest


MIN_NUMERIC_SOURCES = 4


def weak_numeric_projection(audit: dict) -> bool:
    candidate = audit.get('candidate') or {}
    return (audit.get('operation') == 'NUMERIC_CONSENSUS' and
            candidate.get('kind') == 'numeric_consensus' and
            isinstance(candidate.get('sample_count'), int) and
            candidate['sample_count'] < MIN_NUMERIC_SOURCES)


class TypedGroundedMemory(V25Memory):
    def __init__(self, plan, client, directory, calibration, embedder=None):
        super().__init__(plan, client, directory, calibration, embedder=embedder)
        self.signature = digest(dict(base=self.signature, variant='typed_grounded_v26',
                                     min_numeric_source_episodes=MIN_NUMERIC_SOURCES))

    def decorate_system(self, system):
        super().decorate_system(system)
        previous = system.respond

        def respond(query):
            response = previous(query)
            path = system.output / f'typed_action_{system.turn:03d}.json'
            audit = read(path)
            if not weak_numeric_projection(audit):
                return response
            raw = audit['actor']
            original = query.response_schema.model_validate(raw)
            if system.last is None:
                raise ValueError('Missing actor action when reverting projection')
            system.last = (system.last[0], raw)
            audit.update(final=raw, operation='KEEP',
                         reason='Fewer than four independent numeric source episodes',
                         rejected_candidate=audit.pop('candidate'))
            save(path, audit)
            return type(response)(action=original, metadata=response.metadata)

        system.respond = respond
