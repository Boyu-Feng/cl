"""Conservative context-utility gating with a typed public-action ledger.

The gate is learned from paired MemRL/no-memory outcomes on separate
development runs. It never reads the current task's reward before retrieval.
It suppresses native MemRL text only when predicted harm is substantial;
otherwise native retrieval, writer and Q updates are preserved. Public-action
projection is inherited from the frozen schema-aware pilot.
"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import read, sha
from .general_evidence import GeneralEvidenceMemory
from .memory import Memory, digest


STOP = frozenset(('the a to in on at and or of for with you your is are be '
                  'from this that it an by as under into put task').split())


def words(value):
    return frozenset(w for w in re.findall(r'[a-z]{3,}', value.lower()) if w not in STOP)


def retrieval_features(retrieval, metadata):
    """Features available before acting; never include the current reward."""
    mids = retrieval['ids']
    selected = [metadata[mid] for mid in mids]
    n = len(selected)

    def q_value(item):
        value = item.get('q_value')
        return float(value) if value is not None and math.isfinite(float(value)) else 0.

    return [n / 3., retrieval.get('tokens', 0) / 2048.,
            sum(m.get('success') is True for m in selected) / max(1, n),
            sum(m.get('success') is False for m in selected) / max(1, n),
            sum(q_value(m) for m in selected) / max(1, n),
            sum(m.get('writer_token_limit_hit') is True for m in selected) / max(1, n)]


class UtilityGate:
    def __init__(self, path):
        self.path = Path(path)
        self.data = read(self.path)
        if self.data.get('schema') != 'contextual_utility_ridge_v1':
            raise ValueError('Unknown utility-gate schema')
        self.vocab = self.data['vocab']
        self.weights = self.data['weights']
        if len(self.weights) != len(self.vocab) + 7:
            raise ValueError('Utility-gate weight dimension mismatch')
        self.threshold = float(self.data['threshold'])
        self.sha256 = sha(self.path)

    def predict(self, query, features):
        if len(features) != 6:
            raise ValueError('Expected six pre-action retrieval features')
        tokens = words(query)
        score = self.weights[0]
        score += sum(weight for token, weight in zip(self.vocab, self.weights[1:-6])
                     if token in tokens)
        score += sum(float(x) * float(w) for x, w in zip(features, self.weights[-6:]))
        if not math.isfinite(score):
            raise ValueError('Non-finite predicted utility')
        return score


class ContextualUtilityMemory(GeneralEvidenceMemory):
    def __init__(self, plan, client, directory, calibration, gate_path, embedder=None):
        self.gate = UtilityGate(gate_path)
        super().__init__(plan, client, directory, calibration, embedder=embedder)
        self.signature = digest(dict(base=self.signature,
                                     variant='contextual_utility_v1',
                                     gate_sha256=self.gate.sha256))

    def retrieve(self, query):
        base = Memory.retrieve(self, query)
        metadata = {mid: self.store.get(mid).metadata.model_dump() for mid in base['ids']}
        features = retrieval_features(base, metadata)
        predicted = self.gate.predict(query, features)
        if not base['ids'] or predicted > self.gate.threshold:
            return dict(base, gate_prediction=predicted, gate_threshold=self.gate.threshold,
                        gate_features=features, gate='native')
        old_ids = base['ids']
        return dict(base, context='', ids=[], tokens=0,
                    dropped_whole_entries=base['dropped_whole_entries'] + old_ids,
                    context_sha256=hashlib.sha256(b'').hexdigest(),
                    gate_prediction=predicted, gate_threshold=self.gate.threshold,
                    gate_features=features, gate='suppressed')
