"""Retrieve the nearest successful source evidence independently of native Q."""
from __future__ import annotations

import hashlib
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import sha
from .evidence_selected_memory import EvidenceSelectedMemory, compact
from .memory import Memory


class EvidenceSelectedMemoryV2(EvidenceSelectedMemory):
    def __init__(self, plan, client, directory, calibration, embedder=None):
        super().__init__(plan, client, directory, calibration, embedder=embedder)
        self.signature = hashlib.sha256((self.signature + sha(Path(__file__))).encode()).hexdigest()

    def retrieve(self, query):
        result = Memory.retrieve(self, query)
        candidates = []
        for mid, record in self.evidence.items():
            if not record['selected'] or mid not in self.store.items:
                continue
            metadata = self.store.get(mid).metadata.model_dump()
            source_task = metadata['task_description']
            candidates.append((mid, source_task, record))
        result['evidence_candidates'] = len(candidates)
        result['evidence_cards'] = []
        if not candidates:
            return result
        embedded = self.service.embedding_provider.embed([query] + [x[1] for x in candidates])
        current = embedded[0]
        scores = [sum(a * b for a, b in zip(current, vector)) for vector in embedded[1:]]
        ranking = sorted(range(len(candidates)), key=lambda i: (-scores[i], candidates[i][0]))
        for i in ranking:
            mid, _, record = candidates[i]
            source = record['source_binding']
            lines = []
            for row in record['selected']:
                lines.append(f'  step {row["step"]}: {compact(row["action"], 170)} '
                             f'→ {compact(row["feedback"], 260)}')
            text = ('\nVerbatim public evidence from a completed earlier task '
                    f'[source {source.get("game_sha256", source.get("initial_query_sha256", ""))[:12]}]:\n' +
                    '\n'.join(lines))
            proposed = result['context'] + text
            tokens = len(self.client.tokenizer.encode(proposed, add_special_tokens=False))
            if tokens > self.plan['memory_tokens']:
                continue
            result['context'] = proposed
            result['evidence_cards'] = [dict(memory_id=mid,
                source_input_sha256=source.get('game_sha256', source.get('initial_query_sha256')),
                similarity=scores[i], steps=[row['step'] for row in record['selected']])]
            result['tokens'] = tokens
            result['context_sha256'] = hashlib.sha256(proposed.encode()).hexdigest()
            break
        return result
