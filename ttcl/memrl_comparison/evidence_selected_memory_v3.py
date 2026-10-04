"""Select observed transitions at any official reward scale.

Source reward is retained in native MemRL metadata. A zero or negative task
reward does not make every observed transition useless; selected rows are raw
facts and receive no positive skill label here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import save, sha
from .evidence_selected_memory import public_rows
from .evidence_selected_memory_v2 import EvidenceSelectedMemoryV2
from .memory import Memory


class EvidenceSelectedMemoryV3(EvidenceSelectedMemoryV2):
    def __init__(self, plan, client, directory, calibration, embedder=None):
        super().__init__(plan, client, directory, calibration, embedder=embedder)
        self.signature = hashlib.sha256((self.signature + sha(Path(__file__))).encode()).hexdigest()

    def retrieve(self, query):
        result = super().retrieve(query)
        if result['evidence_cards']:
            result['context'] = result['context'].replace(
                'Verbatim public evidence from a completed earlier task',
                'Verbatim public evidence from one past instance; verify current inputs')
            result['tokens'] = len(self.client.tokenizer.encode(
                result['context'], add_special_tokens=False))
            if result['tokens'] > self.plan['memory_tokens']:
                raise ValueError('Evidence prefix exceeded memory budget')
            result['context_sha256'] = hashlib.sha256(result['context'].encode()).hexdigest()
        return result

    def update(self, query, public_trace, reward, success, retrieval, binding):
        result = Memory.update(self, query, public_trace, reward, success, retrieval, binding)
        memory_id = result['new_memory_id']
        selected = self._select(public_rows(json.loads(public_trace)), binding)
        self.evidence[memory_id] = selected
        save(self.directory / f'evidence_{memory_id}.json', selected)
        result['evidence_steps'] = [row['step'] for row in selected['selected']]
        return result
