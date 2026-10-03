"""Generic, public structured evidence attached to each MemRL memory.

The field is derived from the public trajectory before the official reward is
used. It records actions and their observed feedback with a source hash. The
existing MemRL writer, embedding index, and Q update are unchanged. Earlier
schema-aware experiments remain separate comparison methods.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .memory import Memory, digest


def _public_value(value: Any, depth: int = 0) -> Any:
    """Keep JSON-shaped public actions without free-form reasoning fields."""
    if depth > 4:
        return None
    if isinstance(value, dict):
        return {str(k): _public_value(v, depth + 1) for k, v in value.items()
                if k not in {'thought', 'thinking', 'reasoning', 'analysis'}}
    if isinstance(value, list):
        return [_public_value(v, depth + 1) for v in value[:128]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def structured_field(public_trace: str, raw_action: Any = None) -> dict:
    """Create a bounded, task-independent record from public episode events."""
    trace = json.loads(public_trace)
    if isinstance(trace, dict):
        steps = trace.get('trajectory', [])
    elif isinstance(trace, list):
        steps = trace
    else:
        raise ValueError('Expected public episode object or step list')
    if not isinstance(steps, list):
        raise ValueError('Public steps must be a list')
    events = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        action = step.get('action')
        feedback = step.get('public_feedback', step.get('observation', ''))
        if action is None:
            continue
        events.append(dict(index=index, action=_public_value(action),
                           feedback=str(feedback)[:1200]))
    return dict(schema='public_structured_evidence_v1',
                public_trace_sha256=hashlib.sha256(public_trace.encode()).hexdigest(),
                event_count=len(events), events=events[-8:],
                raw_final_action=_public_value(raw_action if raw_action is not None else
                                               events[-1]['action'] if events else None))


class StructuredEvidenceMemory(Memory):
    """Native MemRL plus a generic per-memory evidence field in the prompt."""

    def __init__(self, plan, client, directory, calibration, embedder=None):
        super().__init__(plan, client, directory, calibration, embedder=embedder)
        self.signature = digest(dict(base=self.signature,
                                     variant='structured_evidence_v1',
                                     max_events=8, max_feedback_chars=1200))

    def retrieve(self, query):
        # Keep native embedding/Q ranking, but allow a candidate regardless
        # of the fixed absolute similarity cutoff.
        original = self.service.rl_config.sim_threshold
        self.service.rl_config.sim_threshold = 0.
        try:
            result = super().retrieve(query)
        finally:
            self.service.rl_config.sim_threshold = original
        context, ids = '', []
        for mid in result['selected_before_budget']:
            metadata = self.store.get(mid).metadata.model_dump()
            field = metadata.get('structured_evidence')
            compact = ([{'action': event['action'], 'feedback': event['feedback'][:240]}
                        for event in field['events'][-2:]] if field else [])
            abstract = str(metadata.get('public_abstract') or '')
            abstract_ids = self.client.tokenizer.encode(abstract, add_special_tokens=False)
            abstract = self.client.tokenizer.decode(abstract_ids[:240], skip_special_tokens=True)
            entry = ('Task: ' + str(metadata['task_description']) + '\nLesson: ' + abstract +
                     '\nPublic action/observation evidence: ' +
                     json.dumps(compact, ensure_ascii=False, separators=(',', ':')))
            proposed = (context + '\n\n' + entry).strip()
            if len(self.client.tokenizer.encode(proposed, add_special_tokens=False)) <= self.plan['memory_tokens']:
                context = proposed
                ids.append(mid)
        return dict(result, context=context,
                    ids=ids,
                    dropped_whole_entries=[mid for mid in result['selected_before_budget'] if mid not in ids],
                    tokens=len(self.client.tokenizer.encode(context, add_special_tokens=False)),
                    context_sha256=hashlib.sha256(context.encode()).hexdigest(),
                    structured_projection='public_structured_evidence_v1')

    def update(self, query, public_trace, reward, success, retrieval, binding):
        # Validate and construct the public field before the native Q update.
        field = structured_field(public_trace)
        result = super().update(query, public_trace, reward, success, retrieval, binding)
        mid = result['new_memory_id']
        item = self.store.get(mid).model_dump()
        item['metadata']['structured_evidence'] = field
        self.store.update(mid, item)
        return dict(result, structured_evidence_sha256=digest(field))
