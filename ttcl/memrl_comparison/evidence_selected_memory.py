"""MemRL plus model-selected, verbatim public trajectory evidence.

The selector sees completed public trajectories and returns step indices only.
It does not author a rule. Nonpositive source outcomes are retained by native
MemRL, but are not promoted as positive evidence cards.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests

from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .memory import Memory


SELECT_SCHEMA = {'type': 'object', 'properties': {
    'step_ids': {'type': 'array', 'items': {'type': 'integer'}}},
    'required': ['step_ids'], 'additionalProperties': False}


def compact(value, limit):
    value = str(value)
    return value if len(value) <= limit else value[:limit // 2] + ' […] ' + value[-limit // 2:]


def public_rows(trace):
    if isinstance(trace, dict):
        rows = trace.get('trajectory', [])
    elif isinstance(trace, list):
        rows = trace
    else:
        raise ValueError('Unsupported public trajectory')
    out = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError('Invalid public step')
        action = row.get('action')
        if action is None:
            raise ValueError('Missing public action')
        if not isinstance(action, str):
            action = json.dumps(action, ensure_ascii=False, sort_keys=True)
        feedback = row.get('observation', row.get('public_feedback', ''))
        out.append(dict(step=i, action=action, feedback=str(feedback),
                        query=str(row.get('query', ''))))
    return out


class EvidenceSelectedMemory(Memory):
    def __init__(self, plan, client, directory, calibration, embedder=None):
        super().__init__(plan, client, directory, calibration, embedder=embedder)
        self.signature = hashlib.sha256((self.signature + sha(Path(__file__))).encode()).hexdigest()
        self.evidence = {}
        self.selection_calls = 0
        self.selection_tokens = 0

    def _select(self, rows, binding):
        prompt = ('From this completed public interaction, select up to three '
            'steps whose action and following feedback explicitly show a '
            'potentially reusable state change or achieved subgoal. Return only '
            'step IDs; do not infer advice, hidden state, optimality, or a '
            'counterfactual. Exclude actions with no observed effect, passive '
            'scene descriptions, and final scoring alone. If there is no '
            'clear action effect, return an empty list. JSON only.')
        shown = [dict(step=row['step'], action=compact(row['action'], 240),
                      feedback=compact(row['feedback'], 320),
                      query=compact(row['query'], 220)) for row in rows]
        messages = [dict(role='system', content=prompt),
                    dict(role='user', content=json.dumps(shown, ensure_ascii=False))]
        body = dict(model='frozen-actor', messages=messages,
                    seed=98000 + self.selection_calls, temperature=0,
                    max_tokens=256, response_format={'type': 'json_schema',
                        'json_schema': {'name': 'evidence_steps', 'schema': SELECT_SCHEMA,
                                        'strict': True}})
        response = requests.post(self.plan['url'].rstrip('/') + '/v1/chat/completions',
                                 json=body, timeout=300)
        response.raise_for_status()
        result = response.json()
        choice = result['choices'][0]
        self.selection_calls += 1
        self.selection_tokens += (result.get('usage', {}).get('prompt_tokens', 0) +
                                  result.get('usage', {}).get('completion_tokens', 0))
        raw = choice['message']['content']
        try:
            ids = json.loads(raw)['step_ids'] if choice['finish_reason'] == 'stop' else []
        except (ValueError, TypeError, KeyError):
            ids = []
        if (not isinstance(ids, list) or len(ids) > 3 or
                any(type(i) is not int or i < 0 or i >= len(rows) for i in ids) or
                len(set(ids)) != len(ids)):
            ids = []
        selected = [dict(step=i, action=rows[i]['action'],
                         feedback=rows[i]['feedback']) for i in ids]
        record = dict(source_binding=binding, rows_sha256=hashlib.sha256(
            json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            prompt_sha256=hashlib.sha256(json.dumps(messages,
                sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            raw=raw, finish_reason=choice['finish_reason'], selected=selected,
            usage=result.get('usage', {}))
        return record

    def retrieve(self, query):
        result = super().retrieve(query)
        context = result['context']
        cards = []
        for memory_id in result['ids']:
            record = self.evidence.get(memory_id)
            if not record:
                continue
            source = record['source_binding']
            for row in record['selected']:
                text = ('Past observed action and feedback '
                    f'[source {source.get("game_sha256", source.get("initial_query_sha256", ""))[:12]}, '
                    f'step {row["step"]}]: {compact(row["action"], 170)} '
                    f'→ {compact(row["feedback"], 260)}')
                proposed = (context + '\n' + text).strip()
                if len(self.client.tokenizer.encode(proposed, add_special_tokens=False)) > self.plan['memory_tokens']:
                    break
                context = proposed
                cards.append(dict(memory_id=memory_id, step=row['step']))
        result['context'] = context
        result['evidence_cards'] = cards
        result['tokens'] = len(self.client.tokenizer.encode(context, add_special_tokens=False))
        result['context_sha256'] = hashlib.sha256(context.encode()).hexdigest()
        return result

    def update(self, query, public_trace, reward, success, retrieval, binding):
        result = super().update(query, public_trace, reward, success, retrieval, binding)
        memory_id = result['new_memory_id']
        if reward > 0:
            rows = public_rows(json.loads(public_trace))
            selected = self._select(rows, binding)
            self.evidence[memory_id] = selected
            save(self.directory / f'evidence_{memory_id}.json', selected)
            result['evidence_steps'] = [row['step'] for row in selected['selected']]
        else:
            result['evidence_steps'] = []
        return result

    def snapshot(self, path):
        super().snapshot(path)
        state = read(path)
        state['evidence'] = self.evidence
        state['selection_calls'] = self.selection_calls
        state['selection_tokens'] = self.selection_tokens
        save(path, state)

    def restore(self, path):
        super().restore(path)
        state = read(path)
        if not isinstance(state.get('evidence'), dict):
            raise ValueError('Missing evidence state')
        self.evidence = state['evidence']
        self.selection_calls = state['selection_calls']
        self.selection_tokens = state['selection_tokens']
