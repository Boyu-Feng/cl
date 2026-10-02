"""Benchmark-agnostic, typed public-action memory on top of native MemRL.

The controller still writes native memories and updates their Q values. This
extension selects one concise, nonfailed lesson and carries forward public
structured actions. Numeric maps are averaged fieldwise; recurrent records in
entity lists are retained. Neither operator reads a task name or a score.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

from ttcl.icl_mem0_comparison.protocol import read, save
from .memory import Memory, digest


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _numeric_map(action):
    return isinstance(action, dict) and len(action) >= 12 and all(_number(v) for v in action.values())


def _record_list(action):
    if not isinstance(action, dict) or len(action) != 1:
        return None
    key, value = next(iter(action.items()))
    if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
        return None
    if not value:
        return key, None, None
    fields = set.intersection(*(set(v) for v in value))
    center = next((f for f in fields if f in ('center_freq', 'center', 'frequency', 'freq_mhz')), None)
    width = next((f for f in fields if f in ('bandwidth', 'width', 'width_mhz')), None)
    if not center or not width or not all(_number(v[center]) and _number(v[width]) for v in value):
        return None
    return key, center, width


class GeneralEvidenceMemory(Memory):
    """One method for textual strategies and structured public evidence."""

    def __init__(self, plan, client, directory, calibration, embedder=None):
        self.raw_actions = []
        self.pending_raw_action = None
        super().__init__(plan, client, directory, calibration, embedder=embedder)
        self.signature = digest(dict(base=self.signature, variant='general_evidence_v1',
                                     lesson_tokens=320, numeric_min_fields=12,
                                     record_min_count=2, record_fraction=.05))

    def retrieve(self, query):
        # Native embedding, similarity/Q ranking and writer remain in use. A
        # fixed absolute threshold was empty for most Cohort studies, so rank
        # whatever candidates exist and gate on observed quality instead.
        original = self.service.rl_config.sim_threshold
        self.service.rl_config.sim_threshold = 0.
        try:
            base = super().retrieve(query)
        finally:
            self.service.rl_config.sim_threshold = original
        context, ids = '', []
        for mid in base['selected_before_budget']:
            md = self.store.get(mid).metadata.model_dump()
            if md.get('success') is False or md.get('writer_token_limit_hit'):
                continue
            q = md.get('q_value', 0)
            if q is not None and float(q) < 0:
                continue
            lesson = str(md.get('public_abstract') or '').strip()
            if not lesson:
                continue
            ids = [mid]
            context = 'Prior experience (check against current observations):\n' + lesson
            tokens = self.client.tokenizer.encode(context, add_special_tokens=False)
            if len(tokens) > 320:
                context = self.client.tokenizer.decode(tokens[:320], skip_special_tokens=True)
            break
        if len(self.client.tokenizer.encode(context, add_special_tokens=False)) > self.plan['memory_tokens']:
            context, ids = '', []
        return dict(base, context=context, ids=ids,
                    tokens=len(self.client.tokenizer.encode(context, add_special_tokens=False)),
                    dropped_whole_entries=[mid for mid in base['selected_before_budget'] if mid not in ids],
                    context_sha256=hashlib.sha256(context.encode()).hexdigest(),
                    projection='general_evidence_v1')

    def _project(self, actor):
        if _numeric_map(actor):
            compatible = [a for a in self.raw_actions if _numeric_map(a) and a.keys() == actor.keys()]
            if compatible:
                return ({k: sum(float(a[k]) for a in compatible) / len(compatible)
                         for k in actor}, 'numeric_mean')
            return actor, 'numeric_first'
        spec = _record_list(actor)
        if spec is None:
            return actor, 'unchanged'
        key, center, width = spec
        if center is None:
            for old in reversed(self.raw_actions):
                old_spec = _record_list(old)
                if old_spec and old_spec[0] == key and old_spec[1]:
                    center, width = old_spec[1:]
                    break
        if center is None:
            return actor, 'empty_records'
        clusters = []
        scans = 0
        for old in self.raw_actions:
            old_spec = _record_list(old)
            if not old_spec or old_spec != (key, center, width):
                continue
            scans += 1
            for item in old[key]:
                match = next((c for c in clusters
                              if abs(float(c['item'][center]) - float(item[center])) <=
                              max(2.5, .25 * float(c['item'][width]))
                              and c['last'] != scans), None)
                if match:
                    n = match['count']
                    for field in (center, width):
                        match['item'][field] = (n * float(match['item'][field]) + float(item[field])) / (n + 1)
                    match['count'] += 1
                    match['last'] = scans
                else:
                    clusters.append(dict(item=copy.deepcopy(item), count=1, last=scans))
        answer = copy.deepcopy(actor)
        minimum = max(2, math.ceil(.05 * scans))
        for cluster in clusters:
            if cluster['count'] < minimum:
                continue
            item = cluster['item']
            if any(abs(float(current[center]) - float(item[center])) <=
                   max(2.5, .25 * float(item[width])) for current in answer[key]):
                continue
            if 'currently_active' in item:
                item['currently_active'] = False
            answer[key].append(item)
        return answer, 'recurrent_records'

    def decorate_system(self, system):
        original = system.respond

        def respond(query):
            response = original(query)
            raw = response.action.model_dump()
            final, operator = self._project(raw)
            action = query.response_schema.model_validate(final)
            if system.last is None:
                raise RuntimeError('No actor action to project')
            system.last = (system.last[0], action.model_dump())
            self.pending_raw_action = raw
            save(system.output / 'policy_action.json', dict(actor=raw, final=action.model_dump(),
                 operator=operator, previous_actions=len(self.raw_actions)))
            return type(response)(action=action, metadata=response.metadata)

        system.respond = respond

    def update(self, query, public_trace, reward, success, retrieval, binding):
        result = super().update(query, public_trace, reward, success, retrieval, binding)
        if self.pending_raw_action is not None:
            self.raw_actions.append(self.pending_raw_action)
            self.pending_raw_action = None
        return result

    def snapshot(self, path):
        super().snapshot(path)
        data = read(path)
        data['raw_actions'] = self.raw_actions
        save(path, data)

    def restore(self, path):
        super().restore(path)
        data = read(path)
        if 'raw_actions' not in data:
            raise ValueError('Typed-action history absent')
        self.raw_actions = data['raw_actions']
