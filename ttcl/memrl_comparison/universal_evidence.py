"""One evidence-grounded memory update and retrieval rule for all CLBench tasks.

Every episode yields immutable public events. A single model prompt proposes
add/revise/retire operations on a compact evidence notebook. Operations must
cite existing event hashes; unmentioned entries remain. The same notebook and
retrieval rule is used for every task, with no action-schema inspection or
final-action rewrite. Official reward is used only by native MemRL's Q update.
"""
from __future__ import annotations

import hashlib
import json

from ttcl.icl_mem0_comparison.protocol import read, save, seed
from .memory import Memory, digest


REASONING_FIELDS = {'thought', 'thinking', 'reasoning', 'analysis'}


def _public(value, depth=0):
    if depth > 4:
        return None
    if isinstance(value, dict):
        return {str(k): _public(v, depth + 1) for k, v in value.items()
                if k not in REASONING_FIELDS}
    if isinstance(value, list):
        return [_public(v, depth + 1) for v in value[:32]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def public_events(public_trace: str, episode: int) -> list[dict]:
    steps = json.loads(public_trace)
    if not isinstance(steps, list):
        raise ValueError('Expected public step list')
    events = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        query = str(step.get('query') or '')
        event = dict(episode=episode, step=index + 1,
                     query_excerpt=(query[:600] + ('\n[...]\n' if len(query) > 1400 else '') +
                                    query[-800:] if len(query) > 1400 else query),
                     action=_public(step.get('action')),
                     feedback=str(step.get('public_feedback') or '')[:1200])
        event['id'] = 'e' + digest(event)[:16]
        events.append(event)
    return events


def _json_object(raw: str):
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw):
        if char == '{':
            try:
                value, _ = decoder.raw_decode(raw[index:])
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                continue
    raise ValueError('No JSON object in notebook update')


def _operations(raw: str):
    """Recover only complete JSON operations if generation hit its token cap."""
    try:
        value = _json_object(raw)
        operations = value.get('operations')
        if not isinstance(operations, list):
            raise ValueError('Notebook operations must be a list')
        return operations, False
    except ValueError:
        marker = raw.find('"operations"')
        start = raw.find('[', marker) if marker >= 0 else -1
        if start < 0:
            raise
        decoder = json.JSONDecoder()
        operations = []
        position = start + 1
        while True:
            position += len(raw[position:]) - len(raw[position:].lstrip(' \r\n\t,'))
            if position >= len(raw) or raw[position] == ']':
                break
            try:
                value, used = decoder.raw_decode(raw[position:])
            except json.JSONDecodeError:
                break
            if not isinstance(value, dict):
                break
            operations.append(value)
            position += used
        if not operations:
            raise ValueError('No complete notebook operations')
        return operations, True


class UniversalEvidenceMemory(Memory):
    """Native MemRL plus a source-bound, cumulative public-evidence notebook."""

    def __init__(self, plan, client, directory, calibration, embedder=None):
        self.events = {}
        self.entries = {}
        self.next_entry = 1
        self.episode_count = 0
        self.update_records = []
        self.notebook_input_tokens = 0
        self.notebook_output_tokens = 0
        super().__init__(plan, client, directory, calibration, embedder=embedder)
        self.signature = digest(dict(base=self.signature, variant='universal_evidence_v2',
                                     max_entries=128, max_operations=8, max_new_events=12,
                                     notebook_tokens=1536))

    def _tokens(self, text):
        return len(self.client.tokenizer.encode(text, add_special_tokens=False))

    def _notebook_prompt(self, fresh):
        entries = [dict(id=key, claim=value['claim'], support=value['support'])
                   for key, value in self.entries.items()]
        # The durable event store retains full public records. The writer sees
        # bounded excerpts in one fixed budget for every CLBench task.
        new = [dict(id=e['id'], query=e['query_excerpt'], action=e['action'],
                    feedback=e['feedback']) for e in fresh[-12:]]
        id_rule = ('There are no existing notebook entries: use add only.' if not entries else
                   'For revise or retire, id must be an existing m-prefixed notebook ID. '
                   'Event IDs beginning with e belong only in support, never in id.')
        return [dict(role='system', content=(
            'Maintain a compact cross-episode evidence notebook. Use only public '
            'observations, actions and feedback. Do not treat an unverified action '
            'or hypothesis as ground truth. Return JSON only: '
            '{"operations":[{"op":"add|revise|retire","id":"existing id for '
            'revise/retire","claim":"short conditional fact","support":["event id"]}]}. '
            'Every operation must cite at least one NEW event id. Revise or '
            'retire only when new evidence warrants it. Unmentioned entries persist. '
            'Use at most eight operations. Prefer a revised cumulative pattern '
            'over one entry per raw observation. Keep claims short, reusable, '
            'source specific, and free of opaque instance IDs. ' + id_rule)),
            dict(role='user', content=json.dumps(dict(current_entries=entries,
                new_public_events=new), ensure_ascii=False, separators=(',', ':')))]

    def _apply_operations(self, operations, fresh):
        if not isinstance(operations, list):
            raise ValueError('Invalid notebook operations')
        known = set(self.events) | {e['id'] for e in fresh}
        new_ids = {e['id'] for e in fresh}
        entries = {k: dict(v) for k, v in self.entries.items()}
        next_entry = self.next_entry
        applied, rejected = [], []
        for index, operation in enumerate(operations[:8]):
            try:
                if not isinstance(operation, dict) or operation.get('op') not in {'add', 'revise', 'retire'}:
                    raise ValueError('Invalid notebook operation')
                support = operation.get('support')
                if (not isinstance(support, list) or not support or len(support) > 8
                        or not all(isinstance(x, str) and x in known for x in support)
                        or not new_ids.intersection(support)):
                    raise ValueError('Notebook citation is missing or unbound')
                kind = operation['op']
                if kind == 'add':
                    if len(entries) >= 128:
                        raise ValueError('Notebook capacity exceeded')
                    key = f'm{next_entry:06d}'
                else:
                    key = operation.get('id')
                    if key not in entries:
                        raise ValueError('Notebook entry ID is missing')
                if kind != 'retire':
                    claim = operation.get('claim')
                    if not isinstance(claim, str) or not 1 <= len(claim.strip()) <= 320:
                        raise ValueError('Notebook claim is missing or too long')
                if kind == 'add':
                    next_entry += 1
                if kind == 'retire':
                    del entries[key]
                else:
                    prior = entries.get(key, {})
                    entries[key] = dict(claim=claim.strip(),
                                        support=list(dict.fromkeys(prior.get('support', []) + support))[-8:])
                applied.append(dict(op=kind, id=key, support=support))
            except (ValueError, TypeError, KeyError) as exc:
                rejected.append(dict(index=index, reason=str(exc)))
        return entries, next_entry, applied, rejected, max(0, len(operations) - 8)

    def retrieve(self, query):
        base = super().retrieve(query)
        if not self.entries:
            return dict(base, evidence_entries=[], projection='native_until_public_evidence')
        keys = list(self.entries)
        claims = [self.entries[k]['claim'] for k in keys]
        vectors = self.service.embedding_provider.embed([query] + claims)
        ranked = sorted(range(len(keys)), key=lambda i: sum(a*b for a, b in zip(vectors[0], vectors[i+1])),
                        reverse=True)
        context, evidence_ids = '', []
        notebook_cap = min(1024, self.plan['memory_tokens'] // 2)
        for index in ranked:
            key = keys[index]
            item = self.entries[key]
            line = f"[{key}; sources={','.join(item['support'])}] {item['claim']}"
            proposed = (context + '\n' + line).strip()
            if self._tokens(proposed) > notebook_cap:
                continue
            context, evidence_ids = proposed, evidence_ids + [key]
            if len(evidence_ids) == 12:
                break
        ids = []
        for mid in base['selected_before_budget']:
            metadata = self.store.get(mid).metadata.model_dump()
            task = str(metadata.get('task_description') or '').strip()
            lesson = str(metadata.get('public_abstract') or '').strip()
            if not lesson:
                continue
            task_tokens = self.client.tokenizer.encode(task, add_special_tokens=False)[:100]
            lesson_tokens = self.client.tokenizer.encode(lesson, add_special_tokens=False)[:300]
            task_snippet = self.client.tokenizer.decode(task_tokens, skip_special_tokens=True)
            lesson_snippet = self.client.tokenizer.decode(lesson_tokens, skip_special_tokens=True)
            entry = f"MemRL memory {mid}\nTask: {task_snippet}\nExperience: {lesson_snippet}"
            proposed = (context + '\n' + entry).strip()
            if self._tokens(proposed) <= self.plan['memory_tokens']:
                context, ids = proposed, ids + [mid]
        return dict(base, context=context, ids=ids, tokens=self._tokens(context),
                    evidence_entries=evidence_ids, projection='public_evidence_notebook_v2',
                    dropped_whole_entries=[mid for mid in base['selected_before_budget'] if mid not in ids],
                    context_sha256=hashlib.sha256(context.encode()).hexdigest())

    def update(self, query, public_trace, reward, success, retrieval, binding):
        fresh = public_events(public_trace, self.episode_count + 1)
        if fresh:
            prompt = self._notebook_prompt(fresh)
            raw = self.client.complete(prompt, seed(self.client.repeat, 'universal_notebook',
                                                   self.episode_count), tokens=1536,
                                       temperature=0., top_p=1.)
            self.notebook_input_tokens += raw['input_tokens']
            self.notebook_output_tokens += raw['output_tokens']
        else:
            raw = dict(raw_response='{"operations":[]}', input_tokens=0, output_tokens=0)
        failure = None
        try:
            operations, truncated = _operations(raw['raw_response'])
            entries, next_entry, applied, rejected, omitted = self._apply_operations(operations, fresh)
        except (ValueError, TypeError, KeyError) as exc:
            failure = repr(exc)
            entries, next_entry, applied, rejected, omitted, truncated = (
                self.entries, self.next_entry, [], [], 0, False)
        result = super().update(query, public_trace, reward, success, retrieval, binding)
        self.events.update({e['id']: e for e in fresh})
        self.entries, self.next_entry = entries, next_entry
        self.episode_count += 1
        record = dict(episode=self.episode_count, public_trace_sha256=hashlib.sha256(
                      public_trace.encode()).hexdigest(), new_event_ids=[e['id'] for e in fresh],
                      applied=applied, rejected=rejected, omitted_operations=omitted,
                      truncated_response=truncated, failure=failure,
                      writer_response_sha256=hashlib.sha256(
                      raw['raw_response'].encode()).hexdigest())
        self.update_records.append(record)
        save(self.directory / f'notebook_update_{self.episode_count:03d}.json',
             dict(record=record, raw_response=raw['raw_response']))
        return dict(result, notebook_update=record)

    def snapshot(self, path):
        super().snapshot(path)
        state = read(path)
        state['universal_evidence'] = dict(events=self.events, entries=self.entries,
            next_entry=self.next_entry, episode_count=self.episode_count,
            update_records=self.update_records,
            notebook_input_tokens=self.notebook_input_tokens,
            notebook_output_tokens=self.notebook_output_tokens)
        save(path, state)

    def restore(self, path):
        super().restore(path)
        state = read(path).get('universal_evidence')
        if state is None:
            raise ValueError('Universal evidence state absent')
        for key in ('events', 'entries', 'next_entry', 'episode_count', 'update_records',
                    'notebook_input_tokens', 'notebook_output_tokens'):
            setattr(self, key, state[key])
