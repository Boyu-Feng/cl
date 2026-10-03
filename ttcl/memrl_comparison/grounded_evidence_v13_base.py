"""Source-grounded, task-independent evidence for the MemRL adaptation.

Public input, an agent submission, and environment feedback are separate
fields. A submission is never promoted to a verified observation. The same
event extraction, retrieval, and budget rule serves ALFWorld and CLBench.
"""
from __future__ import annotations

import hashlib
import json
import math

from ttcl.icl_mem0_comparison.protocol import append, read, save, seed
from .memory import Memory, digest
from .typed_projection import typed_candidates


PRIVATE = {'thought', 'thinking', 'reasoning', 'analysis'}
MAX_EVENTS_PER_EPISODE = 32
MAX_EVIDENCE_EVENTS = 5


def public_value(value, depth=0):
    if depth > 8:
        return '[nested value omitted]'
    if isinstance(value, dict):
        return {str(k): public_value(v, depth + 1) for k, v in value.items()
                if str(k).lower() not in PRIVATE}
    if isinstance(value, list):
        return [public_value(v, depth + 1) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Non-finite public value')
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def public_events(public_trace, episode):
    trace = json.loads(public_trace)
    steps = trace.get('trajectory', []) if isinstance(trace, dict) else trace
    if not isinstance(steps, list):
        raise ValueError('Public trace must contain steps')
    trace_hash = hashlib.sha256(public_trace.encode()).hexdigest()
    events = []
    previous_observation = trace.get('initial_observation', '') if isinstance(trace, dict) else ''
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        observation = step.get('public_feedback', step.get('observation'))
        if observation is None:
            continue
        event = dict(episode=episode, step=index + 1,
                     public_input=public_value(step.get('query', previous_observation)),
                     submitted_action=public_value(step.get('action')),
                     observed_feedback=public_value(observation),
                     instance_complete=bool(step.get('instance_complete', False)),
                     public_trace_sha256=trace_hash)
        event['source_sha256'] = digest(event)
        events.append(event)
        previous_observation = observation
    return events[-MAX_EVENTS_PER_EPISODE:]


def _clip(tokenizer, value, limit):
    text = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= limit:
        return text
    front = tokenizer.decode(ids[:limit // 2], skip_special_tokens=True)
    back = tokenizer.decode(ids[-(limit - limit // 2):], skip_special_tokens=True)
    return front + '\n[omitted middle; inspect source hash]\n' + back


def _json_object(raw):
    decoder = json.JSONDecoder()
    for offset, char in enumerate(raw):
        if char == '{':
            try:
                value, _ = decoder.raw_decode(raw[offset:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise ValueError('No complete JSON object')


class GroundedEvidenceMemory(Memory):
    """Native Q learning plus an exact public-event store and bounded readout."""

    def __init__(self, plan, client, directory, calibration, embedder=None):
        self.public_events = []
        self.episode_count = 0
        self.selected_evidence = []
        self.selected_context = ''
        self.selected_public_context = ''
        super().__init__(plan, client, directory, calibration, embedder=embedder)
        self.signature = digest(dict(base=self.signature, variant='grounded_evidence_v3',
                                     event_cap=MAX_EVENTS_PER_EPISODE,
                                     retrieval_cap=MAX_EVIDENCE_EVENTS,
                                     memory_tokens=plan['memory_tokens']))

    def _tokens(self, text):
        return len(self.client.tokenizer.encode(text, add_special_tokens=False))

    def retrieve(self, query):
        # Native ranking/Q remain intact. An absolute similarity threshold can
        # hide every prior event, so the public ledger has a separate soft rank.
        native = super().retrieve(query)
        candidates = self.public_events
        if candidates:
            texts = [_clip(self.client.tokenizer, e['public_input'], 256)
                     for e in candidates]
            vectors = self.service.embedding_provider.embed(
                [_clip(self.client.tokenizer, query, 256)] + texts)
            scores = [sum(a*b for a, b in zip(vectors[0], vector))
                      for vector in vectors[1:]]
            ranked = sorted(range(len(candidates)),
                            key=lambda i: (scores[i], candidates[i]['episode']),
                            reverse=True)
        else:
            scores, ranked = [], []

        budget = self.plan['memory_tokens']
        # Keep useful native MemRL text first, but leave room for raw evidence.
        native_budget = budget // 2 if candidates else budget
        context, ids = '', []
        for mid in native['ids']:
            metadata = self.store.get(mid).metadata.model_dump()
            task = _clip(self.client.tokenizer, metadata.get('task_description') or '', 80)
            lesson = _clip(self.client.tokenizer, metadata.get('public_abstract') or '', 220)
            entry = f'Past task: {task}\nMemRL lesson (unverified): {lesson}'
            proposed = (context + '\n\n' + entry).strip()
            if self._tokens(proposed) <= native_budget:
                context = proposed
                ids.append(mid)

        selected = []
        public_context = ''
        seen_episodes = set()
        for index in ranked:
            event = candidates[index]
            if event['episode'] in seen_episodes:
                continue
            entry = ('Public event source=' + event['source_sha256'] + '\n'
                     'Prior public input: ' + _clip(self.client.tokenizer,
                         event['public_input'], 180) + '\n'
                     'Prior submitted action (unverified): ' + _clip(
                         self.client.tokenizer, event['submitted_action'], 240) + '\n'
                     'Observed environment response: ' + _clip(
                         self.client.tokenizer, event['observed_feedback'], 180))
            proposed = (context + '\n\n' + entry).strip()
            if self._tokens(proposed) > budget:
                continue
            context = proposed
            public_context = (public_context + '\n\n' + entry).strip()
            selected.append(dict(source_sha256=event['source_sha256'],
                                 public_trace_sha256=event['public_trace_sha256'],
                                 episode=event['episode'], step=event['step'],
                                 similarity=scores[index]))
            seen_episodes.add(event['episode'])
            if len(selected) >= MAX_EVIDENCE_EVENTS:
                break
        self.selected_evidence = selected
        self.selected_context = context
        self.selected_public_context = public_context
        return dict(native, context=context, ids=ids, evidence=selected,
                    tokens=self._tokens(context),
                    dropped_whole_entries=[mid for mid in native['selected_before_budget']
                                           if mid not in ids],
                    context_sha256=hashlib.sha256(context.encode()).hexdigest(),
                    projection='grounded_evidence_v3')

    def decorate_system(self, system):
        """One schema-valid proposal on a direct action, never on tool calls."""
        original = system.respond
        evidence = list(self.selected_evidence)
        context = self.selected_public_context
        reviewed = False

        def respond(query):
            nonlocal reviewed
            response = original(query)
            actor_action = response.action.model_dump()
            audit = dict(actor=actor_action, final=actor_action,
                         decision='KEEP', sources=[], reason=None)
            if not evidence:
                audit['reason'] = 'No prior public evidence'
            elif reviewed:
                audit['reason'] = 'One-review budget exhausted'
            elif 'tool_call' in actor_action:
                audit['reason'] = 'Environment-changing tool action'
            else:
                reviewed = True
                candidates = typed_candidates(self.public_events, actor_action)
                candidate_sources = {source for candidate in candidates
                    for source in (candidate.get('source_sha256', []) if
                                   candidate['kind'] == 'numeric_consensus' else
                                   [source for item in candidate['records']
                                    for source in item['source_sha256']])}
                available = {x['source_sha256'] for x in evidence} | candidate_sources
                messages = [dict(role='system', content=(
                    'Independently construct the best current action from the task, '
                    'current public observations, and historical public evidence. '
                    'Historical submitted actions and type-derived candidates are '
                    'unverified; an acknowledgment does not verify them. The '
                    'candidates are generic numeric consensus or recurring records '
                    'computed from repeated JSON action shapes. Use a candidate '
                    'only if the current task calls for that kind of aggregation. '
                    'Preserve exact numerical detail when the task requires it. '
                    'Return JSON with decision KEEP or REVISE, '
                    'action conforming to the current schema, and sources listing '
                    'prior source hashes actually used. If historical evidence '
                    'cannot justify a change, choose KEEP. Never invent feedback.')),
                    dict(role='user', content=json.dumps(dict(
                        current_task=query.prompt,
                        current_public_steps=public_value(system.public_steps[-3:]),
                        actor_proposal=public_value(actor_action),
                        response_schema=query.response_schema.model_json_schema(),
                        historical_evidence=context,
                        typed_candidates=candidates,
                    ), ensure_ascii=False, allow_nan=False))]
                audit['typed_candidate_kinds'] = [x['kind'] for x in candidates]
                try:
                    completion = self.client.complete(
                        messages, seed(self.client.repeat, query.instance_id,
                                       system.turn, 'grounded_evidence_v3'),
                        tokens=4096, temperature=0., top_p=1.)
                    system.calls += 1
                    system.input_tokens += completion['input_tokens']
                    system.output_tokens += completion['output_tokens']
                    system.max_input_tokens = max(system.max_input_tokens,
                                                  completion['input_tokens'])
                    append(system.output / 'grounded_review.jsonl',
                           dict(messages=messages, completion=completion))
                    proposal = _json_object(completion['raw_response'])
                    cited = proposal.get('sources', [])
                    if not isinstance(cited, list) or not all(
                            isinstance(x, str) and x in available for x in cited):
                        raise ValueError('Unknown evidence source')
                    if proposal.get('decision') == 'REVISE':
                        if not cited:
                            raise ValueError('Revision needs prior evidence')
                        action = query.response_schema.model_validate(proposal['action'])
                        if system.last is None:
                            raise ValueError('Missing actor action')
                        system.last = (system.last[0], action.model_dump())
                        response = type(response)(action=action, metadata=response.metadata)
                        audit.update(final=action.model_dump(), decision='REVISE',
                                     sources=cited)
                    elif proposal.get('decision') == 'KEEP':
                        audit['sources'] = cited
                    else:
                        raise ValueError('Unknown review decision')
                except Exception as exc:
                    audit['reason'] = 'Rejected review: ' + str(exc)
            save(system.output / f'grounded_action_{system.turn:03d}.json', audit)
            return response

        system.respond = respond

    def update(self, query, public_trace, reward, success, retrieval, binding):
        new_events = public_events(public_trace, self.episode_count + 1)
        result = super().update(query, public_trace, reward, success, retrieval, binding)
        self.public_events.extend(new_events)
        self.episode_count += 1
        return dict(result, public_event_sources=[e['source_sha256'] for e in new_events],
                    public_trace_sha256=hashlib.sha256(public_trace.encode()).hexdigest())

    def snapshot(self, path):
        super().snapshot(path)
        state = read(path)
        state['grounded_evidence'] = dict(events=self.public_events,
                                          episode_count=self.episode_count)
        save(path, state)

    def restore(self, path):
        super().restore(path)
        state = read(path).get('grounded_evidence')
        if state is None:
            raise ValueError('Grounded evidence state absent')
        self.public_events = state['events']
        self.episode_count = state['episode_count']
