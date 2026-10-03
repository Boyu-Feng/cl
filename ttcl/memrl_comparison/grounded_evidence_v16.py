"""General typed-state readout over source-grounded MemRL public events."""
from __future__ import annotations

import json
import hashlib
import re

from ttcl.icl_mem0_comparison.protocol import read, save, seed
from .grounded_evidence_v13_base import (
    GroundedEvidenceMemory, MAX_EVIDENCE_EVENTS, _clip, _json_object)
from .memory import Memory, digest
from .typed_executor import apply_candidate
from .typed_projection import typed_candidates


OPERATIONS = {'numeric_consensus': 'NUMERIC_CONSENSUS',
              'recurring_records': 'RECURRING_RECORDS'}


def _head(tokenizer, value, limit):
    content = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    ids = tokenizer.encode(content, add_special_tokens=False)
    return tokenizer.decode(ids[:limit], skip_special_tokens=True)


def _task_focus(query):
    """Remove repeated action instructions from the retrieval query."""
    sections = re.split(r'\n\s*\n', query.strip())
    return '\n\n'.join(sections[:2]) if sections else query


def _informative_feedback(value):
    """Keep observations with substantive content beyond a status header."""
    content = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    sections = re.split(r'\n\s*\n', content, maxsplit=1)
    payload = sections[-1].strip()
    if re.match(r'(?i)^(?:error|exception|failed)\s*[:\s]', payload):
        return False
    terms = set(re.findall(r'\w+', payload.lower()))
    return len(payload) >= 24 and len(terms) >= 3


class TypedGroundedMemory(GroundedEvidenceMemory):
    """Freeze the public task contract, then execute compatible typed operators."""

    def __init__(self, plan, client, directory, calibration, embedder=None):
        super().__init__(plan, client, directory, calibration, embedder=embedder)
        self.target_contract = None
        self._text_retry = False
        self._text_attempt = 0
        self._had_initial_memory = None
        self.signature = digest(dict(base=self.signature, variant='typed_grounded_v16',
                                     operations=OPERATIONS,
                                     retry_policy='initial_context_adaptive_dropout',
                                     evidence_retrieval='nonterminal_success_feedback_two_per_episode'))

    def begin_task(self, interface, binding):
        if interface not in ('retryable_text_command', 'json_schema'):
            raise ValueError('Unknown action interface')
        self._text_retry = interface == 'retryable_text_command'
        self._text_attempt = 0
        self._had_initial_memory = None
        self._current_task_binding = binding

    def retrieve(self, query):
        if not self._text_retry:
            native = Memory.retrieve(self, query)
            tokenizer = self.client.tokenizer
            unique = {}
            for event in self.public_events:
                if (event.get('instance_complete', False) or
                    not _informative_feedback(event['observed_feedback'])):
                    continue
                fingerprint = digest(dict(action=event['submitted_action'],
                                          feedback=event['observed_feedback']))
                unique[fingerprint] = event
            candidates = list(unique.values())
            texts = [(_clip(tokenizer, _task_focus(event['public_input']), 160) + '\n' +
                      _head(tokenizer, event['submitted_action'], 80) + '\n' +
                      _head(tokenizer, event['observed_feedback'], 200))
                     for event in candidates]
            if candidates:
                vectors = self.service.embedding_provider.embed(
                    [_clip(tokenizer, _task_focus(query), 256)] + texts)
                scores = [sum(a * b for a, b in zip(vectors[0], vector))
                          for vector in vectors[1:]]
                ranked = sorted(range(len(candidates)),
                                key=lambda i: (scores[i], candidates[i]['episode']),
                                reverse=True)
            else:
                scores, ranked = [], []
            context = native['context']
            public_context = ''
            selected = []
            seen_episodes = {}
            for index in ranked:
                event = candidates[index]
                if seen_episodes.get(event['episode'], 0) >= 2:
                    continue
                entry = ('Public event source=' + event['source_sha256'] + '\n'
                         'Prior public input (brief): ' + _clip(
                             tokenizer, event['public_input'], 80) + '\n'
                         'Prior submitted action (unverified): ' + _clip(
                             tokenizer, event['submitted_action'], 160) + '\n'
                         'Observed environment response: ' + _head(
                             tokenizer, event['observed_feedback'], 320))
                proposed = (context + '\n\n' + entry).strip()
                if self._tokens(proposed) > self.plan['memory_tokens']:
                    continue
                context = proposed
                public_context = (public_context + '\n\n' + entry).strip()
                selected.append(dict(source_sha256=event['source_sha256'],
                                     public_trace_sha256=event['public_trace_sha256'],
                                     episode=event['episode'], step=event['step'],
                                     similarity=scores[index]))
                seen_episodes[event['episode']] = seen_episodes.get(event['episode'], 0) + 1
                if len(selected) >= MAX_EVIDENCE_EVENTS:
                    break
            self.selected_context = context
            self.selected_public_context = public_context
            self.selected_evidence = selected
            return dict(native, context=context, evidence=selected,
                        tokens=self._tokens(context),
                        context_sha256=hashlib.sha256(context.encode()).hexdigest(),
                        projection='nonterminal_success_feedback_two_per_episode_v14')
        native = Memory.retrieve(self, query)
        attempt = self._text_attempt
        self._text_attempt += 1
        if attempt == 0:
            self._had_initial_memory = bool(native['ids'])
        self.selected_evidence = []
        self.selected_context = native['context']
        self.selected_public_context = ''
        dropout_attempt = 2 if self._had_initial_memory else 1
        if attempt == dropout_attempt and self.plan['alf']['max_attempts'] >= 3:
            empty = ''
            return dict(native, context=empty, ids=[], tokens=0,
                        context_sha256=hashlib.sha256(empty.encode()).hexdigest(),
                        suppressed_ids=native['ids'],
                        retry_policy='initial_context_adaptive_dropout',
                        had_initial_memory=self._had_initial_memory,
                        action_interface='retryable_text_command')
        return dict(native, evidence=[], retry_policy='native_exact',
                    had_initial_memory=self._had_initial_memory,
                    action_interface='retryable_text_command')

    def _classify_target(self, initial_task, system):
        if self.target_contract is not None:
            return self.target_contract
        brief = initial_task.split('=======================', 1)[0].strip()
        brief = _clip(self.client.tokenizer, brief or initial_task, 700)
        messages = [dict(role='system', content=(
            'Classify the public task objective, without solving an instance. '
            'Return exactly one JSON object with boolean persistent_set and '
            'shared_numeric_target, plus a short reason. persistent_set is '
            'true only if final actions should carry entities from earlier '
            'observations even when absent now. shared_numeric_target is true '
            'only if sequential instances estimate the same named numerical '
            'target fields. Ignore current samples, submissions, and rewards.')),
            dict(role='user', content='Public task brief for classification only:\n' + brief)]
        contract = dict(persistent_set=False, shared_numeric_target=False,
                        reason='Classification unavailable', brief_sha256=digest(brief))
        try:
            completion = self.client.complete(messages,
                seed(self.client.repeat, brief, 'typed_grounded_v9_contract'),
                tokens=256, temperature=0., top_p=1.)
            system.calls += 1
            system.input_tokens += completion['input_tokens']
            system.output_tokens += completion['output_tokens']
            system.max_input_tokens = max(system.max_input_tokens,
                                          completion['input_tokens'])
            choice = _json_object(completion['raw_response'])
            for field in ('persistent_set', 'shared_numeric_target'):
                if type(choice.get(field)) is not bool:
                    raise ValueError(f'Missing boolean {field}')
            contract.update(persistent_set=choice['persistent_set'],
                            shared_numeric_target=choice['shared_numeric_target'],
                            reason=str(choice.get('reason', ''))[:500])
            save(system.output / 'typed_contract.json', dict(
                messages=messages, completion=completion, contract=contract))
        except Exception as exc:
            contract['error'] = repr(exc)
            save(system.output / 'typed_contract.json', dict(
                messages=messages, contract=contract))
        self.target_contract = contract
        return contract

    def decorate_system(self, system):
        original = system.respond
        reviewed = False
        self._native_context_was_shown = False

        def respond(query):
            nonlocal reviewed
            schema_json = json.dumps(
                query.response_schema.model_json_schema(), ensure_ascii=False)
            schema_chars = len(schema_json)
            marker = '\n\nPast experience:\n'
            if schema_chars > 10000:
                instruction = system.messages[0]['content']
                if marker in instruction:
                    system.messages[0]['content'] = instruction.split(marker, 1)[0]
                    save(system.output / 'typed_context_budget.json', dict(
                        decision='remove_past_context_for_large_action_schema',
                        schema_chars=schema_chars, turn=system.turn + 1,
                        previous_context_sha256=digest(instruction)))
            elif '"tool_call"' in schema_json and marker in system.messages[0]['content']:
                instruction, context = system.messages[0]['content'].split(marker, 1)
                evidence = self.selected_public_context
                if evidence and context.endswith(evidence):
                    native = context[:-len(evidence)].rstrip()
                    system.messages[0]['content'] = (instruction + marker + native
                                                     if native else instruction)
                    save(system.output / 'typed_context_budget.json', dict(
                        decision='remove_unverified_event_context_for_tool_schema',
                        schema_chars=schema_chars, turn=system.turn + 1,
                        previous_context_sha256=digest(instruction + marker + context),
                        native_context_retained=bool(native)))
                    self._native_context_was_shown = bool(native)
                else:
                    self._native_context_was_shown = True
            elif marker in system.messages[0]['content']:
                self._native_context_was_shown = True
            response = original(query)
            actor_action = response.action.model_dump()
            audit = dict(actor=actor_action, final=actor_action, operation='KEEP',
                         reason=None, typed_candidate_kinds=[])
            if 'tool_call' in actor_action:
                audit['reason'] = 'Environment-changing tool action'
            elif reviewed:
                audit['reason'] = 'One-selection budget exhausted'
            else:
                candidates = typed_candidates(self.public_events, actor_action)
                audit['typed_candidate_kinds'] = [c['kind'] for c in candidates]
                if not candidates:
                    audit['reason'] = 'No type-compatible historical candidate'
                else:
                    reviewed = True
                    current_start = (system.public_steps[0]['query']
                                     if system.public_steps else query.prompt)
                    try:
                        contract = self._classify_target(current_start, system)
                        audit['target_contract'] = contract
                        applicable = [c for c in candidates if (
                            c['kind'] == 'numeric_consensus' and
                            contract['shared_numeric_target']) or (
                            c['kind'] == 'recurring_records' and
                            contract['persistent_set'])]
                        if applicable:
                            candidate = max(applicable, key=lambda c: (
                                c.get('sample_count', 0) if c['kind'] ==
                                'numeric_consensus' else sum(
                                    r['seen_in_episodes'] for r in c['records'])))
                            proposed = apply_candidate(actor_action, candidate,
                                                       self.episode_count)
                            action = query.response_schema.model_validate(proposed)
                            if system.last is None:
                                raise ValueError('Missing actor action')
                            system.last = (system.last[0], action.model_dump())
                            response = type(response)(action=action,
                                                      metadata=response.metadata)
                            audit.update(final=action.model_dump(),
                                         operation=OPERATIONS[candidate['kind']],
                                         candidate=candidate)
                        else:
                            audit['reason'] = 'No candidate matches frozen target contract'
                    except Exception as exc:
                        audit['reason'] = 'Rejected contract or projection: ' + str(exc)
            save(system.output / f'typed_action_{system.turn:03d}.json', audit)
            return response

        system.respond = respond

    def update(self, query, public_trace, reward, success, retrieval, binding):
        # Do not give Q credit to native memories that were removed before the
        # first actor action in a large-schema instance.
        if getattr(self, '_native_context_was_shown', True) is False:
            retrieval = dict(retrieval, ids=[])
        return super().update(query, public_trace, reward, success, retrieval,
                              binding)

    def snapshot(self, path):
        super().snapshot(path)
        state = read(path)
        state['typed_target_contract'] = self.target_contract
        save(path, state)

    def restore(self, path):
        super().restore(path)
        state = read(path)
        if 'typed_target_contract' not in state:
            raise ValueError('Typed target contract missing')
        self.target_contract = state['typed_target_contract']
