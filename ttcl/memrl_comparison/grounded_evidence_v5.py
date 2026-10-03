"""General typed-state readout over source-grounded MemRL public events."""
from __future__ import annotations

import json

from ttcl.icl_mem0_comparison.protocol import append, save, seed
from .grounded_evidence import GroundedEvidenceMemory, _clip, _json_object, public_value
from .memory import digest
from .typed_executor import apply_candidate
from .typed_projection import typed_candidates


OPERATIONS = {'numeric_consensus': 'NUMERIC_CONSENSUS',
              'recurring_records': 'RECURRING_RECORDS'}


class TypedGroundedMemory(GroundedEvidenceMemory):
    """Use the model only to select a general operator; execute it exactly."""

    def __init__(self, plan, client, directory, calibration, embedder=None):
        super().__init__(plan, client, directory, calibration, embedder=embedder)
        self.signature = digest(dict(base=self.signature, variant='typed_grounded_v5',
                                     operations=OPERATIONS))

    def decorate_system(self, system):
        original = system.respond
        reviewed = False
        self._native_context_was_shown = False

        def respond(query):
            nonlocal reviewed
            schema_chars = len(json.dumps(
                query.response_schema.model_json_schema(), ensure_ascii=False))
            if schema_chars > 10000:
                instruction = system.messages[0]['content']
                marker = '\n\nPast experience:\n'
                if marker in instruction:
                    system.messages[0]['content'] = instruction.split(marker, 1)[0]
                    save(system.output / 'typed_context_budget.json', dict(
                        decision='remove_past_context_for_large_action_schema',
                        schema_chars=schema_chars, turn=system.turn + 1,
                        previous_context_sha256=digest(instruction)))
            elif '\n\nPast experience:\n' in system.messages[0]['content']:
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
                    prior_starts = [event for event in self.public_events
                                    if event['step'] == 1]
                    current_start = (system.public_steps[0]['query']
                                     if system.public_steps else query.prompt)
                    messages = [dict(role='system', content=(
                        'Select a general state operator for the current task. '
                        'Prior submitted actions are unverified predictions, not '
                        'ground truth; acknowledgments do not validate them. '
                        'NUMERIC_CONSENSUS averages prior and current numerical '
                        'vectors with identical field names to reduce variation '
                        'across samples. Choose it when instances estimate the '
                        'same underlying target fields from different samples. '
                        'An already complete current vector can still benefit '
                        'from consensus; field completeness is irrelevant. '
                        'Do not choose it when each instance has a different '
                        'underlying target. RECURRING_RECORDS appends prior '
                        'record clusters absent from the current action. Choose '
                        'it only when the task explicitly asks for a persistent '
                        'set over time, including currently absent members. A '
                        'list complete for the current observation may still '
                        'omit persistent members. '
                        'Otherwise choose KEEP. Return exactly a JSON object '
                        'with operation KEEP, NUMERIC_CONSENSUS, or '
                        'RECURRING_RECORDS, and a short reason. Do not produce '
                        'an action or invent evidence.')),
                        dict(role='user', content=json.dumps(dict(
                            current_task=query.prompt,
                            episode_initial_task=_clip(self.client.tokenizer,
                                                       current_start, 320),
                            prior_episode_starts=[dict(source_sha256=e['source_sha256'],
                                public_input=_clip(self.client.tokenizer,
                                                   e['public_input'], 220))
                                for e in prior_starts[-3:]],
                            current_public_steps=public_value(system.public_steps[-3:]),
                            actor_action=public_value(actor_action),
                            candidate_summaries=[dict(kind=c['kind'],
                                sample_count=c.get('sample_count'),
                                path=c.get('path'),
                                identity_field=c.get('identity_field'),
                                recurrent_records=len(c.get('records', [])))
                                for c in candidates],
                        ), ensure_ascii=False, allow_nan=False))]
                    try:
                        completion = self.client.complete(
                            messages, seed(self.client.repeat, query.instance_id,
                                           system.turn, 'typed_grounded_v5'),
                            tokens=256, temperature=0., top_p=1.)
                        system.calls += 1
                        system.input_tokens += completion['input_tokens']
                        system.output_tokens += completion['output_tokens']
                        system.max_input_tokens = max(system.max_input_tokens,
                                                      completion['input_tokens'])
                        append(system.output / 'typed_selector.jsonl',
                               dict(messages=messages, completion=completion))
                        choice = _json_object(completion['raw_response'])
                        operation = choice.get('operation')
                        audit['selector_reason'] = choice.get('reason')
                        if operation == 'KEEP':
                            pass
                        elif operation in OPERATIONS.values():
                            kind = next(k for k, v in OPERATIONS.items()
                                        if v == operation)
                            candidate = next((c for c in candidates
                                              if c['kind'] == kind), None)
                            if candidate is None:
                                raise ValueError('Selected operator has no candidate')
                            proposed = apply_candidate(actor_action, candidate,
                                                       self.episode_count)
                            action = query.response_schema.model_validate(proposed)
                            if system.last is None:
                                raise ValueError('Missing actor action')
                            system.last = (system.last[0], action.model_dump())
                            response = type(response)(action=action,
                                                      metadata=response.metadata)
                            audit.update(final=action.model_dump(), operation=operation,
                                         candidate=candidate)
                        else:
                            raise ValueError('Unknown operator selection')
                    except Exception as exc:
                        audit['reason'] = 'Rejected selector or projection: ' + str(exc)
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
