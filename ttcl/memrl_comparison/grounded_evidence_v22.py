"""Break repeated, escalating structured actions with one schema-valid recheck.

The guard is based only on public actions and their numeric arguments. It
does not name a benchmark or choose an alternative itself. The original
action is retained if the bounded recheck does not yield a valid different
action type. All extra model calls and rejected replies are recorded.
"""
from __future__ import annotations

import copy
import math

from ttcl.icl_mem0_comparison.protocol import append, read, save
from ttcl.structured_memory import run_benchmark as base
from .grounded_evidence_v21 import TypedGroundedMemory as V21Memory
from .memory import digest


def escalating_loop(public_steps: list[dict], current: dict) -> dict | None:
    if len(public_steps) < 2 or not isinstance(current, dict):
        return None
    actions = [step.get('action') for step in public_steps[-2:]] + [current]
    if any(not isinstance(action, dict) or not isinstance(action.get('action'), str)
           for action in actions):
        return None
    kind = current['action']
    if not kind or any(action['action'] != kind for action in actions):
        return None
    common = set.intersection(*(set(action) for action in actions)) - {'action', 'thinking'}
    for key in sorted(common):
        values = [action[key] for action in actions]
        if (all(isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) for value in values) and
                values[0] < values[1] < values[2]):
            return dict(kind=kind, numeric_key=key, numeric_values=values,
                        prior_public_step_numbers=[step.get('step')
                                                   for step in public_steps[-2:]])
    return None


class TypedGroundedMemory(V21Memory):
    def __init__(self, plan, client, directory, calibration, embedder=None):
        super().__init__(plan, client, directory, calibration, embedder=embedder)
        self.signature = digest(dict(base=self.signature, variant='typed_grounded_v22',
                                     rule='bounded_escalating_action_loop_guard'))

    def decorate_system(self, system):
        super().decorate_system(system)
        previous = system.respond

        def respond(query):
            response = previous(query)
            actor_action = response.action.model_dump()
            pattern = escalating_loop(system.public_steps, actor_action)
            if pattern is None:
                return response
            original = copy.deepcopy(actor_action)
            messages = copy.deepcopy(system.messages)
            messages.append(dict(role='user', content=(
                'The proposed action has the same action type as the previous '
                'two executed actions, while its numeric argument rises each '
                'time. Recheck the current public situation and choose a '
                'different permitted action type if one is valid. Do not '
                'assume that repeating an escalating action helps. Return '
                'one complete JSON action matching the current schema.')))
            random_seed = base.generation_seed(
                system.plan['task_seed'], query.instance_id,
                f'{system.turn}:escalating_loop_recheck')
            completion = system.model.generate(messages, random_seed)
            system.calls += 1
            system.input_tokens += completion['input_tokens']
            system.output_tokens += completion['output_tokens']
            system.max_input_tokens = max(system.max_input_tokens,
                                          completion['input_tokens'])
            event = dict(call=system.calls, turn=system.turn,
                         loop_guard=True, instance_id=query.instance_id,
                         pattern=pattern, messages=messages, **completion)
            accepted = False
            try:
                candidate, packaging_repair = base.normalize_action(
                    completion['raw_response'], query.response_schema)
                alternative = candidate.model_dump()
                accepted = alternative.get('action') != pattern['kind']
                event.update(action=alternative, packaging_repair=packaging_repair,
                             accepted=accepted, parse_error=None)
            except ValueError as exc:
                event.update(accepted=False, parse_error=str(exc))
            append(system.output / 'responses.jsonl', event)
            audit_path = system.output / f'typed_action_{system.turn:03d}.json'
            audit = read(audit_path)
            audit['loop_guard'] = dict(pattern=pattern, accepted=accepted,
                                       extra_call=system.calls,
                                       original_action=original,
                                       prompt_sha256=digest(messages))
            if accepted:
                system.messages[-1] = dict(role='assistant',
                                           content=completion['raw_response'])
                system.last = (system.last[0], alternative)
                audit.update(final=alternative, operation='ESCALATION_RECHECK')
                response = type(response)(action=candidate,
                                          metadata=response.metadata)
            save(audit_path, audit)
            return response

        system.respond = respond
