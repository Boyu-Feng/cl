from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from .grounded_evidence_v4 import TypedGroundedMemory
from .test_grounded_evidence import ReviewClient
from .test_protocol import Embeddings
from .typed_executor import apply_candidate


class VectorSchema:
    @staticmethod
    def model_json_schema():
        return {'type': 'object'}

    @staticmethod
    def model_validate(value):
        if len(value) != 8 or not all(isinstance(v, (int, float)) for v in value.values()):
            raise ValueError('Invalid vector')
        return SimpleNamespace(model_dump=lambda: value)


class LargeVectorSchema(VectorSchema):
    @staticmethod
    def model_json_schema():
        return {'description': 'field ' * 2200}


class SelectorClient(ReviewClient):
    selection = 'KEEP'

    def complete(self, messages, *args, **kwargs):
        if 'Select a general state operator' in messages[0]['content']:
            self.messages.append(messages)
            return dict(raw_response=json.dumps(dict(operation=self.selection,
                                                     reason='Task semantics')),
                        input_tokens=50, output_tokens=10, finish_reason='stop')
        return super().complete(messages, *args, **kwargs)


class TypedGroundedTests(unittest.TestCase):
    def test_deterministic_numeric_and_record_actions(self):
        current = {f'f{i}': .5 for i in range(8)}
        consensus = dict(kind='numeric_consensus', sample_count=2,
                         values={f'f{i}': .8 for i in range(8)})
        result = apply_candidate(current, consensus, 3)
        self.assertAlmostEqual(result['f0'], .7)
        self.assertEqual(current['f0'], .5)
        records = dict(kind='recurring_records', path='records',
                       identity_field='coordinate', tolerance=1.,
                       records=[dict(identity_value=42., seen_in_episodes=3,
                                     sample_record=dict(coordinate=42.2,
                                                        active=True, power=-50.),
                                     source_sha256=['source-1'])])
        projected = apply_candidate({'records': [{'coordinate': 8.,
                              'active': True, 'power': -20.}]}, records, 4)
        self.assertEqual(len(projected['records']), 2)
        self.assertEqual(projected['records'][1]['coordinate'], 42.)
        self.assertIs(projected['records'][1]['active'], False)
        self.assertEqual(len(apply_candidate(projected, records, 4)['records']), 2)

    def test_selector_executes_general_numeric_operator(self):
        with TemporaryDirectory() as tmp:
            plan = dict(upstream=str(Path(__file__).resolve().parents[2] /
                                     'current_work/MemRL'),
                        writer_max_tokens=768, memory_tokens=2048,
                        rl_config=dict(epsilon=0., alpha=.3, gamma=0., topk=3))
            client = SelectorClient()
            memory = TypedGroundedMemory(plan, client, tmp,
                dict(threshold=.5, mean=.5, std=.1), Embeddings())
            for i in (1, 2):
                trace = json.dumps([dict(query='same public target',
                    action={f'f{k}': i / 10 for k in range(8)},
                    public_feedback='Submission recorded')])
                memory.update('same public target', trace, 1., True,
                              memory.retrieve('same public target'), {})
            memory.retrieve('same public target')
            output = Path(tmp) / 'actor'
            output.mkdir()
            system = SimpleNamespace(output=output, public_steps=[], turn=1,
                calls=0, input_tokens=0, output_tokens=0, max_input_tokens=0,
                last=None, messages=[{'role': 'system', 'content':
                    'Solve task\n\nPast experience:\nHistorical text'}])

            def actor(query):
                value = {f'f{k}': .9 for k in range(8)}
                action = VectorSchema.model_validate(value)
                system.last = (query.prompt, value)
                return SimpleNamespace(action=action, metadata={})

            system.respond = actor
            memory.decorate_system(system)
            client.selection = 'NUMERIC_CONSENSUS'
            result = system.respond(SimpleNamespace(prompt='same public target',
                instance_id='new-target', response_schema=VectorSchema))
            self.assertAlmostEqual(result.action.model_dump()['f0'], .4)
            self.assertEqual(system.last[1], result.action.model_dump())
            self.assertEqual(system.calls, 1)

    def test_large_schema_suppresses_actor_memory_before_generation(self):
        with TemporaryDirectory() as tmp:
            plan = dict(upstream=str(Path(__file__).resolve().parents[2] /
                                     'current_work/MemRL'),
                        writer_max_tokens=768, memory_tokens=2048,
                        rl_config=dict(epsilon=0., alpha=.3, gamma=0., topk=3))
            memory = TypedGroundedMemory(plan, SelectorClient(), tmp,
                dict(threshold=.5, mean=.5, std=.1), Embeddings())
            output = Path(tmp) / 'large_actor'
            output.mkdir()
            system = SimpleNamespace(output=output, public_steps=[], turn=0,
                calls=0, input_tokens=0, output_tokens=0, max_input_tokens=0,
                last=None, messages=[{'role': 'system', 'content':
                    'Solve task\n\nPast experience:\nPotentially distracting text'}])

            def actor(query):
                self.assertEqual(system.messages[0]['content'], 'Solve task')
                value = {f'f{k}': .5 for k in range(8)}
                action = LargeVectorSchema.model_validate(value)
                system.last = (query.prompt, value)
                return SimpleNamespace(action=action, metadata={})

            system.respond = actor
            memory.decorate_system(system)
            system.respond(SimpleNamespace(prompt='current target',
                instance_id='large-schema', response_schema=LargeVectorSchema))
            self.assertFalse(memory._native_context_was_shown)
            self.assertTrue((output / 'typed_context_budget.json').exists())


if __name__ == '__main__':
    unittest.main()
