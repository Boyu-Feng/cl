from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from ttcl.icl_mem0_comparison.protocol import read
from .grounded_evidence import GroundedEvidenceMemory, public_events
from .test_protocol import Client, Embeddings


class Tokenizer:
    def encode(self, value, **kwargs):
        return str(value).split()

    def decode(self, pieces, **kwargs):
        return ' '.join(pieces)


class ReviewClient(Client):
    tokenizer = Tokenizer()
    review = None

    def complete(self, messages, *args, **kwargs):
        if 'Independently construct' in messages[0]['content']:
            self.messages.append(messages)
            return dict(raw_response=json.dumps(self.review), input_tokens=40,
                        output_tokens=20, finish_reason='stop')
        return super().complete(messages, *args, **kwargs)


class NumberSchema:
    @staticmethod
    def model_json_schema():
        return {'type': 'object', 'properties': {'value': {'type': 'number'}}}

    @staticmethod
    def model_validate(value):
        if set(value) != {'value'} or not isinstance(value['value'], (int, float)):
            raise ValueError('Invalid number action')
        return SimpleNamespace(model_dump=lambda: value)


class GroundedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.plan = dict(upstream=str(Path(__file__).resolve().parents[2] /
                                      'current_work/MemRL'),
                         writer_max_tokens=768, memory_tokens=2048,
                         rl_config=dict(epsilon=0., alpha=.3, gamma=0., topk=3))
        self.client = ReviewClient()
        self.memory = GroundedEvidenceMemory(self.plan, self.client,
            self.tmp.name, dict(threshold=.5, mean=.5, std=.1), Embeddings())

    def test_public_events_keep_exact_values_but_separate_submission_from_feedback(self):
        values = {f'field_{i}': i / 108 for i in range(108)}
        trace = json.dumps([dict(query='public task', action=dict(values,
            thinking='private rationale'), public_feedback='Report recorded.')])
        event = public_events(trace, 1)[0]
        self.assertEqual(len(event['submitted_action']), 108)
        self.assertNotIn('thinking', event['submitted_action'])
        self.assertEqual(event['observed_feedback'], 'Report recorded.')
        self.assertNotIn('verified', event)
        self.assertEqual(len(event['source_sha256']), 64)
        alf = json.dumps(dict(initial_observation='Room description', trajectory=[
            dict(action='go north', observation='You enter the hall.'),
            dict(action='take cup', observation='You take the cup.')], reward=1.0,
            steps=2))
        steps = public_events(alf, 2)
        self.assertEqual(steps[0]['public_input'], 'Room description')
        self.assertEqual(steps[1]['public_input'], 'You enter the hall.')
        self.assertNotIn('reward', json.dumps(steps))

    def test_same_rule_retrieves_text_and_structured_actions_with_budget_and_restore(self):
        for action in ('go to desk', {'value': .7},
                       {'items': [{'coordinate': 42.3}]}, {'action': 'CALL'}):
            trace = json.dumps([dict(query='public task', action=action,
                                     public_feedback='Observed response')])
            self.memory.update('public task', trace, 1., True,
                               self.memory.retrieve('public task'), {})
        retrieved = self.memory.retrieve('public task')
        self.assertEqual(retrieved['projection'], 'grounded_evidence_v3')
        self.assertLessEqual(retrieved['tokens'], self.plan['memory_tokens'])
        self.assertTrue(retrieved['evidence'])
        self.assertIn('submitted action (unverified)', retrieved['context'])
        self.assertEqual(self.memory.episode_count, 4)
        path = Path(self.tmp.name) / 'snapshot.json'
        self.memory.snapshot(path)
        restored = GroundedEvidenceMemory(self.plan, self.client,
            Path(self.tmp.name) / 'restored',
            dict(threshold=.5, mean=.5, std=.1), Embeddings())
        restored.restore(path)
        self.assertEqual(restored.public_events, self.memory.public_events)
        self.assertEqual(restored.retrieve('public task')['context'], retrieved['context'])
        self.assertEqual(len(read(path)['grounded_evidence']['events']), 4)

    def test_review_requires_known_source_and_valid_schema(self):
        trace = json.dumps([dict(query='public task', action={'value': 2},
                                 public_feedback='Measured 3')])
        self.memory.update('public task', trace, 1., True,
                           self.memory.retrieve('public task'), {})
        self.memory.retrieve('public task')
        source = self.memory.selected_evidence[0]['source_sha256']
        output = Path(self.tmp.name) / 'actor'
        output.mkdir()
        system = SimpleNamespace(output=output, public_steps=[], turn=1, calls=0,
                                 input_tokens=0, output_tokens=0, max_input_tokens=0,
                                 last=None)

        def actor(query):
            action = NumberSchema.model_validate({'value': 2})
            system.last = (query.prompt, action.model_dump())
            return SimpleNamespace(action=action, metadata={})

        system.respond = actor
        self.memory.decorate_system(system)
        query = SimpleNamespace(prompt='current public task', instance_id='item-2',
                                response_schema=NumberSchema)
        self.client.review = dict(decision='REVISE', action={'value': 3},
                                  sources=['invented'])
        self.assertEqual(system.respond(query).action.model_dump(), {'value': 2})
        self.assertEqual(system.last[1], {'value': 2})
        self.assertNotIn('Inspect the current evidence before acting.',
                         self.client.messages[-1][1]['content'])
        self.assertIn('Unknown evidence source',
                      read(output / 'grounded_action_001.json')['reason'])

        second = SimpleNamespace(**vars(system))
        second.output = Path(self.tmp.name) / 'actor2'
        second.output.mkdir()
        second.respond = lambda q: SimpleNamespace(
            action=NumberSchema.model_validate({'value': 2}), metadata={})
        second.last = ('current public task', {'value': 2})
        self.memory.decorate_system(second)
        self.client.review = dict(decision='REVISE', action={'value': 3},
                                  sources=[source])
        self.assertEqual(second.respond(query).action.model_dump(), {'value': 3})
        self.assertEqual(second.last[1], {'value': 3})

        third = SimpleNamespace(**vars(system))
        third.output = Path(self.tmp.name) / 'actor3'
        third.output.mkdir()
        third.respond = lambda q: SimpleNamespace(
            action=NumberSchema.model_validate({'value': 2}), metadata={})
        third.last = ('current public task', {'value': 2})
        self.memory.decorate_system(third)
        self.client.complete = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError('review service unavailable'))
        self.assertEqual(third.respond(query).action.model_dump(), {'value': 2})
        self.assertIn('review service unavailable',
                      read(third.output / 'grounded_action_001.json')['reason'])


if __name__ == '__main__':
    unittest.main()
