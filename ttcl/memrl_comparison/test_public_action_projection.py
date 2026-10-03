from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from pydantic import BaseModel

from .public_action_projection import (PublicActionProjectionMemory,
                                       public_action_projection)
from .test_protocol import Client, Embeddings


class PublicActionProjectionTests(unittest.TestCase):
    def test_source_binding_and_unverified_action(self):
        trace = json.dumps([{'action': {'thinking': 'guess', 'transmitters': [
            {'center_freq': 42.0, 'currently_active': True}]},
            'public_feedback': 'Scan recorded.'}])
        field = public_action_projection(trace)
        self.assertEqual(field['public_trace_sha256'], hashlib.sha256(trace.encode()).hexdigest())
        self.assertEqual(field['event_count'], 1)
        self.assertNotIn('guess', json.dumps(field))
        self.assertIn('Scan recorded.', field['events'][0]['observation'])
        self.assertIn('currently_active', field['events'][0]['action'])
        changed = trace.replace('Scan recorded.', 'Scan rejected.')
        other = public_action_projection(changed)
        self.assertNotEqual(field['events'][0]['source_sha256'],
                            other['events'][0]['source_sha256'])

    def test_text_and_structured_actions_follow_same_path(self):
        for steps in ([{'action': 'go to desk', 'observation': 'You arrive.'}],
                      [{'action': {'sql': 'SELECT 1'},
                        'public_feedback': '1 row returned'}]):
            field = public_action_projection(json.dumps({'trajectory': steps}))
            self.assertEqual(field['event_count'], 1)
            self.assertEqual(field['events'][0]['step'], 1)

    def test_retrieval_shows_public_response_without_reward_or_answer_rewrite(self):
        with TemporaryDirectory() as tmp:
            plan = dict(upstream=str(Path(__file__).resolve().parents[2] / 'current_work/MemRL'),
                        writer_max_tokens=768, memory_tokens=2048,
                        rl_config=dict(epsilon=0., alpha=.3, gamma=0., topk=3))
            client = Client()
            memory = PublicActionProjectionMemory(
                plan, client, tmp, dict(threshold=.5, mean=.5, std=.1), Embeddings())
            trace = json.dumps([{'action': {'report': 'source active'},
                                 'public_feedback': 'Report recorded.'}])
            initial = memory.retrieve('public task')
            self.assertEqual(initial['context'], '')
            memory.update('public task', trace, 12345., True, initial, {})
            view = memory.retrieve('public task')
            self.assertEqual(view['projection'], 'public_action_projection_v4')
            self.assertIn('actor action (unverified)', view['context'])
            self.assertIn('Report recorded.', view['context'])
            self.assertNotIn('12345', view['context'])
            self.assertEqual(len(view['projected_events']), 1)

    def test_review_changes_action_only_with_valid_source_and_schema(self):
        class Action(BaseModel):
            value: int

        class Client:
            repeat = 303

            def __init__(self, raw):
                self.raw = raw
                self.calls = 0

            def complete(self, *args, **kwargs):
                self.calls += 1
                return dict(raw_response=self.raw, input_tokens=20,
                            output_tokens=10)

        class System:
            def __init__(self, output):
                self.output = output
                self.turn = self.calls = self.input_tokens = 0
                self.output_tokens = self.max_input_tokens = 0
                self.public_steps = []
                self.last = None

            def respond(self, query):
                self.turn += 1
                self.last = (query.prompt, {'value': 1})
                return SimpleNamespace(action=Action(value=1), metadata=None)

        source = 'a' * 64
        query = SimpleNamespace(prompt='current task', instance_id='item-1',
                                response_schema=Action)
        for cited, expected in ((source, 2), ('unknown', 1)):
            with self.subTest(cited=cited), TemporaryDirectory() as tmp:
                client = Client(json.dumps(dict(decision='REVISE',
                    action={'value': 2}, sources=[cited])))
                memory = object.__new__(PublicActionProjectionMemory)
                memory.client = client
                memory.project_actions = True
                memory.projection_context = 'Prior public observation.'
                memory.projection_evidence = [{'source_sha256': source}]
                system = System(Path(tmp))
                memory.decorate_system(system)
                result = system.respond(query)
                self.assertEqual(result.action.value, expected)
                self.assertEqual(system.last[1]['value'], expected)
                self.assertEqual(client.calls, 1)
                self.assertEqual(system.calls, 1)
                audit = json.loads((Path(tmp) / 'policy_action_001.json').read_text())
                self.assertEqual(audit['final']['value'], expected)
                second = system.respond(query)
                self.assertEqual(second.action.value, 1)
                self.assertEqual(client.calls, 1)
                next_audit = json.loads((Path(tmp) / 'policy_action_002.json').read_text())
                self.assertEqual(next_audit['reason'],
                                 'Review budget exhausted for this instance')

    def test_tool_call_is_never_rewritten(self):
        class Action(BaseModel):
            tool_call: dict

        class Client:
            repeat = 303
            calls = 0

            def complete(self, *args, **kwargs):
                self.calls += 1
                raise AssertionError('Tool action review must not run')

        with TemporaryDirectory() as tmp:
            memory = object.__new__(PublicActionProjectionMemory)
            memory.client = Client()
            memory.project_actions = True
            memory.projection_context = 'prior evidence'
            memory.projection_evidence = [{'source_sha256': 'a' * 64}]
            system = SimpleNamespace(output=Path(tmp), turn=1,
                respond=lambda query: SimpleNamespace(
                    action=Action(tool_call={'tool': 'query'}), metadata=None))
            memory.decorate_system(system)
            query = SimpleNamespace(prompt='current', response_schema=Action)
            self.assertEqual(system.respond(query).action.tool_call['tool'], 'query')
            self.assertEqual(memory.client.calls, 0)
            audit = json.loads((Path(tmp) / 'policy_action_001.json').read_text())
            self.assertEqual(audit['reason'], 'Tool action is not projected')

    def test_no_evidence_preserves_action_without_extra_call(self):
        memory = object.__new__(PublicActionProjectionMemory)
        memory.project_actions = True
        memory.projection_evidence = []
        memory.projection_context = ''
        with TemporaryDirectory() as tmp:
            class Action(BaseModel):
                value: int
            query = SimpleNamespace(prompt='current', response_schema=Action)
            system = SimpleNamespace(output=Path(tmp), turn=1,
                respond=lambda query: SimpleNamespace(action=Action(value=1),
                                                      metadata=None))
            memory.decorate_system(system)
            self.assertEqual(system.respond(query).action.value, 1)
            audit = json.loads((Path(tmp) / 'policy_action_001.json').read_text())
            self.assertEqual(audit['decision'], 'KEEP')


if __name__ == '__main__':
    unittest.main()
