from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from .grounded_evidence_v8 import TypedGroundedMemory, _candidate_summary
from .test_protocol import Embeddings
from .test_typed_grounded import SelectorClient, VectorSchema


class TypedGroundedV8Tests(unittest.TestCase):
    def test_tool_schema_removes_event_context_but_keeps_native_lesson(self):
        class ToolSchema(VectorSchema):
            @staticmethod
            def model_json_schema():
                return {'type': 'object', 'properties': {'tool_call': {'type': 'object'}}}

        with TemporaryDirectory() as tmp:
            plan = dict(upstream=str(Path(__file__).resolve().parents[2] /
                                     'current_work/MemRL'), writer_max_tokens=768,
                        memory_tokens=2048,
                        rl_config=dict(epsilon=0., alpha=.3, gamma=0., topk=3))
            memory = TypedGroundedMemory(plan, SelectorClient(), tmp,
                dict(threshold=.5, mean=.5, std=.1), Embeddings())
            memory.selected_public_context = 'Public event source=source1'
            output = Path(tmp) / 'actor'
            output.mkdir()
            system = SimpleNamespace(output=output, public_steps=[], turn=0,
                calls=0, input_tokens=0, output_tokens=0, max_input_tokens=0,
                last=None, messages=[{'role': 'system', 'content':
                    'Solve task\n\nPast experience:\nNative lesson\n\n'
                    'Public event source=source1'}])

            def actor(query):
                self.assertEqual(system.messages[0]['content'],
                                 'Solve task\n\nPast experience:\nNative lesson')
                value = {f'f{k}': .5 for k in range(8)}
                action = ToolSchema.model_validate(value)
                system.last = (query.prompt, value)
                return SimpleNamespace(action=action, metadata={})

            system.respond = actor
            memory.decorate_system(system)
            system.respond(SimpleNamespace(prompt='Use available tool',
                instance_id='tool-instance', response_schema=ToolSchema))
            self.assertTrue(memory._native_context_was_shown)
            self.assertEqual(json.loads((output/'typed_context_budget.json').read_text())
                             ['decision'], 'remove_unverified_event_context_for_tool_schema')

    def test_tool_schema_with_only_events_marks_native_context_unshown(self):
        class ToolSchema(VectorSchema):
            @staticmethod
            def model_json_schema():
                return {'properties': {'tool_call': {'type': 'object'}}}

        with TemporaryDirectory() as tmp:
            plan = dict(upstream=str(Path(__file__).resolve().parents[2] /
                                     'current_work/MemRL'), writer_max_tokens=768,
                        memory_tokens=2048,
                        rl_config=dict(epsilon=0., alpha=.3, gamma=0., topk=3))
            memory = TypedGroundedMemory(plan, SelectorClient(), tmp,
                dict(threshold=.5, mean=.5, std=.1), Embeddings())
            memory.selected_public_context = 'Public event source=source1'
            output = Path(tmp) / 'actor'
            output.mkdir()
            system = SimpleNamespace(output=output, public_steps=[], turn=0,
                calls=0, input_tokens=0, output_tokens=0, max_input_tokens=0,
                last=None, messages=[{'role': 'system', 'content':
                    'Solve task\n\nPast experience:\nPublic event source=source1'}])

            def actor(query):
                self.assertEqual(system.messages[0]['content'], 'Solve task')
                value = {f'f{k}': .5 for k in range(8)}
                action = ToolSchema.model_validate(value)
                system.last = (query.prompt, value)
                return SimpleNamespace(action=action, metadata={})

            system.respond = actor
            memory.decorate_system(system)
            system.respond(SimpleNamespace(prompt='Use available tool',
                instance_id='tool-instance', response_schema=ToolSchema))
            self.assertFalse(memory._native_context_was_shown)

    def test_recurring_summary_exposes_source_bound_record_values(self):
        record = dict(identity_value=42., seen_in_episodes=3,
                      sample_record=dict(coordinate=42.2, active=False),
                      source_sha256=['first', 'second', 'third'])
        summary = _candidate_summary(dict(kind='recurring_records',
            path='records', identity_field='coordinate', records=[record]), 1)
        self.assertEqual(summary['selection_label'], 'APPLY_1')
        self.assertEqual(summary['recurrent_records'][0]['sample_record'],
                         record['sample_record'])
        self.assertEqual(summary['recurrent_records'][0]['source_sha256'],
                         ['second', 'third'])

    def test_selector_sees_initial_and_prior_public_sample_context(self):
        with TemporaryDirectory() as tmp:
            plan = dict(upstream=str(Path(__file__).resolve().parents[2] /
                                     'current_work/MemRL'), writer_max_tokens=768,
                        memory_tokens=2048,
                        rl_config=dict(epsilon=0., alpha=.3, gamma=0., topk=3))
            client = SelectorClient()
            client.selection = 'APPLY_1'
            memory = TypedGroundedMemory(plan, client, tmp,
                dict(threshold=.5, mean=.5, std=.1), Embeddings())
            for episode in (1, 2):
                trace = json.dumps([dict(query=f'Prior public sample {episode}',
                    action={f'f{i}': episode / 10 for i in range(8)},
                    public_feedback='Submission recorded')])
                memory.update('same target', trace, 1., True,
                              memory.retrieve('same target'), {})
            memory.retrieve('same target')
            output = Path(tmp) / 'actor'
            output.mkdir()
            system = SimpleNamespace(output=output,
                public_steps=[dict(query='Current initial sample description',
                                   action={'tool_call': {'tool': 'inspect'}},
                                   public_feedback='Measured values')],
                turn=3, calls=0, input_tokens=0, output_tokens=0,
                max_input_tokens=0, last=None,
                messages=[{'role': 'system', 'content': 'Solve task'}])

            def actor(query):
                values = {f'f{i}': .9 for i in range(8)}
                action = VectorSchema.model_validate(values)
                system.last = (query.prompt, values)
                return SimpleNamespace(action=action, metadata={})

            system.respond = actor
            memory.decorate_system(system)
            response = system.respond(SimpleNamespace(prompt='Submit all fields',
                instance_id='current', response_schema=VectorSchema))
            selector = next(messages for messages in reversed(client.messages)
                            if 'Select a general state operator' in messages[0]['content'])
            payload = json.loads(selector[1]['content'])
            self.assertIn('Current initial sample', payload['episode_initial_task'])
            self.assertEqual(len(payload['prior_episode_starts']), 2)
            self.assertEqual(payload['allowed_operations'], ['KEEP', 'APPLY_1'])
            self.assertEqual(payload['candidate_summaries'][0]['selection_label'],
                             'APPLY_1')
            self.assertAlmostEqual(response.action.model_dump()['f0'], .4)
            self.assertIn('field completeness is irrelevant',
                          selector[0]['content'])


if __name__ == '__main__':
    unittest.main()
