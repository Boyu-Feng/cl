from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from .grounded_evidence_v9 import TypedGroundedMemory
from .test_protocol import Embeddings
from .test_typed_grounded import SelectorClient, VectorSchema


class ContractClient(SelectorClient):
    persistent_set = False
    shared_numeric_target = True

    def complete(self, messages, *args, **kwargs):
        if 'Classify the public task objective' in messages[0]['content']:
            self.messages.append(messages)
            return dict(raw_response=json.dumps(dict(persistent_set=self.persistent_set,
                shared_numeric_target=self.shared_numeric_target,
                reason='Public task contract')),
                input_tokens=40, output_tokens=20, finish_reason='stop')
        return super().complete(messages, *args, **kwargs)


class TypedGroundedV9Tests(unittest.TestCase):
    def test_persistent_contract_applies_recurring_records(self):
        class RecordSchema:
            @staticmethod
            def model_json_schema():
                return {'properties': {'records': {'type': 'array'}}}

            @staticmethod
            def model_validate(value):
                if not isinstance(value.get('records'), list):
                    raise ValueError('Expected records')
                return SimpleNamespace(model_dump=lambda: value)

        with TemporaryDirectory() as tmp:
            plan = dict(upstream=str(Path(__file__).resolve().parents[2] /
                                     'current_work/MemRL'), writer_max_tokens=768,
                        memory_tokens=2048,
                        rl_config=dict(epsilon=0., alpha=.3, gamma=0., topk=3))
            client = ContractClient()
            client.persistent_set = True
            client.shared_numeric_target = False
            memory = TypedGroundedMemory(plan, client, tmp,
                dict(threshold=.5, mean=.5, std=.1), Embeddings())
            memory.episode_count = 4
            memory.public_events = [dict(episode=episode,
                source_sha256=f'source-{episode}', submitted_action={'records': [
                    {'coordinate': 10 + .2 * episode, 'width': 3 + episode},
                    {'coordinate': 80 + .1 * episode, 'width': 8 + episode}]})
                for episode in range(1, 5)]
            output = Path(tmp) / 'actor'
            output.mkdir()
            system = SimpleNamespace(output=output,
                public_steps=[dict(query='=== Brief ===\nKeep a persistent set'
                                   '\n=======================\nCurrent sample')],
                turn=1, calls=0, input_tokens=0, output_tokens=0,
                max_input_tokens=0, last=None,
                messages=[{'role': 'system', 'content': 'Solve task'}])

            def actor(query):
                value = {'records': [{'coordinate': 10.8, 'width': 7}]}
                action = RecordSchema.model_validate(value)
                system.last = (query.prompt, value)
                return SimpleNamespace(action=action, metadata={})

            system.respond = actor
            memory.decorate_system(system)
            response = system.respond(SimpleNamespace(prompt='Submit records',
                instance_id='current', response_schema=RecordSchema))
            self.assertEqual(len(response.action.model_dump()['records']), 2)
            self.assertEqual(json.loads((output/'typed_action_001.json').read_text())
                             ['operation'], 'RECURRING_RECORDS')
            self.assertEqual(system.calls, 1)

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

    def test_contract_uses_public_brief_and_survives_snapshot(self):
        with TemporaryDirectory() as tmp:
            plan = dict(upstream=str(Path(__file__).resolve().parents[2] /
                                     'current_work/MemRL'), writer_max_tokens=768,
                        memory_tokens=2048,
                        rl_config=dict(epsilon=0., alpha=.3, gamma=0., topk=3))
            client = ContractClient()
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
                public_steps=[dict(query='=== Brief ===\nShared target fields'
                                   '\n=======================\nCurrent sample data',
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
            classifier = next(messages for messages in reversed(client.messages)
                              if 'Classify the public task objective' in messages[0]['content'])
            self.assertIn('Shared target fields', classifier[1]['content'])
            self.assertNotIn('Current sample data', classifier[1]['content'])
            self.assertAlmostEqual(response.action.model_dump()['f0'], .4)
            self.assertEqual(memory.target_contract['shared_numeric_target'], True)
            snapshot = Path(tmp) / 'snapshot.json'
            memory.snapshot(snapshot)
            memory.target_contract = None
            memory.restore(snapshot)
            self.assertEqual(memory.target_contract['shared_numeric_target'], True)


if __name__ == '__main__':
    unittest.main()
