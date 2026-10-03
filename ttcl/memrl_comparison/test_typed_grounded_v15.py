from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .grounded_evidence_v15 import (TypedGroundedMemory, _informative_feedback,
                                    _task_focus)
from .grounded_evidence_v13_base import public_events
from .memory import Memory
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


class TypedGroundedV15Tests(unittest.TestCase):
    def test_public_event_preserves_terminal_marker_in_source_binding(self):
        trace = json.dumps([dict(query='Find item', action={'action': 'inspect'},
                                public_feedback='name | value\nalpha | 23',
                                instance_complete=False),
                            dict(query='Find item', action={'action': 'submit'},
                                 public_feedback='Final feedback with details',
                                 instance_complete=True)])
        events = public_events(trace, 1)
        self.assertEqual([event['instance_complete'] for event in events],
                         [False, True])
        self.assertNotEqual(events[0]['source_sha256'], events[1]['source_sha256'])

    def test_feedback_filter_and_task_focus_are_generic(self):
        self.assertFalse(_informative_feedback(
            'Tool result (1/15 calls):\n\n(no results)'))
        self.assertTrue(_informative_feedback(
            'Tool result (1/15 calls):\n\nname | value\nalpha | 23\nbeta | 91'))
        self.assertFalse(_informative_feedback(
            'Tool result (1/15 calls):\n\nERROR: no such object: alpha'))
        self.assertEqual(_task_focus('Item 2/30\n\nFind a count.\n\n'
                                     'Repeated tool instructions'),
                         'Item 2/30\n\nFind a count.')

    def test_structured_retrieval_keeps_native_text_and_deduplicates_feedback(self):
        with TemporaryDirectory() as tmp:
            plan = dict(upstream=str(Path(__file__).resolve().parents[2] /
                                     'current_work/MemRL'), writer_max_tokens=768,
                        memory_tokens=2048, alf=dict(max_attempts=3),
                        rl_config=dict(epsilon=0., alpha=.3, gamma=0., topk=3))
            memory = TypedGroundedMemory(plan, ContractClient(), tmp,
                dict(threshold=.5, mean=.5, std=.1), Embeddings())
            memory.public_events = [dict(episode=i, step=1,
                source_sha256=f'source-{i}', public_trace_sha256=f'trace-{i}',
                public_input=f'Question {i}',
                submitted_action={'action': 'inspect'},
                observed_feedback='A stable public tool result')
                for i in (1, 2)]
            memory.public_events.append(dict(episode=3, step=1,
                source_sha256='source-empty', public_trace_sha256='trace-empty',
                public_input='current question', submitted_action={'action': 'inspect'},
                observed_feedback='Tool result:\n\n(no results)'))
            memory.public_events.append(dict(episode=4, step=2,
                source_sha256='source-terminal', public_trace_sha256='trace-terminal',
                public_input='current question', submitted_action={'action': 'submit'},
                observed_feedback='A stable public tool result',
                instance_complete=True))
            native = dict(query='current question', context='Exact native lesson',
                          ids=['memory-1'], tokens=3, context_sha256='original',
                          selected_before_budget=[], similarities=[])
            with patch.object(Memory, 'retrieve', return_value=native):
                retrieved = memory.retrieve('current question')
            self.assertEqual(retrieved['ids'], native['ids'])
            self.assertTrue(retrieved['context'].startswith(native['context']))
            self.assertEqual([e['source_sha256'] for e in retrieved['evidence']],
                             ['source-2'])
            self.assertIn('A stable public tool result', retrieved['context'])

    def test_structured_retrieval_can_use_two_distinct_events_from_one_episode(self):
        with TemporaryDirectory() as tmp:
            plan = dict(upstream=str(Path(__file__).resolve().parents[2] /
                                     'current_work/MemRL'), writer_max_tokens=768,
                        memory_tokens=2048, alf=dict(max_attempts=3),
                        rl_config=dict(epsilon=0., alpha=.3, gamma=0., topk=3))
            memory = TypedGroundedMemory(plan, ContractClient(), tmp,
                dict(threshold=.5, mean=.5, std=.1), Embeddings())
            memory.public_events = [dict(episode=1, step=i,
                source_sha256=f'source-{i}', public_trace_sha256='trace-1',
                public_input='Current public task',
                submitted_action={'action': 'inspect', 'item': i},
                observed_feedback=f'name | value\nitem{i} | {i * 20}\nmore | 42',
                instance_complete=False) for i in (1, 2)]
            native = dict(query='Current public task', context='', ids=[],
                          tokens=0, context_sha256='empty',
                          selected_before_budget=[], similarities=[])
            with patch.object(Memory, 'retrieve', return_value=native):
                retrieved = memory.retrieve('Current public task')
            self.assertEqual({e['source_sha256'] for e in retrieved['evidence']},
                             {'source-1', 'source-2'})

    def test_retry_schedule_preserves_two_native_attempts_then_drops(self):
        with TemporaryDirectory() as tmp:
            plan = dict(upstream=str(Path(__file__).resolve().parents[2] /
                                     'current_work/MemRL'), writer_max_tokens=768,
                        memory_tokens=2048, alf=dict(max_attempts=3),
                        rl_config=dict(epsilon=0., alpha=.3, gamma=0., topk=3))
            memory = TypedGroundedMemory(plan, ContractClient(), tmp,
                dict(threshold=.5, mean=.5, std=.1), Embeddings())
            native = dict(query='task', context='Exact native memory',
                          ids=['memory_000001'], tokens=3,
                          context_sha256='original', selected_before_budget=[],
                          similarities=[])
            memory.begin_task('retryable_text_command', 'input-hash')
            with patch.object(Memory, 'retrieve', return_value=native):
                first = memory.retrieve('task')
                second = memory.retrieve('task')
                third = memory.retrieve('task')
            self.assertEqual(first['context'], native['context'])
            self.assertEqual(first['ids'], native['ids'])
            self.assertEqual(second['context'], native['context'])
            self.assertEqual(second['ids'], native['ids'])
            self.assertEqual(third['context'], '')
            self.assertEqual(third['ids'], [])
            self.assertEqual(third['suppressed_ids'], native['ids'])

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
