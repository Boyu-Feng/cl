"""No-training v36 intervention and accounting checks."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from .grounded_evidence_v27 import TypedGroundedMemory as V27Memory
from .grounded_evidence_v36 import TypedGroundedMemory
from .immediate_repair_actor import ImmediateRepairActor
from .test_guided_choice_actor import Session, Tokenizer


class Client:
    def __init__(self, native_text):
        self.native_text = native_text
        self.calls = []
        self.session = Session()
        self.tokenizer = Tokenizer()
        self.plan = {'context': 1024}

    def complete(self, messages, random_seed, **kwargs):
        self.calls.append((messages, random_seed, kwargs))
        return dict(raw_response=self.native_text, finish_reason='stop',
                    input_tokens=5, output_tokens=2,
                    rendered_prompt_sha256='native-prompt', seconds=0.)


class ImmediateRepairTests(TestCase):
    def actor(self, raw):
        client = Client(raw)
        actor = ImmediateRepairActor(dict(actor_url='http://actor',
                                          actor_max_tokens=32,
                                          actor_temperature=.7), client)
        return actor, client

    def test_valid_native_command_is_not_replaced_or_recalled(self):
        actor, client = self.actor('look')
        try:
            result = actor.generate([
                dict(role='system', content='Do task'),
                dict(role='user', content='Room\nAvailable commands:\nlook\ninventory')],
                123)
        finally:
            actor.pool.shutdown(wait=True)
        self.assertEqual(result['text'], 'look')
        self.assertEqual(result['completion_count'], 1)
        self.assertFalse(result['repair_used'])
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(client.session.calls), 0)

    def test_invalid_native_command_is_repaired_before_environment_step(self):
        actor, client = self.actor('success')
        try:
            result = actor.generate([
                dict(role='system', content='Do task'),
                dict(role='user', content='Room\nAvailable commands:\nlook\ninventory')],
                123)
        finally:
            actor.pool.shutdown(wait=True)
        self.assertEqual(result['text'], 'look')
        self.assertEqual(result['native_command'], 'success')
        self.assertTrue(result['repair_used'])
        self.assertEqual(result['completion_count'], 2)
        self.assertEqual(result['usage']['prompt_tokens'], 5 +
                         result['repair_usage']['prompt_tokens'])
        self.assertEqual(result['usage']['completion_tokens'], 3)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.session.calls[0]['guided_choice'],
                         ['look', 'inventory'])

    def test_episode_executes_only_repaired_command_and_counts_two_calls(self):
        actor, client = self.actor('success')
        actor.plan.update(data_root='/unused', max_steps=2)
        class Environment:
            def __init__(self):
                self.commands = []

            def reset(self):
                return dict(feedback='Room', admissible_commands=['look', 'inventory'])

            def step(self, command):
                self.commands.append(command)
                return dict(feedback='Done', admissible_commands=['look'], won=True), 1., True

            def close(self):
                pass

        environment = Environment()
        try:
            with TemporaryDirectory() as directory:
                output = Path(directory) / 'episode'
                output.mkdir()
                with patch('ttcl.memrl_comparison.immediate_repair_actor.'
                           'alf_environment.make_env',
                           return_value=environment):
                    episode = actor.run_many([dict(game='game.tw-pddl', memory='',
                                                   seed=123, output=output)])[0]
        finally:
            actor.pool.shutdown(wait=True)
        self.assertEqual(environment.commands, ['look'])
        self.assertEqual(episode['reward'], 1.)
        self.assertEqual(episode['actor_completion_count'], 2)
        self.assertEqual(episode['repair_count'], 1)
        self.assertEqual(len(episode['trajectory']), 1)

    def test_no_unbudgeted_repair_call_or_invalid_environment_step(self):
        actor, client = self.actor('success')
        actor.plan.update(data_root='/unused', max_steps=1)
        class Environment:
            def __init__(self):
                self.commands = []

            def reset(self):
                return dict(feedback='Room', admissible_commands=['look'])

            def step(self, command):
                self.commands.append(command)
                raise AssertionError('Invalid command reached environment')

            def close(self):
                pass

        environment = Environment()
        try:
            with TemporaryDirectory() as directory:
                output = Path(directory) / 'episode'
                with patch('ttcl.memrl_comparison.immediate_repair_actor.'
                           'alf_environment.make_env',
                           return_value=environment):
                    episode = actor.run_many([dict(game='game.tw-pddl', memory='',
                                                   seed=123, output=output)])[0]
        finally:
            actor.pool.shutdown(wait=True)
        self.assertEqual(environment.commands, [])
        self.assertEqual(episode['actor_completion_count'], 1)
        self.assertEqual(episode['steps'], 0)
        self.assertEqual(episode['termination'],
                         'actor_call_budget_invalid_command')
        self.assertEqual(len(client.session.calls), 0)


class DeferredCreditTests(TestCase):
    def test_multiple_text_memories_keep_provenance_without_q_broadcast(self):
        memory = TypedGroundedMemory.__new__(TypedGroundedMemory)
        memory._text_retry = True
        q = {'a': -.2, 'b': .4}
        memory.store = SimpleNamespace(get=lambda mid: SimpleNamespace(
            metadata=SimpleNamespace(q_value=q[mid])))
        def native_update(_, query, trace, reward, success, retrieval, binding):
            updates = {mid: memory.service.update_value(mid, reward)
                       for mid in retrieval['ids']}
            return dict(q_updates=updates, writer_seen_ids=list(retrieval['ids']))
        def update_value(mid, reward):
            q[mid] += reward
            return q[mid]
        memory.service = SimpleNamespace(update_value=update_value)
        retrieval = dict(ids=['a', 'b'])
        with patch.object(V27Memory, 'update', native_update):
            result = memory.update('q', '[]', -1., False, retrieval, {})
        self.assertEqual(q, {'a': -.2, 'b': .4})
        self.assertEqual(result['deferred_q_ids'], ['a', 'b'])
        self.assertEqual(result['q_updates'], {})
        self.assertEqual(result['writer_seen_ids'], ['a', 'b'])
        self.assertIs(memory.service.update_value, update_value)

    def test_singleton_and_structured_actions_keep_native_q_update(self):
        memory = TypedGroundedMemory.__new__(TypedGroundedMemory)
        memory._text_retry = True
        with patch.object(V27Memory, 'update', return_value={'q_updates': {'a': 1.}}) as update:
            result = memory.update('q', '[]', 1., True, {'ids': ['a']}, {})
            self.assertEqual(result['deferred_q_ids'], [])
            memory._text_retry = False
            result = memory.update('q', '[]', 1., True, {'ids': ['a', 'b']}, {})
            self.assertEqual(result['deferred_q_ids'], [])
        self.assertEqual(update.call_count, 2)
