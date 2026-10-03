"""Focused checks of ordered Docker-task memory boundaries without Docker."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import docker_v21_protocol as protocol


class FakeMemory:
    def __init__(self):
        self.calls = self.input_tokens = self.output_tokens = self.limit_hits = 0
        self.events = []

    def begin_task(self, interface, binding):
        self.events.append(('begin', interface, binding))

    def retrieve(self, query):
        self.events.append(('retrieve', query))
        return dict(query=query, context='old lesson', ids=[])

    def decorate_system(self, system):
        self.events.append(('decorate', system.index))

    def update(self, query, trace, reward, success, retrieval, binding):
        self.events.append(('update', binding['canonical_index'], reward))
        assert retrieval['query'] == query
        assert json.loads(trace)[0]['action'] == {'command': 'look'}
        self.calls += 1
        return dict(new_memory_id=f'memory_{self.calls:06d}')

    def snapshot(self, path):
        path.write_text(json.dumps({'calls': self.calls}))


class FakeInner:
    def __init__(self, plan, client, arm, history, native_memory,
                 output, brief, index, count):
        self.index, self.output = index, Path(output)
        self.calls = self.input_tokens = self.output_tokens = 0
        self.max_input_tokens = 0
        self.public_steps = []
        self.messages = [{'role': 'system', 'content': 'instruction'}]

    def respond(self, query):
        self.calls += 1
        self.input_tokens += 3
        self.output_tokens += 1
        self.max_input_tokens = 3
        return SimpleNamespace(action={'command': 'look'})

    def observe(self, observation, next_query=None):
        self.public_steps.append(dict(query='question', action={'command': 'look'},
                                      public_feedback=observation.content))


@dataclass
class FakeOutcome:
    instance_index: int
    instance_id: str
    reward: float
    success: bool | None = None


class DockerProtocolTest(unittest.TestCase):
    def test_reward_update_precedes_next_instance_retrieval(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
                protocol, 'System', FakeInner):
            memory = FakeMemory()
            client = SimpleNamespace(repeat=303)
            system = protocol.OnlineDockerSystem(
                {'tasks': {'sales_prediction': 2}}, client, memory, 'typed_grounded',
                'sales_prediction', '', Path(directory) / 'arm', 2)
            recorder = protocol.OnlineDockerRecorder(
                Path(directory) / 'arm', system, 2)
            for index in range(2):
                query = SimpleNamespace(prompt=f'public question {index}',
                                        instance_index=index)
                system.respond(query)
                system.observe(SimpleNamespace(content='public observation',
                    metadata=None, instance_complete=True))
                recorder.sync_instance_outcomes([FakeOutcome(
                    index, f'sales:{index}', .5 + index)])
            self.assertEqual([event[0] for event in memory.events],
                             ['begin', 'retrieve', 'decorate', 'update'] * 2)
            self.assertEqual(system.calls, 2)
            self.assertEqual(system.finalized, {0, 1})
            for index in range(2):
                row = json.loads((Path(directory) / 'arm' /
                    f'episode_{index + 1:03d}' / 'row.json').read_text())
                self.assertEqual(row['status'], 'complete')
                self.assertEqual(row['actor_calls'], 1)
                self.assertEqual(row['writer_calls'], 1)


if __name__ == '__main__':
    unittest.main()
