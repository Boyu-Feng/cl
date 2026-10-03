"""Failure-only writer intervention preserves source and reset boundaries."""
from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

from .grounded_evidence_v27 import TypedGroundedMemory as V27Memory
from .grounded_evidence_v28 import TypedGroundedMemory, failure_reflection_messages


class FailureReflectionTest(TestCase):
    def test_prompt_uses_public_task_and_trace_without_reward(self):
        message = failure_reflection_messages('find the item', 'look: shelf empty')[0]
        self.assertEqual(message['role'], 'user')
        self.assertIn('find the item', message['content'])
        self.assertIn('look: shelf empty', message['content'])
        self.assertIn('Separate observations from guesses', message['content'])
        with self.assertRaises(ValueError):
            failure_reflection_messages('find the item', '')

    def test_failure_write_scope_clears_even_when_update_fails(self):
        memory = TypedGroundedMemory.__new__(TypedGroundedMemory)
        memory._failure_write = False
        memory._failure_task = ''
        memory._failure_trace = ''

        def check_scope(instance, query, public_trace, reward, success,
                        retrieval, binding):
            self.assertTrue(instance._failure_write)
            self.assertEqual(instance._failure_task, 'task')
            self.assertEqual(instance._failure_trace, 'public trace')
            raise RuntimeError('writer failure')

        with patch.object(V27Memory, 'update', check_scope):
            with self.assertRaisesRegex(RuntimeError, 'writer failure'):
                memory.update('task', 'public trace', -1, False, {}, {})
        self.assertFalse(memory._failure_write)
        self.assertEqual(memory._failure_task, '')
        self.assertEqual(memory._failure_trace, '')
