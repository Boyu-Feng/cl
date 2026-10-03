"""Retry diversity preserves the three-attempt budget and memory credit."""
from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

from .grounded_evidence_v24 import TypedGroundedMemory
from .memory import Memory


class RetryDiversityTest(TestCase):
    def test_only_second_text_attempt_suppresses_memory_and_credit(self):
        memory = TypedGroundedMemory.__new__(TypedGroundedMemory)
        memory._text_retry = True
        memory._text_attempt = 0
        native = dict(context='Historical text', ids=['memory-a'], tokens=3)
        with patch.object(Memory, 'retrieve', return_value=native) as retrieve:
            first = memory.retrieve('public task')
            second = memory.retrieve('public task')
            third = memory.retrieve('public task')
        self.assertEqual(retrieve.call_count, 3)
        self.assertEqual(first['context'], native['context'])
        self.assertEqual(first['ids'], native['ids'])
        self.assertEqual(second['context'], '')
        self.assertEqual(second['ids'], [])
        self.assertEqual(second['suppressed_ids'], native['ids'])
        self.assertEqual(second['tokens'], 0)
        self.assertEqual(third['context'], native['context'])
        self.assertEqual(third['ids'], native['ids'])
        self.assertEqual(memory._text_attempt, 3)

    def test_structured_branch_keeps_existing_projection(self):
        memory = TypedGroundedMemory.__new__(TypedGroundedMemory)
        memory._text_retry = False
        memory._text_attempt = 1
        from .grounded_evidence_v23 import TypedGroundedMemory as V23Memory
        with patch.object(V23Memory, 'retrieve', return_value={'projection': 'v23'}) as retrieve:
            self.assertEqual(memory.retrieve('public task'), {'projection': 'v23'})
        retrieve.assert_called_once_with('public task')
