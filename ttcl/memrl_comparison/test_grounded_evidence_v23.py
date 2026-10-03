"""The text branch must preserve native retrieval at every retry."""
from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

from .grounded_evidence_v21 import TypedGroundedMemory as V21Memory
from .grounded_evidence_v23 import TypedGroundedMemory
from .memory import Memory


class NativeTextRetryTest(TestCase):
    def test_third_attempt_preserves_native_context_and_credit_ids(self):
        memory = TypedGroundedMemory.__new__(TypedGroundedMemory)
        memory._text_retry = True
        memory._text_attempt = 2
        native = dict(context='Bound historical text', ids=['memory-a'], tokens=4)
        with patch.object(Memory, 'retrieve', return_value=native) as retrieve:
            result = memory.retrieve('current public task')
        retrieve.assert_called_once_with(memory, 'current public task')
        self.assertEqual(result['context'], native['context'])
        self.assertEqual(result['ids'], native['ids'])
        self.assertEqual(result['retry_policy'], 'native_exact_all_attempts')
        self.assertEqual(memory._text_attempt, 3)
        self.assertEqual(memory.selected_context, native['context'])
        self.assertEqual(memory.selected_evidence, [])

    def test_structured_branch_delegates_to_existing_projection(self):
        memory = TypedGroundedMemory.__new__(TypedGroundedMemory)
        memory._text_retry = False
        with patch.object(V21Memory, 'retrieve', return_value={'projection': 'v21'}) as retrieve:
            self.assertEqual(memory.retrieve('current public task'),
                             {'projection': 'v21'})
        retrieve.assert_called_once_with('current public task')
