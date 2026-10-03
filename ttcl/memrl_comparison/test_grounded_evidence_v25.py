"""Structured actors use native text while the typed operator remains active."""
from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

from .grounded_evidence_v15 import TypedGroundedMemory as V15Memory
from .grounded_evidence_v24 import TypedGroundedMemory as V24Memory
from .grounded_evidence_v25 import TypedGroundedMemory
from .memory import Memory


class NativeStructuredContextTest(TestCase):
    def test_structured_retrieval_keeps_native_context_and_ids(self):
        memory = TypedGroundedMemory.__new__(TypedGroundedMemory)
        memory._text_retry = False
        memory.selected_public_context = 'stale public event'
        memory.selected_evidence = [{'stale': True}]
        native = dict(context='Task: current\nExperience: native',
                      ids=['memory-a'], tokens=7)
        with patch.object(Memory, 'retrieve', return_value=native) as retrieve:
            result = memory.retrieve('public task')
        retrieve.assert_called_once_with(memory, 'public task')
        self.assertEqual(result['context'], native['context'])
        self.assertEqual(result['ids'], native['ids'])
        self.assertEqual(memory.selected_public_context, '')
        self.assertEqual(memory.selected_evidence, [])

    def test_text_retries_keep_v24_schedule(self):
        memory = TypedGroundedMemory.__new__(TypedGroundedMemory)
        memory._text_retry = True
        with patch.object(V24Memory, 'retrieve', return_value={'retry_policy': 'v24'}) as retrieve:
            self.assertEqual(memory.retrieve('public task'), {'retry_policy': 'v24'})
        retrieve.assert_called_once_with('public task')

    def test_typed_operator_skips_v21_withdrawal(self):
        memory = TypedGroundedMemory.__new__(TypedGroundedMemory)
        system = object()
        with patch.object(V15Memory, 'decorate_system') as decorate:
            memory.decorate_system(system)
        decorate.assert_called_once_with(memory, system)
