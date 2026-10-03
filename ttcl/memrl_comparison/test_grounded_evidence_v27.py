"""The rejected text dropout does not leak into the typed policy."""
from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

from .grounded_evidence_v23 import TypedGroundedMemory as V23Memory
from .grounded_evidence_v26 import TypedGroundedMemory as V26Memory
from .grounded_evidence_v27 import TypedGroundedMemory


class TextRetryRestorationTest(TestCase):
    def test_text_retry_uses_native_all_attempts(self):
        memory = TypedGroundedMemory.__new__(TypedGroundedMemory)
        memory._text_retry = True
        with patch.object(V23Memory, 'retrieve', return_value={'policy': 'native'}) as retrieve:
            self.assertEqual(memory.retrieve('task'), {'policy': 'native'})
        retrieve.assert_called_once_with(memory, 'task')

    def test_structured_actions_keep_v26_path(self):
        memory = TypedGroundedMemory.__new__(TypedGroundedMemory)
        memory._text_retry = False
        with patch.object(V26Memory, 'retrieve', return_value={'policy': 'typed'}) as retrieve:
            self.assertEqual(memory.retrieve('task'), {'policy': 'typed'})
        retrieve.assert_called_once_with('task')
