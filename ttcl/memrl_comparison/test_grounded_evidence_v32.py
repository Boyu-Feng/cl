"""Single-source readout abstention without removing native MemRL context."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from .grounded_evidence_v27 import TypedGroundedMemory as V27Memory
from .grounded_evidence_v30 import TypedGroundedMemory as V30Memory
from .grounded_evidence_v32 import TypedGroundedMemory


class FeedbackDiversityTests(unittest.TestCase):
    def test_one_group_reverts_to_native_context(self):
        memory = object.__new__(TypedGroundedMemory)
        memory._text_retry = False
        proposed = dict(context='native plus one readout', ids=['m1'],
                        recurrent_feedback=[{'source_sha256': ['one', 'two']}])
        native = dict(context='native', ids=['m1'])
        with (patch.object(V30Memory, 'retrieve', return_value=proposed),
              patch.object(V27Memory, 'retrieve', return_value=native)):
            result = memory.retrieve('current task')
        self.assertEqual(result['context'], 'native')
        self.assertEqual(result['ids'], ['m1'])
        self.assertEqual(len(result['rejected_recurrent_feedback']), 1)

    def test_two_groups_are_kept_without_extra_retrieval(self):
        memory = object.__new__(TypedGroundedMemory)
        memory._text_retry = False
        proposed = dict(context='native plus two readouts', ids=['m1'],
                        recurrent_feedback=[{'a': 1}, {'b': 2}])
        with (patch.object(V30Memory, 'retrieve', return_value=proposed),
              patch.object(V27Memory, 'retrieve') as native):
            self.assertIs(memory.retrieve('current task'), proposed)
        native.assert_not_called()


if __name__ == '__main__':
    unittest.main()
