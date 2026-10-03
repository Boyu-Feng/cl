"""Checks for the structured-action memory withdrawal fallback."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

from .grounded_evidence_v21 import withdraw_unprojectable_memory


class WithdrawMemoryTest(unittest.TestCase):
    def test_unprojectable_first_action_removes_all_historical_text(self):
        system = SimpleNamespace(messages=[dict(
            content='Instructions\n\nPast experience:\nNative lesson\n\nPublic event')])
        result = withdraw_unprojectable_memory(system,
            {'typed_candidate_kinds': [], 'reason': 'No type-compatible historical candidate'})
        self.assertEqual(system.messages[0]['content'], 'Instructions')
        self.assertNotEqual(result['before_sha256'], result['after_sha256'])

    def test_projection_or_absent_context_keeps_instruction(self):
        original = 'Instructions\n\nPast experience:\nNative lesson'
        system = SimpleNamespace(messages=[dict(content=original)])
        self.assertIsNone(withdraw_unprojectable_memory(system,
                          {'typed_candidate_kinds': ['numeric_consensus']}))
        self.assertEqual(system.messages[0]['content'], original)
        system.messages[0]['content'] = 'Instructions'
        self.assertIsNone(withdraw_unprojectable_memory(system,
                          {'typed_candidate_kinds': []}))


if __name__ == '__main__':
    unittest.main()
