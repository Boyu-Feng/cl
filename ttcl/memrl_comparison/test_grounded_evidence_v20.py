"""Checks for the bounded structured-action evidence withdrawal rule."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

from .grounded_evidence_v20 import withdraw_unprojectable_evidence


class WithdrawEventTextTest(unittest.TestCase):
    def test_unprojectable_first_action_keeps_native_context(self):
        system = SimpleNamespace(messages=[dict(
            content='System instructions\n\nPast experience:\nNative lesson\n\nPublic event')])
        result = withdraw_unprojectable_evidence(
            system, 'Public event',
            {'typed_candidate_kinds': [], 'reason': 'No type-compatible historical candidate'})
        self.assertEqual(system.messages[0]['content'],
                         'System instructions\n\nPast experience:\nNative lesson')
        self.assertTrue(result['native_context_retained'])
        self.assertNotEqual(result['before_sha256'], result['after_sha256'])

    def test_compatible_projection_preserves_event_text(self):
        prompt = 'System instructions\n\nPast experience:\nPublic event'
        system = SimpleNamespace(messages=[dict(content=prompt)])
        result = withdraw_unprojectable_evidence(
            system, 'Public event', {'typed_candidate_kinds': ['recurring_records']})
        self.assertIsNone(result)
        self.assertEqual(system.messages[0]['content'], prompt)

    def test_absent_or_nonmatching_event_is_not_removed(self):
        prompt = 'System instructions\n\nPast experience:\nNative lesson'
        system = SimpleNamespace(messages=[dict(content=prompt)])
        self.assertIsNone(withdraw_unprojectable_evidence(
            system, 'Different event', {'typed_candidate_kinds': []}))
        self.assertEqual(system.messages[0]['content'], prompt)


if __name__ == '__main__':
    unittest.main()
