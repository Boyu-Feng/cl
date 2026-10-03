"""Checks the source-bound escalating-action loop pattern."""
from __future__ import annotations

import unittest

from .grounded_evidence_v22 import escalating_loop


class EscalatingLoopTest(unittest.TestCase):
    def test_three_same_actions_with_rising_number_trigger(self):
        steps = [dict(step=1, action={'action':'RAISE','amount':30}),
                 dict(step=2, action={'action':'RAISE','amount':50})]
        pattern = escalating_loop(steps, {'action':'RAISE','amount':70})
        self.assertEqual(pattern['kind'], 'RAISE')
        self.assertEqual(pattern['numeric_values'], [30,50,70])

    def test_requires_repetition_and_strict_escalation(self):
        steps = [dict(step=1, action={'action':'CALL','amount':0}),
                 dict(step=2, action={'action':'RAISE','amount':50})]
        self.assertIsNone(escalating_loop(steps, {'action':'RAISE','amount':70}))
        steps[0]['action'] = {'action':'RAISE','amount':30}
        self.assertIsNone(escalating_loop(steps, {'action':'RAISE','amount':50}))
        self.assertIsNone(escalating_loop(steps, {'tool_call':'SQL','limit':70}))


if __name__ == '__main__':
    unittest.main()
