"""Behavioral tests for action control without domain command syntax."""

import unittest

from ttcl.trajectory_hyperlora.action_feedback_guard_v28 import (
    ActionFeedbackGuard, stable_key,
)


class ActionFeedbackGuardTest(unittest.TestCase):
    def test_revisited_state_action_is_suppressed(self):
        guard = ActionFeedbackGuard(repeat_limit=2)
        for _ in range(2):
            guard.observe({"screen": "A"}, {"button": 1},
                          {"screen": "B"})
        self.assertEqual(guard.allowed({"screen": "A"},
            [{"button": 1}, {"button": 2}]), [{"button": 2}])
        self.assertEqual(guard.allowed({"screen": "C"},
            [{"button": 1}, {"button": 2}]),
            [{"button": 1}, {"button": 2}])

    def test_repeated_action_stagnation_and_progress(self):
        guard = ActionFeedbackGuard(repeat_limit=9, stagnation_limit=2)
        guard.observe("same", "retry", "same")
        guard.observe("same", "retry", "same")
        self.assertEqual(guard.allowed("other", ["retry", "inspect"]),
                         ["inspect"])
        guard.observe("other", "retry", "changed")
        self.assertEqual(guard.allowed("changed", ["retry", "inspect"]),
                         ["retry", "inspect"])

    def test_all_suppressed_falls_back_to_supplied_choices(self):
        guard = ActionFeedbackGuard(repeat_limit=1)
        guard.observe("s", 0, "s")
        self.assertEqual(guard.allowed("s", [0]), [0])
        self.assertEqual(guard.summary()["fallbacks"], 1)
        self.assertEqual(guard.allowed("s", []), [])

    def test_public_progress_can_override_observation_change(self):
        guard = ActionFeedbackGuard(repeat_limit=9, stagnation_limit=2)
        guard.observe("prompt 1", "repeat", "prompt 2", progressed=False)
        guard.observe("prompt 2", "repeat", "prompt 3", progressed=False)
        self.assertEqual(guard.allowed("prompt 3", ["repeat", "different"]),
                         ["different"])
        self.assertEqual(stable_key({"b": 2, "a": 1}),
                         stable_key({"a": 1, "b": 2}))


if __name__ == "__main__":
    unittest.main()
