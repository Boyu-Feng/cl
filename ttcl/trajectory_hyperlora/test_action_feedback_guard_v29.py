"""Progress-sensitive behavior checks for the generic action guard."""

import unittest

from ttcl.trajectory_hyperlora.action_feedback_guard_v29 import ActionFeedbackGuard


class ConservativeGuardTest(unittest.TestCase):
    def test_repeated_useful_transition_remains_available(self):
        guard = ActionFeedbackGuard(2, 2)
        for _ in range(3):
            guard.observe({"x": 1}, 4, {"x": 2})
        self.assertEqual(guard.allowed({"x": 1}, [4, 5]), [4, 5])

    def test_repeated_stall_is_blocked_even_with_structured_actions(self):
        guard = ActionFeedbackGuard(2, 2)
        for _ in range(2):
            guard.observe({"x": 1}, {"move": "north"}, {"x": 1})
        self.assertEqual(guard.allowed({"x": 1},
            [{"move": "north"}, {"move": "east"}]),
            [{"move": "east"}])

    def test_explicit_progress_feedback_overrides_changed_text(self):
        guard = ActionFeedbackGuard(2, 2)
        for i in range(2):
            guard.observe(f"message {i}", "retry", f"message {i+1}",
                          progressed=False)
        self.assertEqual(guard.allowed("new", ["retry", "other"]),
                         ["other"])

    def test_fallback_keeps_only_supplied_action(self):
        guard = ActionFeedbackGuard(1, 1)
        guard.observe("same", "only", "same")
        self.assertEqual(guard.allowed("same", ["only"]), ["only"])
        self.assertEqual(guard.summary()["fallbacks"], 1)


if __name__ == "__main__":
    unittest.main()
