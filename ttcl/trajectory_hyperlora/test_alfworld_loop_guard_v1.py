"""Behavioral checks for generic observation-action repetition control."""

import unittest

from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import guarded_commands


class LoopGuardTest(unittest.TestCase):
    def test_suppresses_only_repeated_choice_at_same_observation(self):
        seen = {("room A", "go to room B"): 2,
                ("room B", "go to room A"): 2}
        commands = ["go to room B", "take key"]
        self.assertEqual(guarded_commands("room A", commands, seen, 2),
                         ["take key"])
        self.assertEqual(guarded_commands("room C", commands, seen, 2),
                         commands)
        self.assertEqual(guarded_commands("room A", commands, seen, None),
                         commands)

    def test_preserves_environment_commands_when_all_were_repeated(self):
        seen = {("room A", "go to room B"): 2,
                ("room A", "take key"): 3}
        commands = ["go to room B", "take key"]
        self.assertEqual(guarded_commands("room A", commands, seen, 2),
                         commands)


if __name__ == "__main__":
    unittest.main()
