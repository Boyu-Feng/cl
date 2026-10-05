"""Guard the empty-start online protocol and trajectory feedback binding."""

import unittest

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    records_from_episode, select_games,
)


class OnlineFromEmptyTest(unittest.TestCase):
    def test_select_games_never_reuses_evaluation_target(self):
        plan = {"training": [{"games": ["a/train/one", "a/train/two"]}],
                "evaluation": [{"games": ["a/train/two"]}]}
        with self.assertRaisesRegex(ValueError, "non-training"):
            select_games(plan, 0, 1)

    def test_own_steps_keep_feedback_and_terminal_outcome(self):
        episode = {"status": "complete", "reward": 0.,
                   "initial_observation": "room one",
                   "trajectory": [
                       {"command": "look", "valid": True,
                        "observation": "room two"},
                       {"command": "take object", "valid": False,
                        "observation": "nothing happens"}]}
        rows = records_from_episode(episode)
        self.assertEqual(rows[0]["observation"], "room one")
        self.assertEqual(rows[1]["observation"], "room two")
        self.assertTrue(rows[0]["feedback"].startswith("valid command."))
        self.assertTrue(rows[1]["feedback"].startswith("invalid command."))
        self.assertNotIn("task completed", rows[1]["feedback"])


if __name__ == "__main__":
    unittest.main()
