"""Verify that compact actor prompts preserve the initial task and recent turns."""

import unittest
from pathlib import Path
from unittest.mock import patch

from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import run_episode


class _Env:
    def __init__(self):
        self.turn = 0

    def reset(self):
        return self._state()

    def _state(self):
        return {"feedback": f"observation {self.turn}",
                "admissible_commands": ["look"],
                "won": self.turn == 4}

    def step(self, command):
        assert command == "look"
        self.turn += 1
        return self._state(), float(self.turn == 4), self.turn == 4

    def close(self):
        pass


class _Agent:
    task_conditioned = False
    task_context_scope = "initial"

    def set_source(self, source, target_fields=None):
        pass


class ActorHistoryWindowTest(unittest.TestCase):
    def _run(self, history_turns):
        prompts = []

        def generate(*args):
            prompts.append([dict(message) for message in args[2]])
            return "look"

        with patch("ttcl.trajectory_hyperlora.alfworld_zero_shot_probe.make_env",
                   return_value=_Env()), patch(
                   "ttcl.trajectory_hyperlora.alfworld_zero_shot_probe.generate",
                   side_effect=generate):
            episode = run_episode(_Agent(), None, Path("unused"), {},
                adapter=False, device="cpu", max_steps=4,
                max_new_tokens=8, actor_history_turns=history_turns)
        self.assertEqual(episode["status"], "complete")
        self.assertEqual(episode["reward"], 1.0)
        return prompts

    def test_training_window_keeps_initial_task_and_recent_feedback(self):
        compact = self._run(2)
        full = self._run(None)
        self.assertEqual([len(p) for p in compact], [2, 4, 6, 7])
        self.assertEqual([len(p) for p in full], [2, 4, 6, 8])
        self.assertEqual(compact[-1][1]["content"], full[-1][1]["content"])
        self.assertEqual(compact[-1][-1]["content"], full[-1][-1]["content"])
        self.assertNotIn("observation 0", " ".join(
            message["content"] for message in compact[-1][2:]))


if __name__ == "__main__":
    unittest.main()
