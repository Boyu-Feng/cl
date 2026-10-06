"""Check that only paired official outcomes count as source advantage."""

import json
from pathlib import Path
import tempfile
import unittest

from ttcl.trajectory_hyperlora.analyze_alf_sibling_reward import analyze


def episode(won: bool, steps: int, observation: str = "same") -> dict:
    return {"status": "complete", "reward": float(won),
            "steps": steps, "initial_observation": observation,
            "invalid_commands": 0,
            "trajectory": [{"won": bool(won and index == steps - 1),
                            "command": "look"}
                           for index in range(steps)]}


class SourceRewardAuditTest(unittest.TestCase):
    def test_counts_terminal_and_efficiency_advantage_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollouts.json"
            path.write_text(json.dumps({"protocol": "official won",
                "checkpoint_sha256": "checkpoint",
                "candidates_sha256": "candidates",
                "source_review_sha256": "review", "failures": [],
                "summary": {"n": 2}, "games": [
                    {"game": "one", "family": "a", "arms": {
                        "base": episode(False, 30),
                        "own": episode(True, 5),
                        "wrong": episode(False, 30)}},
                    {"game": "two", "family": "b", "arms": {
                        "base": episode(True, 6),
                        "own": episode(True, 5),
                        "wrong": episode(True, 6)}}]}))
            result = analyze(path)["summary"]
            self.assertEqual(result["own_only_terminal_wins"], 1)
            self.assertEqual(result["positive_advantage_games"], 2)
            self.assertEqual(result["negative_advantage_games"], 0)
            self.assertEqual(result["terminal_wins"],
                             {"base": 1., "own": 2., "wrong": 1.})
            data = json.loads(path.read_text())
            data["games"][0]["arms"]["wrong"]["initial_observation"] = "other"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "target reset"):
                analyze(path)


if __name__ == "__main__":
    unittest.main()
