from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ttcl.trajectory_hyperlora.audit_alf_pairs import audit


class AuditAlfPairsTest(unittest.TestCase):
    def test_train_only_pair_and_content_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plan.json").write_text(json.dumps({
                "training": [{"games": ["train/a", "train/b"]}],
                "evaluation": [{"games": ["valid/c"]}],
            }))
            for stage, game in enumerate(("train/a", "train/b")):
                path = root / "training" / "delta" / "batch_000" / "seq_0" / f"task_{stage}" / "episode.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({
                    "game": game, "initial_observation": f"observation {stage}",
                    "trajectory": [{"action": "look", "observation": "seen",
                                    "valid_command": stage == 0}],
                    "reward": float(stage), "termination": "complete", "seed": stage,
                }))
            result = audit(root)
            self.assertEqual(result["counts"]["stage_0"], 1)
            self.assertFalse(result["pairs"][0]["reviewed_target"])
            self.assertEqual(result["pairs"][0]["source_invalid_commands"], 0)
            self.assertNotEqual(result["pairs"][0]["source_content_sha256"],
                                result["pairs"][0]["future_initial_observation_sha256"])


if __name__ == "__main__":
    unittest.main()
