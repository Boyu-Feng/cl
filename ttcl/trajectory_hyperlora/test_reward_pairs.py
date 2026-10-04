"""Protect source/target content binding for new paired reward episodes."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from ttcl.trajectory_hyperlora.prepare_alf_next_task import digest_json, partition
from ttcl.trajectory_hyperlora.prepare_alf_reward_pairs import validate_reward_review
from ttcl.trajectory_hyperlora.paired_reward_gate import fit_gate, score_gate


class RewardPairReviewTest(unittest.TestCase):
    def test_rejects_changed_game_and_unreviewed_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_game = "json_2.1.1/train/pick_and_place_simple-Book-None-Bed-1/trial_a/game.tw-pddl"
            target_game = next(
                f"json_2.1.1/train/pick_and_place_simple-CD-None-Desk-{i}/trial_b/game.tw-pddl"
                for i in range(100) if partition(
                    f"json_2.1.1/train/pick_and_place_simple-CD-None-Desk-{i}/trial_b/game.tw-pddl") == "train")
            game_path = root / target_game
            game_path.parent.mkdir(parents=True)
            game_path.write_text("frozen game")
            source_path = root / "source_episode.json"
            source_path.write_text("frozen source")
            source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
            game_sha = hashlib.sha256(game_path.read_bytes()).hexdigest()
            content = {"source_game": source_game,
                       "source_episode_sha256": source_sha,
                       "source_records": [{"observation": "o", "action": "a",
                                           "feedback": "valid command"}],
                       "target_game": target_game,
                       "target_game_sha256": game_sha}
            key = digest_json(content)
            row = {"split": "train", "family": "pick_and_place_simple",
                   **content, "source_episode": str(source_path),
                   "input_content_sha256": key}
            manifest = {"plan_sha256": "p", "historical_review_sha256": "h",
                        "candidates": [row]}
            note = {"input_content_sha256": key,
                    "source_episode_sha256": source_sha,
                    "target_game_sha256": game_sha,
                    "approved": True, "review_note": "Checked source and target"}
            review = {"plan_sha256": "p", "historical_review_sha256": "h",
                      "annotations": [note]}
            self.assertEqual(validate_reward_review(manifest, review, root), [row])
            game_path.write_text("changed game")
            with self.assertRaisesRegex(ValueError, "Target game content changed"):
                validate_reward_review(manifest, review, root)
            game_path.write_text("frozen game")
            review["annotations"] = []
            with self.assertRaisesRegex(ValueError, "Every new reward target"):
                validate_reward_review(manifest, review, root)

    def test_signed_reward_gate_keeps_base_on_ties(self) -> None:
        def pair(name: str, task: str, base: float, lora: float) -> dict:
            observation = "Your task is to: " + task
            return {"game": f"json_2.1.1/train/{name}/trial/game.tw-pddl",
                    "base": {"reward": base,
                             "initial_observation": observation},
                    "generated": {"reward": lora,
                                  "initial_observation": observation}}

        train = [pair("look_at_obj_in_light-CD-None-DeskLamp-1",
                      "look at cd under the desklamp.", 0.0, 1.0),
                 pair("pick_and_place_simple-CD-None-Desk-1",
                      "put a cd in desk.", 1.0, 0.0),
                 pair("pick_two_obj_and_place-CD-None-Desk-1",
                      "put two cd in desk.", 1.0, 1.0)]
        gate = fit_gate(train)
        self.assertTrue(gate["look_at_obj_in_light"]["use_lora"])
        self.assertFalse(gate["pick_and_place_simple"]["use_lora"])
        self.assertFalse(gate["pick_two_obj_and_place"]["use_lora"])
        self.assertEqual(score_gate(train, gate)["gated_successes"], 3.0)


if __name__ == "__main__":
    unittest.main()
