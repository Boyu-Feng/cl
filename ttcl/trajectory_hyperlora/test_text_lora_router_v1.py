"""Check that the router reads text and respects paired reward choices."""

import unittest

import torch

from ttcl.trajectory_hyperlora.train_text_lora_router_v1 import (
    fit, score, text_features,
)


class TextLoraRouterTest(unittest.TestCase):
    def test_text_changes_features_without_family_metadata(self):
        source = [{"observation": "Your task is to: put a mug in desk.",
                   "action": "look", "feedback": "ok"}]
        one = text_features("Your task is to: put a mug in desk.", source)
        two = text_features("Your task is to: heat an egg in microwave.", source)
        self.assertEqual(one.shape, (1026,))
        self.assertFalse(torch.equal(one, two))
        self.assertGreater(one[-2], two[-2])

    def test_reward_gate_score(self):
        rows = [{"game": "a", "arms": {"base": {"reward": 1},
                                        "lora": {"reward": 0}}},
                {"game": "b", "arms": {"base": {"reward": 0},
                                        "lora": {"reward": 1}}}]
        result = score(rows, torch.tensor([-1.0, 1.0]), 0)
        self.assertEqual(result["gated"], 2)
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        weights = fit(x, torch.tensor([-1.0, 1.0], dtype=torch.float64), 0.1)
        self.assertLess((x @ weights)[0], 0)
        self.assertGreater((x @ weights)[1], 0)


if __name__ == "__main__":
    unittest.main()
