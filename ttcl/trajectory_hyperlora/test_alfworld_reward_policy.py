"""Small checks for reward-policy action masking and trajectory features."""

from __future__ import annotations

import unittest

import torch

from ttcl.trajectory_hyperlora.train_alfworld_reward_policy import (
    AdapterPolicy, FAMILIES, features, pca_source_features, rl_partition,
)


class RewardPolicyTest(unittest.TestCase):
    def test_source_features_distinguish_trajectories(self) -> None:
        latents = {"a": torch.tensor([1., 0., 0.]),
                   "b": torch.tensor([0., 1., 0.]),
                   "c": torch.tensor([0., 0., 1.])}
        mean, basis, scale = pca_source_features(latents, rank=2)
        one = features(FAMILIES[0], latents["a"], mean, basis, scale)
        two = features(FAMILIES[0], latents["b"], mean, basis, scale)
        self.assertFalse(torch.equal(one, two))
        self.assertEqual(one.numel(), len(FAMILIES) + 2)

    def test_no_source_masks_source_adapter(self) -> None:
        policy = AdapterPolicy(10)
        logits = policy(torch.zeros(10), has_source=False)
        self.assertLess(float(logits[2]), -1e8)
        self.assertEqual(int(logits.argmax()), 0)

    def test_rl_partition_is_stable(self) -> None:
        game = "json_2.1.1/train/pick_and_place_simple-CD-None-Desk-1/trial/game.tw-pddl"
        self.assertEqual(rl_partition(game), rl_partition(game))
        self.assertIn(rl_partition(game), {"train", "dev"})

    def test_expected_reward_gradient_favors_successful_arm(self) -> None:
        policy = AdapterPolicy(10)
        x = torch.ones(1, 10)
        reward = torch.tensor([[0., 0., 1.]])
        probabilities = policy(x).softmax(-1)
        loss = -(probabilities * (reward - reward.mean(-1, keepdim=True))).sum()
        loss.backward()
        self.assertLess(float(policy.linear.bias.grad[2]), 0)
        self.assertGreater(float(policy.linear.bias.grad[0]), 0)


if __name__ == "__main__":
    unittest.main()
