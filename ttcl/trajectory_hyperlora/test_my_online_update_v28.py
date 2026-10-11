"""Regression checks for the trainable event-to-LoRA path."""

import unittest

import torch

from ttcl.trajectory_hyperlora.my_online_update_v28_model import OnlineUpdateModel
from ttcl.trajectory_hyperlora.train_my_online_update_v28_reward import (
    effective_residual_norm,
)


class OnlineUpdateTest(unittest.TestCase):
    def test_zero_residual_norm_has_finite_gradient(self):
        class Layer:
            rank = 2
            base = type("Base", (), {"out_features": 3})()
            a = torch.ones(2, 4)

        class Agent:
            adapters = [Layer()]
            b_heads = [torch.nn.Linear(3, 6, bias=False)]

        mean = torch.zeros(3, requires_grad=True)
        norm = effective_residual_norm(Agent(), torch.eye(3), mean)
        norm.square().backward()
        self.assertTrue(torch.isfinite(mean.grad).all())

    def test_event_padding_is_masked_and_both_heads_train(self):
        torch.manual_seed(7)
        model = OnlineUpdateModel(width=6, hidden=8, code_dim=3,
                                  max_events=4)
        global_vectors = torch.randn(2, 6)
        events = torch.randn(2, 4, 6)
        deltas = torch.randn(2, 4, 6)
        counts = torch.tensor([2, 4])
        queries = torch.randn(3, 6)
        utility, residual = model(global_vectors, events, deltas,
                                  counts, queries)
        self.assertEqual(tuple(utility.shape), (3, 2))
        self.assertEqual(tuple(residual.shape), (3, 2, 3))
        altered = events.clone()
        altered[0, 2:] = 1000
        other, _ = model(global_vectors, altered, deltas, counts, queries)
        self.assertTrue(torch.allclose(utility[:, 0], other[:, 0]))
        (utility.square().mean() + residual.square().mean()).backward()
        self.assertTrue(torch.isfinite(model.global_proj.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
