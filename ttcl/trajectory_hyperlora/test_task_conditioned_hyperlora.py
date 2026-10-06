"""The next task must affect generated factors through the new pair module."""

import unittest

import torch
from torch import nn

from ttcl.trajectory_hyperlora.direct_composition_pilot import (
    DirectRelationHyperLoRA,
)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(8, 8)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyBlock()])

    def get_input_embeddings(self):
        return self.embedding


class TaskConditionedHyperLoRATest(unittest.TestCase):
    def test_pair_module_changes_factors_and_receives_gradient(self):
        agent = DirectRelationHyperLoRA(
            TinyModel(), rank=1, layers=1, encoder_kind="contextual",
            task_conditioned=True)
        source = {"contextual": torch.arange(1., 9.).unsqueeze(0)}
        target_a = {"contextual": torch.tensor(
            [[1., 2., 3., 4., 5., 6., 7., 8.]])}
        target_b = {"contextual": torch.tensor(
            [[8., 7., 6., 5., 4., 3., 2., 1.]])}
        agent.set_source(source, target_fields=target_a)
        initial_a = agent.adapters[0].b.detach().clone()
        agent.set_source(source, target_fields=target_b)
        self.assertTrue(torch.equal(initial_a, agent.adapters[0].b))

        with torch.no_grad():
            agent.task_pair_latent[1].weight[0, 0] = 1.
            agent.b_heads[0].weight[:, 0] = 1.
        agent.set_source(source, target_fields=target_a)
        changed_a = agent.adapters[0].b.detach().clone()
        agent.set_source(source, target_fields=target_b)
        self.assertFalse(torch.equal(changed_a, agent.adapters[0].b))
        agent.adapters[0].b.sum().backward()
        gradient = agent.task_pair_latent[1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.)
        with self.assertRaisesRegex(ValueError, "needs source and target"):
            agent.set_source(source)


if __name__ == "__main__":
    unittest.main()
