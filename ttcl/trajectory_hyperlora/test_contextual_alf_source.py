from __future__ import annotations

import unittest

import torch
from torch import nn

from ttcl.trajectory_hyperlora.contextual_alf_source import source_text
from ttcl.trajectory_hyperlora.direct_composition_pilot import DirectRelationHyperLoRA


class TinyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            nn.Module() for _ in range(2)])
        for layer in self.model.layers:
            layer.mlp = nn.Module()
            layer.mlp.down_proj = nn.Linear(8, 8)

    def get_input_embeddings(self):
        return self.embedding


class ContextualSourceTest(unittest.TestCase):
    def test_full_public_trajectory_is_encoded_without_rule_summary(self):
        records = [{"observation": "room; goal: put a cup on shelf",
                    "action": "go to desk 1", "feedback": "cup visible"}]
        text = source_text(records)
        self.assertIn("goal: put a cup on shelf", text)
        self.assertIn("Action: go to desk 1", text)
        with self.assertRaisesRegex(ValueError, "at least one"):
            source_text([])

    def test_contextual_vector_generates_adapters_and_receives_gradient(self):
        agent = DirectRelationHyperLoRA(TinyActor(), rank=2, layers=2,
                                        encoder_kind="contextual")
        self.assertFalse(any(param.requires_grad
            for param in agent.encoder.parameters()))
        for head in agent.b_heads:
            nn.init.normal_(head.weight, std=.1)
        vector = torch.randn(1, 8)
        agent.set_source({"contextual": vector})
        self.assertEqual(tuple(agent.adapters[0].b.shape), (1, 8, 2))
        loss = sum(layer.b.square().sum() for layer in agent.adapters)
        loss.backward()
        self.assertGreater(float(agent.contextual_latent[1].weight.grad.norm()), 0)
        agent.set_source(None)
        self.assertTrue(all(layer.b is None for layer in agent.adapters))


if __name__ == "__main__":
    unittest.main()
