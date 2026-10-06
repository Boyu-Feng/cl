from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from ttcl.trajectory_hyperlora.relational_router_pilot import (
    centered_relation, tokenize_records,
)


class RelationTest(unittest.TestCase):
    def test_same_marginals_opposite_step_relation(self):
        observations = torch.tensor([[1., 0.], [0., 1.]])
        actions = torch.tensor([[1., 0.], [0., 1.]])
        feedback = torch.ones_like(observations)
        same = centered_relation(observations, actions, feedback)
        opposite = centered_relation(observations, actions.flip(0), feedback)
        self.assertTrue(torch.allclose(same, -opposite))
        self.assertGreater(float(same[0, 0]), 0)

    def test_rejected_attempt_has_zero_weight(self):
        observations = torch.tensor([[1., 0.], [1., 0.],
                                     [0., 1.], [0., 1.]])
        actions = torch.tensor([[0., 1.], [1., 0.],
                                [1., 0.], [0., 1.]])
        feedback = torch.tensor([[0., 0.], [1., 1.],
                                 [0., 0.], [1., 1.]])
        result = centered_relation(observations, actions, feedback)
        self.assertGreater(float(result[0, 0]), 0)

    def test_fields_are_required(self):
        with self.assertRaisesRegex(ValueError, "Completed steps"):
            tokenize_records(None, [{"observation": "here", "action": "move"}], "cpu")

    def test_head_tail_keeps_goal_after_long_observation(self):
        class CharTokenizer:
            pad_token_id = 0

            def __call__(self, texts, **_kwargs):
                return SimpleNamespace(input_ids=[list(map(ord, text))
                                                  for text in texts])

        source = [{"observation": "ROOM" + "x" * 50 + "GOAL",
                   "action": "go", "feedback": "valid"}]
        fields = tokenize_records(CharTokenizer(), source, "cpu",
                                  max_tokens=10,
                                  truncation_mode="head_tail")
        ids, mask = fields["observation"]
        self.assertEqual("".join(map(chr, ids[0, mask[0].bool()].tolist())),
                         "ROOMx" + "xGOAL")

    def test_first_observation_can_condition_every_step(self):
        class CharTokenizer:
            pad_token_id = 0

            def __call__(self, texts, **_kwargs):
                return SimpleNamespace(input_ids=[list(map(ord, text))
                                                  for text in texts])

        source = [{"observation": "ROOM" + "x" * 50 + "GOAL",
                   "action": "look", "feedback": "valid"},
                  {"observation": "LOCAL", "action": "go",
                   "feedback": "valid"}]
        fields = tokenize_records(CharTokenizer(), source, "cpu",
                                  max_tokens=10, truncation_mode="head_tail",
                                  repeat_initial_observation=True)
        ids, mask = fields["observation"]
        self.assertEqual("".join(map(chr, ids[1, mask[1].bool()].tolist())),
                         "LOCAL" + "xGOAL")


if __name__ == "__main__":
    unittest.main()
