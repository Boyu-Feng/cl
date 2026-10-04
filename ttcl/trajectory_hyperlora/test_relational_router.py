from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
