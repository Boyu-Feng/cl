"""Checks for the paired same-query, opposite-source training signal."""

import random
import unittest

import torch

from ttcl.trajectory_hyperlora.direct_composition_pilot import records
from ttcl.trajectory_hyperlora.paired_counterfactual_lora import paired_objective


class PairedCounterfactualTest(unittest.TestCase):
    def test_source_pair_only_changes_actions(self):
        for policy in (0, 1, 3, 4, 6, 7):
            source, numbers = records(policy, random.Random(42), test=False)
            opposite, opposite_numbers = records(policy ^ 7,
                                                  random.Random(42), test=False)
            self.assertEqual(numbers, opposite_numbers)
            for left, right in zip(source, opposite, strict=True):
                self.assertEqual(left["observation"], right["observation"])
                self.assertEqual(left["feedback"], right["feedback"])
                self.assertNotEqual(left["action"], right["action"])

    def test_both_sources_receive_correct_gradient(self):
        source = torch.zeros((1, 3), requires_grad=True)
        opposite = torch.zeros((1, 3), requires_grad=True)
        loss = paired_objective(source, opposite, 0, 1, 2.0, .3)
        loss.backward()
        self.assertLess(float(source.grad[0, 0]), 0)
        self.assertLess(float(opposite.grad[0, 1]), 0)


if __name__ == "__main__":
    unittest.main()
