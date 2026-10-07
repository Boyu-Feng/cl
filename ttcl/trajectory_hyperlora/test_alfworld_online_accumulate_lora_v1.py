"""Checks that online LoRA factors accumulate without mutating old factors."""

import unittest

import torch

from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import (
    add_factors, factors_hash, mean_factors,
)


class AdditiveLoRATest(unittest.TestCase):
    def test_adds_new_factor_without_mutating_previous_state(self):
        first = torch.ones(1, 3, 2)
        old = add_factors(None, [first])
        current = add_factors(old, [torch.full_like(first, 2)])
        self.assertTrue(torch.equal(old[0], first))
        self.assertTrue(torch.equal(current[0], torch.full_like(first, 3)))
        self.assertNotEqual(factors_hash(old), factors_hash(current))

    def test_rejects_nonfinite_or_shape_changed_increment(self):
        old = [torch.ones(1, 3, 2)]
        with self.assertRaises(ValueError):
            add_factors(old, [torch.full((1, 3, 2), float("nan"))])
        with self.assertRaises(ValueError):
            add_factors(old, [torch.ones(1, 2, 2)])

    def test_running_mean_keeps_scale_and_is_order_independent(self):
        one = [torch.ones(1, 3, 2)]
        three = [torch.full((1, 3, 2), 3.)]
        first = mean_factors(None, one, 0)
        combined = mean_factors(first, three, 1)
        self.assertTrue(torch.equal(combined[0], torch.full((1, 3, 2), 2.)))
        self.assertTrue(torch.equal(first[0], one[0]))


if __name__ == "__main__":
    unittest.main()
