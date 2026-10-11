import unittest

import torch

from ttcl.trajectory_hyperlora.contrastive_failure_factors_v30 import (
    contrastive_failure_update, factor_norm,
)


class ContrastiveFailureFactorsTest(unittest.TestCase):
    def test_direction_and_relative_cap(self):
        old = [torch.tensor([1., 0.])]
        good = [torch.tensor([2., 0.])]
        bad = [torch.tensor([0., 0.])]
        new, info = contrastive_failure_update(old, good, bad)
        self.assertTrue(torch.allclose(new[0], torch.tensor([1.25, 0.])))
        self.assertAlmostEqual(info["applied_norm"], .25)
        self.assertLessEqual(factor_norm([new[0] - old[0]]), .25)

    def test_empty_state_and_zero_direction(self):
        good = [torch.tensor([1., 0.])]
        bad = [torch.tensor([1., 0.])]
        new, info = contrastive_failure_update(None, good, bad)
        self.assertTrue(torch.equal(new[0], torch.zeros(2)))
        self.assertEqual(info["applied_norm"], 0.)

    def test_bad_shapes_and_nonfinite(self):
        with self.assertRaises(ValueError):
            contrastive_failure_update(None, [torch.ones(2)], [torch.ones(3)])
        with self.assertRaises(ValueError):
            contrastive_failure_update(None, [torch.tensor([float('nan')])],
                                       [torch.zeros(1)])


if __name__ == "__main__":
    unittest.main()
