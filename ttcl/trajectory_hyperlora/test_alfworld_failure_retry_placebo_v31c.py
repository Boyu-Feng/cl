import unittest
from types import SimpleNamespace

import torch

from ttcl.trajectory_hyperlora.alfworld_failure_retry_placebo_v31c import (
    matched_random_residual,
)
from ttcl.trajectory_hyperlora.alfworld_failure_retry_residual_v31b import (
    effective_norm,
)


class MatchedPlaceboTest(unittest.TestCase):
    def test_reproducible_matched_effective_norm(self):
        layer = SimpleNamespace(a=torch.tensor([[1., 2.]]), b=None, rank=1,
                                base=SimpleNamespace(out_features=2))
        agent = SimpleNamespace(adapters=[layer])
        template = [torch.ones(1, 2, 1)]
        one = matched_random_residual(agent, template, .75, 42)
        two = matched_random_residual(agent, template, .75, 42)
        self.assertTrue(torch.equal(one[0], two[0]))
        self.assertAlmostEqual(effective_norm(agent, one), .75, places=5)


if __name__ == "__main__":
    unittest.main()
