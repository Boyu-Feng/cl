import unittest
from types import SimpleNamespace

import torch

from ttcl.trajectory_hyperlora.alfworld_failure_retry_residual_v31b import (
    bounded_residual, inject_dynamic_residual,
)


class DummyLayer:
    def __init__(self):
        self.base = SimpleNamespace(out_features=2)
        self.a = torch.tensor([[1., 0.]])
        self.b = torch.zeros(1, 2, 1)
        self.rank = 1


class DummyAgent:
    def __init__(self):
        self.adapters = [DummyLayer()]

    def set_source(self, source, **kwargs):
        self.adapters[0].b = (torch.tensor([[[2.], [0.]]]) if source is not None
                              else torch.zeros(1, 2, 1))


class FailureRetryResidualTest(unittest.TestCase):
    def test_bound_and_opposite_direction(self):
        agent = DummyAgent()
        expert = [torch.tensor([[[2.], [0.]]])]
        failed = [torch.tensor([[[4.], [0.]]])]
        agent.set_source(None)
        agent.adapters[0].b = None
        residual, info = bounded_residual(agent, expert, failed)
        self.assertAlmostEqual(info["residual_effective_norm"], .5)
        self.assertTrue(torch.allclose(residual[0],
                                       torch.tensor([[[.5], [0.]]])))
        with inject_dynamic_residual(agent, residual):
            agent.set_source({"source": "expert"})
            self.assertAlmostEqual(float(agent.adapters[0].b[0, 0, 0]), 2.5)
            agent.set_source({"source": "expert"})
            self.assertAlmostEqual(float(agent.adapters[0].b[0, 0, 0]), 2.5)
        self.assertAlmostEqual(float(agent.adapters[0].b[0, 0, 0]), 0.)

    def test_invalid_factor_rejected(self):
        agent = DummyAgent()
        with self.assertRaises(ValueError):
            bounded_residual(agent, [torch.ones(1, 2, 1)],
                             [torch.full((1, 2, 1), float("nan"))])


if __name__ == "__main__":
    unittest.main()
