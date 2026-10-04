from __future__ import annotations

import unittest

from ttcl.trajectory_hyperlora.paired_utility import (
    PairedOutcome, content_hash, reinforce_loss, trajectory_utility,
)


class PairedUtilityTest(unittest.TestCase):
    def row(self, query: str, candidate: float, base: float, wrong: float) -> PairedOutcome:
        return PairedOutcome(content_hash("finished source"), content_hash(query), 17, 30,
                             candidate, base, wrong)

    def test_signed_harm_and_wrong_source_control(self):
        rows = [self.row("future A", 1, 0, 0), self.row("future B", 0, 1, 1)]
        result = trajectory_utility(rows)
        self.assertEqual(result["mean_delta_base"], 0)
        self.assertEqual(result["mean_downside"], 0.5)
        self.assertEqual(result["utility"], -0.25)

    def test_missing_reward_and_duplicate_probe_rejected(self):
        row = self.row("future A", 1, 0, 0)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            trajectory_utility([row, row])
        invalid = self.row("future B", float("nan"), 0, 0)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            trajectory_utility([invalid])

    def test_wrong_adapter_can_be_worse_without_candidate_gain(self):
        result = trajectory_utility([self.row("future A", 1, 1, 0)])
        self.assertEqual(result["mean_delta_wrong"], 1)
        self.assertEqual(result["utility"], 0)

    def test_policy_gradient_keeps_negative_utility(self):
        import torch
        logit = torch.tensor(0.0, requires_grad=True)
        sampled_logp = torch.nn.functional.logsigmoid(logit)
        reinforce_loss(sampled_logp, -1.0).backward()
        self.assertGreater(float(logit.grad), 0)


if __name__ == "__main__":
    unittest.main()
