"""Small policy-gradient mechanism check; not a benchmark result."""

import unittest

import torch

from ttcl.trajectory_hyperlora.trajectory_lora_rl import (
    Episode, PairedCase, content_sha256, paired_episode_policy_gradient,
)


class ToyTrajectoryActor(torch.nn.Module):
    """A two-action actor whose generated logit shift depends on source text."""

    def __init__(self) -> None:
        super().__init__()
        self.generator = torch.nn.Embedding(2, 1)
        torch.nn.init.zeros_(self.generator.weight)

    def _source_index(self, source_text: str) -> int:
        return 1 if '"signal":"blue"' in source_text else 0

    def rollout(self, case, source_text, *, sample_seed):
        index = self._source_index(source_text or "")
        with torch.no_grad():
            logit = (self.generator.weight[index, 0] if source_text
                     else torch.tensor(0.0))
            generator = torch.Generator().manual_seed(sample_seed)
            action = int(torch.bernoulli(logit.sigmoid(), generator=generator))
        desired = int(case.source_events[0]["signal"] == "blue")
        return Episode(case.target_id, case.target_content_sha256,
                       case.reset_seed, float(action == desired), "complete",
                       (action,), 1)

    def action_log_probability(self, episode, source_text):
        logit = self.generator.weight[self._source_index(source_text), 0]
        return torch.distributions.Bernoulli(logits=logit).log_prob(
            torch.tensor(float(episode.decisions[0])))


class PairedPolicyGradientTest(unittest.TestCase):
    def test_reward_trains_source_conditioned_generator(self):
        actor = ToyTrajectoryActor()
        optimizer = torch.optim.SGD(actor.parameters(), lr=0.5)
        cases = [PairedCase([{"signal": signal, "action": "seen",
                             "feedback": "done"}], signal,
                            content_sha256({"target": signal}), 7, "train")
                 for signal in ("red", "blue")]
        for step in range(300):
            case = cases[step % 2]
            loss, record = paired_episode_policy_gradient(
                actor, case, sample_seed=step * 31 + 11, max_budget=1)
            self.assertEqual(record["status"], "complete")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertLess(actor.generator.weight[0, 0].item(), -1.0)
        self.assertGreater(actor.generator.weight[1, 0].item(), 1.0)

    def test_rejects_test_targets(self):
        with self.assertRaises(ValueError):
            PairedCase([{"event": 1}], "x", "a" * 64, 1, "test")


if __name__ == "__main__":
    unittest.main()
