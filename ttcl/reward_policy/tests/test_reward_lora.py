"""Gradient direction, causal reward baseline, and online ordering checks."""

import tempfile
import unittest
import copy
from pathlib import Path
from types import SimpleNamespace

import torch

from ttcl.reward_policy.reward_lora import (
    RewardBaseline,
    RewardLoRA,
    Sample,
    categorical_kl,
    clipped_policy_loss,
)
from ttcl.reward_policy.run_reward_policy_benchmark import BENCH, run


class RewardTests(unittest.TestCase):
    def test_causal_baseline(self):
        baseline = RewardBaseline(warmup=2)
        self.assertFalse(baseline.observe(0.4)["ready"])
        self.assertEqual(baseline.observe(0.4)["advantage"], 0)
        result = baseline.observe(0.5)
        self.assertEqual(result["baseline"], 0.4)
        self.assertAlmostEqual(result["advantage"], 2)
        self.assertLess(baseline.observe(0.2)["advantage"], 0)
        with self.assertRaises(ValueError):
            baseline.observe(float("nan"))

    def test_gradient_direction(self):
        for advantage in (1.0, -1.0):
            logits = torch.zeros(2, requires_grad=True)
            logp = logits.log_softmax(-1)[0:1]
            clipped_policy_loss(logp, logp.detach(), advantage).backward()
            updated = (logits - 0.1 * logits.grad).softmax(-1)[0]
            self.assertGreater(float((updated - 0.5) * advantage), 0)

    def test_clipping_stops_favorable_overshoot(self):
        for ratio, advantage in ((1.5, 1.0), (0.5, -1.0)):
            logp = torch.tensor([ratio]).log().requires_grad_()
            clipped_policy_loss(logp, torch.zeros(1), advantage).backward()
            self.assertEqual(float(logp.grad), 0)

    def test_full_distribution_kl(self):
        logits = torch.tensor([2.0, 0.0], requires_grad=True)
        logp = logits.log_softmax(-1)
        self.assertAlmostEqual(float(categorical_kl(logp, logp)), 0)
        loss = categorical_kl(logp, torch.zeros(2).log_softmax(-1))
        self.assertGreater(float(loss), 0)
        loss.backward()
        self.assertGreater(float(logits.grad[0]), 0)

    def test_score_before_update_and_terminal_skip(self):
        policies = []

        class FakePolicy:
            def __init__(self, args):
                self.updates = 0
                self.baseline = RewardBaseline(warmup=1)
                policies.append(self)

            def sample(self, prompt):
                self.assert_no_reward(prompt)
                return SimpleNamespace(text='{"transmitters": []}', truncated=False)

            @staticmethod
            def assert_no_reward(prompt):
                assert '"reward":' not in prompt

            def observe(self, sample, reward, *, update):
                assert isinstance(reward, float)
                stats = self.baseline.observe(reward)
                accepted = update and stats["ready"]
                self.updates += int(accepted)
                return {**stats, "reason": "updated" if accepted else "skipped"}

            def save(self, path):
                pass

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                output_dir=directory,
                data_path=str(
                    BENCH / "data/blind_spectrum_monitoring/mixed_grid_lifecycle.jsonl"
                ),
                num_scans=4,
                seed=42,
                mode="online",
                history_scans=2,
                model="fake",
            )
            metrics = run(args, policy_factory=FakePolicy)
            import json

            records = [
                json.loads(row)
                for row in (Path(directory) / "responses.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(
                [r["updates_before_answer"] for r in records], [0, 0, 1, 2]
            )
            self.assertEqual(metrics["num_updates"], 2)
            with self.assertRaises(ValueError):
                run(args, policy_factory=FakePolicy)

    def test_real_lora_update_and_kl_rollback(self):
        from peft import LoraConfig, get_peft_model
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(42)
        policy = RewardLoRA.__new__(RewardLoRA)
        policy.args = SimpleNamespace(
            device="cpu", update_steps=1, kl_beta=0.05, max_update_kl=1.0
        )
        base = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=16,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=32,
            )
        )
        policy.model = get_peft_model(
            base,
            LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            ),
        )
        policy.parameters = [p for p in policy.model.parameters() if p.requires_grad]
        policy.optimizer = torch.optim.AdamW(policy.parameters, lr=0.01)
        policy.baseline = RewardBaseline(warmup=1)
        policy.updates = 0
        sample = Sample(torch.tensor([[1, 2, 3, 4]]), 2, torch.zeros(2), "", False)
        with torch.no_grad():
            _, sample.old_logp = policy._distribution(sample)
        frozen = {
            k: p.detach().clone()
            for k, p in policy.model.named_parameters()
            if not p.requires_grad
        }
        policy.observe(sample, 0.4, update=False)
        result = policy.observe(sample, 0.9, update=True)
        self.assertTrue(result["accepted"])
        self.assertGreater(result["mean_sampled_logprob_change"], 0)
        self.assertTrue(
            all(
                torch.equal(frozen[k], p)
                for k, p in policy.model.named_parameters()
                if k in frozen
            )
        )
        saved = [p.detach().clone() for p in policy.parameters]
        optimizer = copy.deepcopy(policy.optimizer.state_dict())
        policy.args.max_update_kl = 1e-12
        with torch.no_grad():
            _, sample.old_logp = policy._distribution(sample)
        result = policy.observe(sample, 0.0, update=True)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "kl_or_numerical_rejection")
        self.assertEqual(policy.updates, 1)
        self.assertTrue(
            all(torch.equal(p, old) for p, old in zip(policy.parameters, saved))
        )
        restored = policy.optimizer.state_dict()
        self.assertEqual(optimizer["param_groups"], restored["param_groups"])
        for key, state in optimizer["state"].items():
            for name, value in state.items():
                self.assertTrue(torch.equal(value, restored["state"][key][name]))


if __name__ == "__main__":
    unittest.main()
