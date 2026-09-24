"""Behavioral CPU tests: gradient direction, causal feedback, rollback and LoRA."""

import copy
import json
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from ttcl.ramp.reward_memory import (
    RewardBuffer,
    RewardConfig,
    RewardLearner,
    pair_loss,
    point_loss,
    preference_pairs,
)
from ttcl.ramp.run_reward_benchmark import parse_args, run


class Tokenizer:
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [2 + ord(x) % 12 for x in text]


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transitions = torch.nn.Embedding(16, 16)
        torch.nn.init.zeros_(self.transitions.weight)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.transitions(input_ids))


class SignalTest(unittest.TestCase):
    def test_gradient_sign_and_reward_magnitude(self):
        for advantage in (-2.0, -0.2, 0.0, 0.2, 2.0):
            delta = torch.tensor(0.0, requires_grad=True)
            point_loss(delta, advantage, 1.0).backward()
            self.assertAlmostEqual(delta.grad.item(), -advantage / 2, places=6)
        winner = torch.tensor(0.0, requires_grad=True)
        loser = torch.tensor(0.0, requires_grad=True)
        pair_loss(winner, loser, 0.8, 1.0).backward()
        self.assertLess(winner.grad.item(), 0)
        self.assertGreater(loser.grad.item(), 0)

    def test_causal_baseline_and_task_isolation(self):
        buffer = RewardBuffer(RewardConfig(warmup=2))
        first = buffer.observe("p", "a", 0.4)
        buffer.observe("p2", "a", 0.4)
        good = buffer.observe("p3", "a", 0.8)
        self.assertIsNone(first.baseline)
        self.assertEqual(first.advantage, 0)
        self.assertAlmostEqual(good.baseline, 0.4)
        self.assertGreater(good.advantage, 0)
        bad = buffer.observe("p4", "a", 0.0)
        self.assertLess(bad.advantage, 0)
        self.assertIsNone(buffer.observe("p", "a", 1.0, task_id="other").baseline)
        self.assertAlmostEqual(good.baseline, 0.4)  # Future reward did not rewrite it.

    def test_equal_rewards_have_no_point_signal(self):
        buffer = RewardBuffer(RewardConfig())
        values = [buffer.observe("p", str(i), 0.3) for i in range(6)]
        self.assertTrue(all(x.advantage == 0 for x in values))
        self.assertEqual(preference_pairs(values, buffer.config), [])

    def test_group_advantages_ignore_question_difficulty_history_and_order(self):
        config = RewardConfig(advantage_mode="group")
        buffer = RewardBuffer(config)
        buffer.observe("old question", "irrelevant", 1.0)
        candidates = [{"response": "good", "reward": 0.4},
                      {"response": "bad", "reward": 0.2}]
        items = buffer.observe_group("current question", candidates)
        other = RewardBuffer(config)
        other.observe("old question", "irrelevant", 0.0)
        shifted = [{**x, "reward": x["reward"] + 0.3} for x in reversed(candidates)]
        shifted_items = other.observe_group("different question", shifted)
        advantages = {x.response: x.advantage for x in shifted_items}
        for item in items:
            self.assertAlmostEqual(item.advantage, advantages[item.response])
            self.assertAlmostEqual(item.baseline, 0.3)
            self.assertEqual(item.group_size, 2)
        self.assertGreater(items[0].advantage, 0)
        self.assertLess(items[1].advantage, 0)
        self.assertEqual(items[0].group_id, items[1].group_id)
        saved = items[0].advantage
        buffer.observe_group("future question", [{"response": "new", "reward": 1.0}])
        self.assertEqual(items[0].advantage, saved)

    def test_group_ties_singletons_and_invalid_outputs(self):
        buffer = RewardBuffer(RewardConfig(advantage_mode="group"))
        for reward in (0.0, 0.3, 0.9):
            items = buffer.observe_group("p", [{"response": str(i), "reward": reward}
                                                for i in range(3)])
            self.assertTrue(all(x.advantage == 0 for x in items))
            self.assertEqual(preference_pairs(items, buffer.config), [])
        self.assertEqual(buffer.observe("new", "answer", 0.9).advantage, 0)
        invalid, valid = buffer.observe_group("another", [
            {"response": "invalid", "reward": 1.0, "valid": False},
            {"response": "valid", "reward": 0.0},
        ])
        self.assertEqual(invalid.advantage, 0)
        self.assertLess(valid.advantage, 0)

    def test_group_validation_is_atomic_and_requires_same_context(self):
        buffer = RewardBuffer(RewardConfig(advantage_mode="group"))
        good = {"response": "good", "reward": 0.9}
        for bad in ({"response": "bad", "reward": float("nan")},
                    {"response": "bad", "reward": 0.1, "prompt": "other"},
                    {"response": "bad", "reward": 0.1, "task_id": "other"}):
            with self.assertRaises(ValueError):
                buffer.observe_group("p", [good, bad])
            self.assertEqual(buffer.pending, [])
            self.assertEqual(buffer.count, 0)
            self.assertEqual(buffer.stats, {})
        with self.assertRaises(ValueError):
            RewardConfig(advantage_mode="unknown")

    def test_backtracking_configuration_validation(self):
        for value in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                RewardConfig(max_backtracks=value)
        for value in (0, 1, -0.5, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                RewardConfig(backtrack_factor=value)

    def test_only_identical_contexts_can_form_pairs(self):
        c = RewardConfig()
        buffer = RewardBuffer(c)
        a = buffer.observe("question 1", "good", 0.9)
        b = buffer.observe("question 2", "bad", 0.1)
        self.assertEqual(preference_pairs([a, b], c), [])
        b = replace(b, prompt=a.prompt)
        self.assertEqual(preference_pairs([a, b], c), [(0, 1, 0.8)])
        for changed in (
            replace(b, task_id="other"),
            replace(b, response="good"),
            replace(b, reward=0.9),
        ):
            self.assertEqual(preference_pairs([a, changed], c), [])
        self.assertEqual(preference_pairs([replace(a, valid=False), b], c), [])

    def test_invalid_outputs_and_reward_validation(self):
        buffer = RewardBuffer(RewardConfig(warmup=1))
        buffer.observe("p", "a", 0)
        invalid = buffer.observe("p", "a", 1, valid=False)
        self.assertEqual(invalid.advantage, 0)
        for reward in (float("nan"), float("inf"), -0.1, 1.1):
            with self.assertRaises(ValueError):
                buffer.observe("p", "a", reward)
        with self.assertRaises(ValueError):
            RewardConfig(scale_floor=0)

    def test_fresh_data_survives_small_replay_capacity(self):
        buffer = RewardBuffer(RewardConfig(capacity=1, replay_size=0))
        for i in range(5):
            buffer.observe("p", str(i), 0.4)
        self.assertEqual(len(buffer.select()), 5)
        buffer.commit()
        self.assertEqual(len(buffer.replay), 1)
        self.assertEqual(buffer.pending, [])


class TrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def learner(self, **kwargs):
        defaults = dict(
            learning_rate=0.02,
            replay_size=0,
            anchor_weight=0,
            pair_weight=0,
            max_token_drift=10,
        )
        defaults.update(kwargs)
        learner = RewardLearner(TinyLM(), Tokenizer(), RewardConfig(**defaults))
        learner.observe("p", "warmup", 0.5)
        learner.observe("p", "warmup", 0.5)
        learner.buffer.commit()
        return learner

    def test_positive_and_negative_rewards_change_probability_in_correct_direction(
        self,
    ):
        for reward, sign in ((0.9, 1), (0.1, -1)):
            learner = self.learner()
            item = learner.observe("p", "answer", reward, response_ids=[3])
            encoded, _ = learner.encode(item)
            before = float(learner.token_logps(encoded).detach().mean())
            report = learner.update()
            after = float(learner.token_logps(encoded).detach().mean())
            self.assertTrue(report["accepted"])
            self.assertGreater(sign * (after - before), 0)

    def test_pair_only_improves_winner_over_loser(self):
        learner = self.learner(point_weight=0, pair_weight=1)
        winner = learner.observe("p", "winner", 0.9, response_ids=[3])
        loser = learner.observe("p", "loser", 0.1, response_ids=[4])
        report = learner.update()
        self.assertEqual(report["pair_count"], 1)
        win = learner.token_logps(learner.encode(winner)[0]).mean()
        lose = learner.token_logps(learner.encode(loser)[0]).mean()
        self.assertGreater(float((win - lose).detach()), 0)

    def test_group_update_uses_preference_and_reports_direction_and_steps(self):
        learner = self.learner(advantage_mode="group", pair_weight=1)
        learner.observe_group("p", [
            {"response": "winner", "reward": 0.9, "response_ids": [3]},
            {"response": "loser", "reward": 0.1, "response_ids": [4]},
        ])
        report = learner.update()
        self.assertTrue(report["accepted"])
        self.assertEqual(report["point_count"], 2)
        self.assertEqual(report["pair_count"], 1)
        self.assertEqual(report["anchor_count"], 0)
        self.assertEqual(report["optimizer_steps"], 1)
        self.assertEqual(report["retained_optimizer_steps"], 1)
        self.assertGreater(report["mean_pair_margin_gain"], 0)
        self.assertGreater(report["mean_signed_logprob_delta"], 0)
        self.assertEqual(report["fraction_advantages_followed"], 1)
        self.assertEqual(learner.state_dict()["optimizer_steps"], 1)

    def test_group_ties_do_not_manufacture_an_anchor_signal(self):
        learner = self.learner(advantage_mode="group", anchor_weight=1, pair_weight=1)
        learner.observe_group("p", [{"response": str(i), "reward": 0.9}
                                     for i in range(3)])
        report = learner.update()
        self.assertEqual(report["reason"], "no_reward_signal")
        self.assertEqual(report["anchor_count"], 0)
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertEqual(learner.updates, 0)

    def test_trust_penalty_only_acts_after_first_step(self):
        for epochs in (1, 2):
            learner = self.learner(epochs=epochs)
            learner.observe("p", "answer", 0.9, response_ids=[3])
            report = learner.update()
            self.assertEqual(report["optimizer_steps"], epochs)
            self.assertEqual(report["branch_losses"]["trust"][0], 0)
            self.assertEqual(report["trust_active_steps"], epochs - 1)

    def test_prompt_tokens_are_not_targets_and_exact_generation_tokens_are_kept(self):
        learner = self.learner()
        item = learner.observe("prompt", "decoded text", 0.9, response_ids=[3, 1])
        tokens, _ = learner.encode(item)
        self.assertEqual(tokens[0][-2:], [3, 1])
        logps = learner.token_logps(tokens)
        self.assertEqual(logps.shape, (2,))
        logps.mean().backward()
        changed = (
            learner.model.transitions.weight.grad.abs()
            .sum(dim=1)
            .nonzero()
            .flatten()
            .tolist()
        )
        self.assertEqual(sorted(changed), sorted(set([tokens[0][-3], 3])))

    def test_overlength_is_skipped_without_truncating_answer(self):
        learner = self.learner(max_seq_length=3)
        learner.observe("long prompt", "answer", 0.9)
        before = learner.model.transitions.weight.detach().clone()
        report = learner.update()
        self.assertFalse(report["accepted"])
        self.assertEqual(report["skipped"][0]["reason"], "overlength")
        self.assertTrue(torch.equal(before, learner.model.transitions.weight))

    def test_zero_weight_sft_does_not_count_as_an_update(self):
        learner = self.learner(point_weight=0, anchor_weight=1, success_threshold=0)
        learner.observe("p", "answer", 0, response_ids=[3])
        self.assertEqual(learner.update()["reason"], "no_reward_signal")
        self.assertEqual(learner.updates, 0)

    def test_drift_rejection_restores_weights_and_adam_state(self):
        learner = self.learner()
        learner.observe("p", "answer", 0.9, response_ids=[3])
        self.assertTrue(learner.update()["accepted"])
        weights = learner.model.transitions.weight.detach().clone()
        optimizer = copy.deepcopy(learner.optimizer.state_dict())
        learner.config = replace(learner.config, max_token_drift=1e-12)
        learner.observe("p", "answer", 0.0, response_ids=[3])
        report = learner.update()
        self.assertFalse(report["accepted"])
        self.assertEqual(report["reason"], "drift_limit")
        self.assertEqual(learner.updates, 1)
        self.assertEqual(report["optimizer_steps"], 1)
        self.assertEqual(report["retained_optimizer_steps"], 0)
        self.assertEqual(learner.optimizer_steps, 2)
        self.assertEqual(learner.retained_optimizer_steps, 1)
        self.assertTrue(all(x["delta"] == 0 for x in report["logprob_diagnostics"]))
        self.assertTrue(torch.equal(weights, learner.model.transitions.weight))
        restored = learner.optimizer.state_dict()
        self.assertEqual(optimizer["param_groups"], restored["param_groups"])
        for key, state in optimizer["state"].items():
            for name, value in state.items():
                self.assertTrue(torch.equal(value, restored["state"][key][name]))

    def test_backtracking_accepts_smaller_step_with_same_guard_and_data(self):
        learner = self.learner(learning_rate=0.08, max_token_drift=0.03, max_backtracks=3)
        item = learner.observe("p", "answer", 0.9, response_ids=[3])
        stats = copy.deepcopy(learner.buffer.stats)
        encoded = learner.encode(item)[0]
        before = float(learner.token_logps(encoded).detach().mean())
        calls = []
        original_logps = learner.token_logps

        def record_logps(tokens):
            calls.append((copy.deepcopy(tokens), torch.is_grad_enabled()))
            return original_logps(tokens)

        with patch.object(learner.buffer, "select", wraps=learner.buffer.select) as select, \
             patch.object(learner.buffer, "commit", wraps=learner.buffer.commit) as commit, \
             patch.object(learner, "token_logps", side_effect=record_logps):
            report = learner.update()
        self.assertTrue(report["accepted"])
        self.assertEqual(report["backtrack_count"], 3)
        self.assertEqual([x["learning_rate"] for x in report["backtrack_history"]],
                         [0.08, 0.04, 0.02, 0.01])
        self.assertTrue(all(x["reason"] == "drift_limit" for x in report["backtrack_history"][:-1]))
        self.assertLessEqual(report["max_token_rms_drift"], 0.03)
        self.assertEqual(learner.config.max_token_drift, 0.03)
        self.assertEqual(report["optimizer_steps"], 4)
        self.assertEqual(report["retained_optimizer_steps"], 1)
        self.assertEqual(learner.optimizer_steps, 4)
        self.assertEqual(learner.retained_optimizer_steps, 1)
        self.assertEqual(learner.attempts, 1)
        self.assertEqual(learner.updates, 1)
        self.assertEqual(select.call_count, 1)
        self.assertEqual(commit.call_count, 1)
        self.assertEqual(learner.buffer.stats, stats)
        self.assertEqual(learner.buffer.pending, [])
        self.assertEqual([x.uid for x in learner.buffer.replay].count(item.uid), 1)
        self.assertEqual(report["source_uids"], [item.uid])
        # One reference computation plus one drift check per trial; no refreshed
        # reference or new example encoding/selection is hidden in the retries.
        self.assertEqual(sum(not grad for _, grad in calls), 5)
        self.assertTrue(all(tokens == encoded for tokens, _ in calls))
        diagnostics = report["logprob_diagnostics"][0]
        after = float(learner.token_logps(encoded).detach().mean())
        self.assertAlmostEqual(diagnostics["mean_logprob_before"], before)
        self.assertAlmostEqual(diagnostics["mean_logprob_after"], after)
        self.assertGreater(diagnostics["delta"], 0)

        # The accepted state must equal one direct step at the final LR, rather
        # than a chain of failed steps with accidentally retained Adam moments.
        direct = self.learner(learning_rate=0.01, max_token_drift=0.03)
        direct.observe("p", "answer", 0.9, response_ids=[3])
        self.assertTrue(direct.update()["accepted"])
        self.assertTrue(torch.equal(learner.model.transitions.weight, direct.model.transitions.weight))
        for key, values in direct.optimizer.state_dict()["state"].items():
            for name, value in values.items():
                self.assertTrue(torch.equal(value, learner.optimizer.state_dict()["state"][key][name]))
        self.assertEqual(report["learning_rate_before"], 0.08)
        self.assertEqual(report["learning_rate_after"], 0.01)
        self.assertEqual(learner.state_dict()["optimizer_learning_rates"], [0.01])
        learner.observe("p", "answer", 0.9, response_ids=[3])
        next_report = learner.update()
        self.assertTrue(next_report["accepted"])
        self.assertEqual(next_report["backtrack_count"], 0)
        self.assertEqual(next_report["learning_rate_before"], 0.01)

    def test_exhausted_backtracking_restores_existing_adam_state_and_learning_rate(self):
        learner = self.learner(max_backtracks=3, epochs=2)
        learner.observe("p", "answer", 0.9, response_ids=[3])
        self.assertTrue(learner.update()["accepted"])
        weights = learner.model.transitions.weight.detach().clone()
        optimizer = copy.deepcopy(learner.optimizer.state_dict())
        retained_before = learner.retained_optimizer_steps
        steps_before = learner.optimizer_steps
        learner.config = replace(learner.config, max_token_drift=1e-12)
        item = learner.observe("p", "answer", 0.0, response_ids=[3])
        stats = copy.deepcopy(learner.buffer.stats)
        report = learner.update()
        self.assertFalse(report["accepted"])
        self.assertEqual(report["reason"], "drift_limit")
        self.assertEqual(report["backtrack_count"], 3)
        self.assertEqual(len(report["backtrack_history"]), 4)
        self.assertEqual(report["optimizer_steps"], 8)
        self.assertEqual(report["retained_optimizer_steps"], 0)
        self.assertEqual(learner.optimizer_steps - steps_before, 8)
        self.assertEqual(learner.retained_optimizer_steps, retained_before)
        self.assertTrue(torch.equal(weights, learner.model.transitions.weight))
        restored = learner.optimizer.state_dict()
        self.assertEqual(optimizer["param_groups"], restored["param_groups"])
        for key, state in optimizer["state"].items():
            for name, value in state.items():
                self.assertTrue(torch.equal(value, restored["state"][key][name]))
        self.assertEqual(report["learning_rate_before"], report["learning_rate_after"])
        self.assertEqual(learner.buffer.stats, stats)
        self.assertEqual([x.uid for x in learner.buffer.replay].count(item.uid), 1)
        self.assertEqual(learner.buffer.pending, [])
        self.assertTrue(all(row["delta"] == 0 for row in report["logprob_diagnostics"]))

    def test_real_tiny_transformer_lora_update_and_adapter_roundtrip(self):
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(42)
        config = LlamaConfig(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
        base = LlamaForCausalLM(config)
        original = copy.deepcopy(base.state_dict())
        model = get_peft_model(
            base,
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            ),
        )
        frozen = {
            k: v.detach().clone()
            for k, v in model.named_parameters()
            if not v.requires_grad
        }
        learner = RewardLearner(
            model,
            Tokenizer(),
            RewardConfig(
                learning_rate=0.01,
                replay_size=0,
                warmup=1,
                max_token_drift=10,
                anchor_weight=0,
            ),
        )
        learner.observe("p", "a", 0.4, response_ids=[3])
        learner.observe("p", "b", 0.9, response_ids=[4])
        report = learner.update()
        self.assertTrue(report["accepted"])
        self.assertTrue(
            any(p.abs().sum() > 0 for k, p in model.named_parameters() if "lora_B" in k)
        )
        self.assertTrue(
            all(
                torch.equal(frozen[k], p)
                for k, p in model.named_parameters()
                if k in frozen
            )
        )
        inputs = torch.tensor([[2, 3, 4]])
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            restored_base = LlamaForCausalLM(config)
            restored_base.load_state_dict(original)
            restored = PeftModel.from_pretrained(restored_base, tmp).eval()
            with torch.no_grad():
                self.assertTrue(
                    torch.allclose(model(inputs).logits, restored(inputs).logits)
                )


class FakeMemory:
    instances = []

    def __init__(self, args):
        self.updates = 0
        self.answers = 0
        self.feedback = []
        self.__class__.instances.append(self)

    def respond(self, query):
        self.answers += 1
        return {
            "prompt": query.prompt,
            "response": "bad JSON" if self.answers == 2 else '{"transmitters": []}',
            "response_ids": [3],
            "prompt_is_rendered": True,
        }

    def observe(self, completion, reward, valid):
        assert set(completion) == {
            "prompt",
            "response",
            "response_ids",
            "prompt_is_rendered",
        }
        assert isinstance(reward, float) and isinstance(valid, bool)
        self.feedback.append((reward, valid))
        return {"reward": reward}

    def adapt(self, output):
        assert len(self.feedback) == self.answers
        self.updates += 1
        return {"accepted": True, "reason": "accepted", "update": self.updates}

    def audit(self, output):
        pass


class ProtocolTest(unittest.TestCase):
    def test_offline_entry_loads_local_model_and_flushes_partial_window(self):
        from tokenizers import Tokenizer as RawTokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

        from ttcl.ramp.run_reward_benchmark import build_parser, validate_args
        from ttcl.ramp.train_reward_memory import run as train

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model"
            vocab = {
                word: i
                for i, word in enumerate(
                    [
                        "[UNK]",
                        "[PAD]",
                        "[EOS]",
                        "[BOS]",
                        "user",
                        "assistant",
                        "p",
                        "a",
                        "b",
                    ]
                )
            }
            raw = RawTokenizer(WordLevel(vocab, unk_token="[UNK]"))
            raw.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=raw,
                unk_token="[UNK]",
                pad_token="[PAD]",
                eos_token="[EOS]",
                bos_token="[BOS]",
                chat_template="{{ bos_token }}{% for message in messages %}"
                "{{ message['role'] }} {{ message['content'] }} "
                "{% endfor %}{% if add_generation_prompt %}assistant {% endif %}",
            )
            tokenizer.save_pretrained(model_path)
            LlamaForCausalLM(
                LlamaConfig(
                    vocab_size=len(vocab),
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    max_position_embeddings=64,
                )
            ).save_pretrained(model_path)
            input_path = root / "data.jsonl"
            input_path.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "prompt": "p",
                            "response": "a" if i % 2 else "b",
                            "reward": reward,
                        }
                    )
                    for i, reward in enumerate([0.4, 0.4, 0.9, 0.1])
                )
            )
            parser = build_parser()
            parser.add_argument("--input")
            args = parser.parse_args(
                [
                    "--model",
                    str(model_path),
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(root / "out"),
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                    "--update-every",
                    "3",
                    "--learning-rate",
                    "0.01",
                    "--max-token-drift",
                    "10",
                ]
            )
            validate_args(parser, args)
            result = train(args)
            self.assertEqual(result["examples"], 4)
            self.assertEqual(result["update_attempts"], 2)
            self.assertEqual(result["num_updates"], 2)
            self.assertTrue(
                (root / "out/latest_adapter/adapter_model.safetensors").exists()
            )
            state = json.loads((root / "out/learner_state.json").read_text())
            self.assertEqual(state["pending"], [])

    def test_score_precedes_update_no_final_update_and_invalid_is_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args(
                [
                    "--model",
                    "fake",
                    "--output-dir",
                    tmp,
                    "--num-scans",
                    "3",
                    "--update-every",
                    "1",
                ]
            )
            result = run(args, memory_factory=FakeMemory)
            records = [
                json.loads(x)
                for x in (Path(tmp) / "responses.jsonl").read_text().splitlines()
            ]
            self.assertEqual([x["updates_before_answer"] for x in records], [0, 1, 2])
            self.assertEqual(records[1]["reward"], 0.0)
            self.assertEqual(result["num_updates"], 2)
            self.assertEqual(FakeMemory.instances[-1].feedback[1], (0.0, False))
            with self.assertRaises(ValueError):
                run(args, memory_factory=FakeMemory)

    def test_frozen_receives_no_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args(
                [
                    "--model",
                    "fake",
                    "--output-dir",
                    tmp,
                    "--num-scans",
                    "2",
                    "--method",
                    "frozen",
                ]
            )
            result = run(args, memory_factory=FakeMemory)
            self.assertEqual(result["num_updates"], 0)
            self.assertEqual(FakeMemory.instances[-1].feedback, [])
            self.assertFalse((Path(tmp) / "experiences.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
