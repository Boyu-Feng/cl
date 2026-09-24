"""Focused contract tests for evidence isolation, caching, and reflection budgets."""
from pathlib import Path
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from ttcl.experience_evolution.core import WRITER_SYSTEM
from .methods import (
    REFLECTION_HEADER, cached_generation, delta_update, pack_reflections,
    public_episode, reflection_prompt, reflexion_update, trajectory_text,
)


class MethodsTests(unittest.TestCase):
    def episode(self, reward=0.0):
        return {"status": "complete", "initial_observation": "Your task is to: put a mug on a shelf.",
                "trajectory": [{"action": "look", "observation": "You see a mug.",
                                "hidden_goal": "SECRET"}],
                "reward": reward, "steps": 1, "future_task": "SECRET",
                "game": "SECRET", "generations": ["SECRET"], "expert_actions": ["SECRET"]}

    def client(self):
        return SimpleNamespace(complete=Mock(return_value={
            "raw_response": "Inspect before acting.", "input_tokens": 123,
            "output_tokens": 9, "seconds": 0.25, "served_model": "frozen-actor",
            "finish_reason": "stop"}))

    def test_public_trace_removes_nested_nonpublic_fields(self):
        public = public_episode(self.episode())
        self.assertEqual(set(public), {"initial_observation", "trajectory", "reward", "steps"})
        self.assertEqual(set(public["trajectory"][0]), {"action", "observation"})
        self.assertNotIn("SECRET", trajectory_text(self.episode()))
        for change in ({"status": "running"}, {"steps": 3}, {"reward": float("nan")}):
            with self.assertRaises(ValueError):
                public_episode(dict(self.episode(), **change))

    def test_cache_reuses_call_but_rejects_changed_prompt_model_seed_or_budget(self):
        client = self.client()
        messages = [{"role": "user", "content": "past evidence"}]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "generation.json"
            first = cached_generation(client, messages, path, 42)
            cached = cached_generation(client, messages, path, 42)
            self.assertFalse(first["cache_hit"])
            self.assertTrue(cached["cache_hit"])
            self.assertEqual(cached["input_tokens"], 123)
            self.assertEqual(cached["seconds"], 0.25)
            client.complete.assert_called_once_with(messages, model="frozen-actor",
                random_seed=42, tokens=768, temperature=0.0, top_p=1.0)
            for changed in (
                {"messages": [{"role": "user", "content": "new evidence"}]},
                {"model": "delta"}, {"seed": 43}, {"tokens": 256},
            ):
                args = dict(client=client, messages=messages, path=path, seed=42)
                args.update(changed)
                with self.assertRaises(ValueError):
                    cached_generation(**args)
            self.assertEqual(client.complete.call_count, 1)

    def test_delta_uses_original_prompt_and_requested_adapter(self):
        client = self.client()
        client.complete.return_value["served_model"] = "delta"
        with tempfile.TemporaryDirectory() as temp:
            result = delta_update(client, "old knowledge", self.episode(),
                                  Path(temp) / "delta.json", 7, "delta")
        messages = result["messages"]
        self.assertEqual(messages[0]["content"], WRITER_SYSTEM)
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["previous_experience"], "old knowledge")
        self.assertEqual(payload["completed_interaction"], public_episode(self.episode()))
        self.assertEqual(result["request"]["model"], "delta")
        self.assertNotIn("SECRET", str(messages))

    def test_failed_reflection_uses_pinned_official_builder(self):
        root = Path(__file__).resolve().parents[2] / "current_work"
        old = ["old plan 0", "old plan 1", "old plan 2", "old plan 3"]
        prompt = reflection_prompt(root, self.episode(), old)
        self.assertIn("Devise a concise, new plan of action", prompt)
        self.assertNotIn("old plan 0", prompt)
        self.assertIn("old plan 3", prompt)
        self.assertIn('"reward": 0.0', prompt)
        self.assertNotIn("SECRET", prompt)
        client = self.client()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "reflection.json"
            result = reflexion_update(client, self.episode(), old, path, 9, root)
            self.assertEqual(result["messages"], [{"role": "user", "content": prompt}])
            with self.assertRaises(ValueError):
                reflexion_update(client, self.episode(1), old, path, 9, root)

    def test_pack_reflections_keeps_whole_newest_plans(self):
        tokenizer = SimpleNamespace(encode=lambda text, **kwargs: text.split())
        old = "An older complete plan with extra detail."
        plans = [old, "Newer complete plan.", "Newest complete plan."]
        desired = REFLECTION_HEADER + "\n\nTrial 2:\n" + plans[1] + "\n\nTrial 3:\n" + plans[2]
        budget = len(desired.split())
        context, audit = pack_reflections(plans, tokenizer, budget)
        self.assertEqual(context, desired)
        self.assertEqual(audit["selected_indices"], [1, 2])
        self.assertEqual(audit["dropped"], [{"index": 0, "reason": "whole_plan_token_budget"}])
        self.assertLessEqual(audit["tokens"], budget)
        self.assertFalse(audit["truncated"])
        with self.assertRaises(ValueError):
            pack_reflections(["one " * 100], tokenizer, 20)

    def test_pack_empty_and_official_last_three(self):
        tokenizer = SimpleNamespace(encode=lambda text, **kwargs: text.split())
        context, audit = pack_reflections([], tokenizer)
        self.assertEqual(context, "")
        self.assertEqual(audit["tokens"], 0)
        context, audit = pack_reflections(["p0", "p1", "p2", "p3"], tokenizer)
        self.assertNotIn("p0", context)
        self.assertEqual(audit["selected_indices"], [1, 2, 3])
        self.assertEqual(audit["dropped"], [{"index": 0, "reason": "last_three_limit"}])


if __name__ == "__main__":
    unittest.main()
