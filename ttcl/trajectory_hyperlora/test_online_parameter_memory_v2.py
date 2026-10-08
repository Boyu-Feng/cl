"""Checks for bounded parameter memory and environment-time adapter selection."""

import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import run_episode
from ttcl.trajectory_hyperlora.online_parameter_memory_v2 import ParameterMemory


class ParameterMemoryTest(unittest.TestCase):
    def test_query_selects_relevant_persistent_factor_without_mutation(self):
        memory = ParameterMemory(temperature=.05, attention_mix=1)
        positive = [torch.ones(1, 2, 1)]
        negative = [-torch.ones(1, 2, 1)]
        memory.write(torch.tensor([1., 0.]), positive)
        before = memory.digest()
        memory.write(torch.tensor([0., 1.]), negative)
        a, audit_a = memory.read(torch.tensor([1., 0.]))
        b, audit_b = memory.read(torch.tensor([0., 1.]))
        self.assertGreater(float(a[0].mean()), .9)
        self.assertLess(float(b[0].mean()), -.9)
        self.assertEqual(audit_a["selected"], 0)
        self.assertEqual(audit_b["selected"], 1)
        self.assertNotEqual(before, memory.digest())
        self.assertTrue(torch.equal(positive[0], torch.ones_like(positive[0])))

    def test_capacity_merges_without_factor_scale_explosion(self):
        memory = ParameterMemory(capacity=1)
        memory.write(torch.tensor([1., 0.]), [torch.ones(1, 2, 1)])
        memory.write(torch.tensor([1., 0.]), [torch.full((1, 2, 1), 3.)])
        result, audit = memory.read(torch.tensor([1., 0.]))
        self.assertEqual(audit["entries"], 1)
        self.assertEqual(memory.entries[0].count, 2)
        self.assertTrue(torch.equal(result[0], torch.full((1, 2, 1), 2.)))

    def test_opposite_keys_do_not_destroy_a_full_slot(self):
        memory = ParameterMemory(capacity=1)
        memory.write(torch.tensor([1., 0.]), [torch.ones(1, 2, 1)])
        memory.write(torch.tensor([-1., 0.]), [torch.ones(1, 2, 1)])
        self.assertGreater(float(memory.entries[0].key.norm()), .99)

    def test_invalid_memory_state_is_rejected(self):
        memory = ParameterMemory()
        with self.assertRaises(ValueError):
            memory.write(torch.zeros(2), [torch.ones(1, 2, 1)])
        memory.write(torch.ones(2), [torch.ones(1, 2, 1)])
        with self.assertRaises(ValueError):
            memory.write(torch.ones(2), [torch.ones(1, 3, 1)])


class _Environment:
    def reset(self):
        return {"feedback": "new task", "admissible_commands": ["look"], "won": False}

    def step(self, command):
        return {"feedback": "done", "admissible_commands": [], "won": True}, 1., True

    def close(self):
        pass


class _Layer:
    def __init__(self):
        self.base = type("Base", (), {"out_features": 2})()
        self.rank = 1
        self.b = None


class _Agent:
    task_conditioned = False
    task_context_scope = "initial"

    def __init__(self):
        self.adapters = [_Layer()]

    def set_source(self, value, target_fields=None):
        if value is None:
            self.adapters[0].b = None


class SelectorTest(unittest.TestCase):
    def test_selector_sees_current_initial_observation_before_action(self):
        agent = _Agent()
        selected = []

        def selector(observation):
            selected.append(observation)
            return [torch.ones(1, 2, 1)]

        def generate(*_args, **_kwargs):
            self.assertTrue(torch.equal(agent.adapters[0].b,
                                        torch.ones(1, 2, 1)))
            return "look"

        with patch("ttcl.trajectory_hyperlora.alfworld_zero_shot_probe.make_env",
                   return_value=_Environment()), patch(
                   "ttcl.trajectory_hyperlora.alfworld_zero_shot_probe.generate",
                   side_effect=generate):
            episode = run_episode(agent, None, Path("unused"), {}, adapter=False,
                fixed_adapter_selector=selector, device="cpu", max_steps=1,
                max_new_tokens=8)
        self.assertEqual(episode["status"], "complete")
        self.assertEqual(selected, ["new task"])
        self.assertIsNone(agent.adapters[0].b)


if __name__ == "__main__":
    unittest.main()
