"""Checks for leakage, first-task injection, and reward selection failure modes."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from src.interface import Query, Observation
from pydantic import BaseModel
from ttcl.structured_memory import run_benchmark as base
from ttcl.structured_memory.test_runner import FrozenFakeModel
from ttcl.structured_memory.verified_experience import (
    paired,
    select_candidate,
    public_evidence,
)


def trial_rows(differences):
    rows = []
    for seed in [42, 43]:
        for index, difference in enumerate(differences):
            for arm in [
                "none",
                "generic",
                "irrelevant",
                "candidate_a",
                "candidate_b",
                "combined",
            ]:
                rows.append(
                    {
                        "arm": arm,
                        "canonical_index": index,
                        "generation_seed": seed,
                        "status": "complete",
                        "instance_id": str(index),
                        "reward": difference if arm == "candidate_a" else 0,
                    }
                )
    return rows


class VerifiedExperienceTests(unittest.TestCase):
    def test_selects_replicated_gain_and_rejects_one_outlier(self):
        self.assertEqual(
            select_candidate(trial_rows([1, 1, 1, 1]), range(4), [42, 43])["selected"],
            "candidate_a",
        )
        self.assertIsNone(
            select_candidate(trial_rows([10, -1, -1, -1]), range(4), [42, 43])[
                "selected"
            ]
        )

    def test_missing_pair_disqualifies_instead_of_changing_denominator(self):
        rows = trial_rows([1, 1, 1, 1])
        rows.pop(0)
        self.assertIsNone(paired(rows, "candidate_a", "none", range(4), [42, 43]))
        self.assertIsNone(select_candidate(rows, range(4), [42, 43])["selected"])

    def test_repeated_seeds_are_not_counted_as_independent_instances(self):
        contrast = paired(
            trial_rows([1, 2, 3, 4]), "candidate_a", "none", range(4), [42, 43]
        )
        self.assertEqual(contrast["n_instances"], 4)
        self.assertEqual(len(contrast["pairs"]), 4)
        self.assertEqual(contrast["mean_delta"], 2.5)

    def test_evidence_excludes_terminal_feedback_and_generated_thought(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "task/independent/independent"
            directory.mkdir(parents=True)
            response = {
                "instance_id": "a",
                "turn": 1,
                "action": {
                    "thought": "unverified claim",
                    "tool_call": {"tool": "inspect"},
                },
            }
            observation = {
                "instance_id": "a",
                "turn": 1,
                "instance_complete": False,
                "content": "observed value",
            }
            (directory / "responses.jsonl").write_text(json.dumps(response) + "\n")
            (directory / "public_observations.jsonl").write_text(
                json.dumps(observation) + "\n"
            )
            evidence = public_evidence(root, "task", "a", 1)
            self.assertNotIn("unverified claim", json.dumps(evidence))
            observation["instance_complete"] = True
            (directory / "public_observations.jsonl").write_text(
                json.dumps(observation) + "\n"
            )
            with self.assertRaises(ValueError):
                public_evidence(root, "task", "a", 1)

    def test_frozen_experience_is_present_on_first_task_and_survives_reset(self):
        class Action(BaseModel):
            command: str

        class System(base.StructuredSystem):
            def experience_context(self, prompt):
                return "Frozen evidence from separate source instances"

        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(
                task="database_exploration",
                memory_chars=16000,
                seed=42,
                num_instances=30,
                max_turns_per_instance=64,
                allow_initial_experience=True,
            )
            model = FrozenFakeModel()
            system = System(args, "independent", model, Path(temporary), "")
            query = Query(
                prompt="A new question",
                response_schema=Action,
                instance_id="heldout",
                instance_index=12,
            )
            for _ in range(2):
                system.reset()
                system.respond(query)
                self.assertTrue(
                    any("Frozen evidence" in m["content"] for m in system.messages)
                )
                system.observe(
                    Observation(content="tool feedback", instance_complete=True)
                )
            events = [
                json.loads(line)
                for line in (Path(temporary) / "memory_contexts.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(events), 2)
            self.assertTrue(
                all(
                    e["memory_context"]
                    == "Frozen evidence from separate source instances"
                    for e in events
                )
            )


if __name__ == "__main__":
    unittest.main()
