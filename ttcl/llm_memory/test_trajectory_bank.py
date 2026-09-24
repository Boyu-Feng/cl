"""Protocol tests using synthetic trajectories, not benchmark answer fixtures."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from pydantic import BaseModel
from src.interface import Query, Observation
from ttcl.llm_memory.trajectory_bank import TrajectoryBank
from ttcl.structured_memory.online_bank import BankSystem, paired
from ttcl.structured_memory.test_runner import FrozenFakeModel


def episode(number=1):
    return {
        "episode": number,
        "reward": -0.2,
        "completed": True,
        "steps": [
            {
                "step": 1,
                "action": {"tool": "inspect"},
                "public_feedback": "field z exists",
            },
            {"step": 2, "action": {"tool": "answer"}, "public_feedback": "not correct"},
        ],
    }


def update(decision="UPDATE", evidence_episode=1):
    return {
        "trajectory_summary": "Inspected a field, then failed to answer.",
        "reward_interpretation": "Negative episode outcome does not invalidate the inspected field.",
        "decision": decision,
        "operations": []
        if decision == "KEEP"
        else [
            {
                "op": "ADD",
                "id": "E1",
                "reason": "Reusable schema observation",
                "entry": {
                    "type": "fact",
                    "title": "Field z",
                    "scope": "Same test database",
                    "lesson": "Field z exists",
                    "application": "Use the observed name",
                    "limitations": "Recheck after schema change",
                    "evidence": [{"episode": evidence_episode, "steps": [1]}],
                },
            }
        ],
    }


def generator(value):
    def generate(messages, seed, **kwargs):
        return {
            "raw_response": json.dumps(value),
            "finish_reason": "stop",
            "input_tokens": 10,
            "output_tokens": 3,
        }

    return generate


class TrajectoryBankTests(unittest.TestCase):
    def test_operation_alias_repair_preserves_all_model_content(self):
        proposal = update()
        proposal["decision"] = "ADD"
        bank = TrajectoryBank()
        result = bank.update(episode(), generator(proposal), 100, len)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["parsed_update"]["operations"], proposal["operations"])
        self.assertEqual(result["attempts"][0]["packaging_repair"]["from"], "ADD")

    def test_keep_is_an_explicit_model_decision_and_preserves_bank(self):
        bank = TrajectoryBank()
        first = bank.update(episode(), generator(update()), 100, len)
        self.assertTrue(first["accepted"])
        previous = copy.deepcopy(bank.entries)
        second = bank.update(episode(2), generator(update("KEEP")), 200, len)
        self.assertEqual(second["decision"], "KEEP")
        self.assertEqual(bank.entries, previous)
        self.assertEqual((bank.last_observed, bank.version), (2, 1))

    def test_future_evidence_is_rejected_without_partial_mutation(self):
        bank = TrajectoryBank()
        proposal = update()
        bad = copy.deepcopy(proposal["operations"][0])
        bad["id"] = "E2"
        bad["entry"]["evidence"] = [{"episode": 9, "steps": [1]}]
        proposal["operations"].append(bad)
        result = bank.update(episode(), generator(proposal), 100, len)
        self.assertFalse(result["accepted"])
        self.assertEqual(bank.entries, [])
        self.assertEqual(len(result["attempts"]), 2)

    def test_revision_and_removal_are_model_controlled(self):
        bank = TrajectoryBank()
        bank.update(episode(), generator(update()), 100, len)
        revision = update(evidence_episode=2)
        revision["operations"][0]["op"] = "REVISE"
        revision["operations"][0]["entry"]["lesson"] = "Field z changed type"
        result = bank.update(episode(2), generator(revision), 200, len)
        self.assertTrue(result["accepted"])
        removal = update()
        removal["operations"] = [
            {"op": "REMOVE", "id": "E1", "reason": "Observation is stale"}
        ]
        result = bank.update(episode(3), generator(removal), 300, len)
        self.assertTrue(result["accepted"])
        self.assertEqual(bank.entries, [])

    def test_reward_is_preserved_and_incomplete_reward_is_not_fabricated(self):
        bank = TrajectoryBank()
        payload = bank.payload(episode())
        self.assertEqual(payload["trajectory"]["reward"], -0.2)
        incomplete = episode()
        incomplete["completed"] = False
        with self.assertRaises(ValueError):
            bank.payload(incomplete)
        incomplete["reward"] = None
        self.assertIsNone(bank.payload(incomplete)["trajectory"]["reward"])

    def test_truncated_summary_is_not_promoted(self):
        bank = TrajectoryBank()

        def generate(*args, **kwargs):
            return {"raw_response": json.dumps(update()), "finish_reason": "length"}

        result = bank.update(episode(), generate, 100, len)
        self.assertFalse(result["accepted"])
        self.assertEqual(bank.entries, [])

    def test_context_budget_does_not_silently_truncate(self):
        bank = TrajectoryBank(max_tokens=3)
        result = bank.update(episode(), generator(update()), 100, len)
        self.assertFalse(result["accepted"])
        self.assertEqual(bank.entries, [])

    def test_capture_keeps_all_operations_but_no_metadata_or_future_query(self):
        class Action(BaseModel):
            command: str

        args = SimpleNamespace(
            task="database_exploration",
            seed=42,
            memory_chars=16000,
            num_instances=12,
            max_turns_per_instance=64,
            allow_initial_experience=True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            system = BankSystem(
                args, FrozenFakeModel(), Path(temporary), "public brief", "old bank"
            )
            for index in range(2):
                query = Query(
                    prompt=f"visible query {index}",
                    response_schema=Action,
                    instance_id="SECRET_ID",
                    instance_index=0,
                    metadata={"hidden": "SECRET_QUERY"},
                )
                system.respond(query)
                system.observe(
                    Observation(
                        content=f"result {index}",
                        instance_complete=index == 1,
                        metadata={"hidden": "SECRET_FEEDBACK"},
                    ),
                    next_query=Query(prompt="SECRET_FUTURE", response_schema=Action),
                )
            text = json.dumps(system.public_steps)
            self.assertEqual(len(system.public_steps), 2)
            self.assertEqual(system.public_steps[-1]["public_feedback"], "result 1")
            self.assertNotIn("SECRET", text)
            self.assertEqual(len(system.public_schemas), 1)
            self.assertTrue(any("old bank" in m["content"] for m in system.messages))

    def test_failed_instances_are_not_zero_scored_or_full_comparisons(self):
        rows = [
            {
                "arm": "independent",
                "canonical_index": 0,
                "instance_id": "a",
                "status": "complete",
                "reward": 1,
            },
            {
                "arm": "online_bank",
                "canonical_index": 0,
                "instance_id": "a",
                "status": "failed",
                "reward": None,
            },
        ]
        result = paired(rows, 1)
        self.assertFalse(result["complete_comparison"])
        self.assertIsNone(result["mean_delta"])


if __name__ == "__main__":
    unittest.main()
