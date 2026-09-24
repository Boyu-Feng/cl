"""Protocol checks with a frozen fake backend; no GPU or task data needed."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pydantic import BaseModel

from ttcl.structured_memory.run_benchmark import (
    MEMORIES,
    Recorder,
    StructuredSystem,
    comparison,
    generation_seed,
    run_mode,
)
from src.interface import InstanceOutcome, Observation, Query


class Action(BaseModel):
    command: str


class FrozenFakeModel:
    def __init__(self):
        self.requests = []

    def generate(self, messages, seed):
        self.requests.append((copy.deepcopy(messages), seed))
        return {
            "raw_response": '{"command":"inspect"}',
            "input_tokens": 10,
            "output_tokens": 3,
            "finish_reason": "stop",
        }


class PublicMemory:
    def __init__(self, max_chars):
        self.events = []

    def observe(self, query, action, observation, *, instance_id, instance_complete):
        self.events.append(
            {
                "query": query,
                "action": action,
                "observation": observation,
                "instance_id": instance_id,
                "instance_complete": instance_complete,
            }
        )

    def context(self, query=""):
        return json.dumps(
            {"observed_facts": [item["observation"] for item in self.events]}
        )

    def state_dict(self):
        return copy.deepcopy(self.events)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.args = SimpleNamespace(
            task="test_public",
            memory_chars=16000,
            seed=42,
            max_turns_per_instance=10,
            num_instances=2,
            output_dir=self.output,
        )
        self.model = FrozenFakeModel()
        self.patcher = patch.dict(MEMORIES, test_public=PublicMemory)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    @staticmethod
    def query(prompt, identity, index=0):
        return Query(
            prompt=prompt,
            response_schema=Action,
            instance_id=identity,
            instance_index=index,
            metadata={"opponent_policy": "SECRET_QUERY_METADATA"},
        )

    def system(self, mode="structured"):
        return StructuredSystem(
            self.args, mode, self.model, self.output, "Public task brief"
        )

    def test_hidden_metadata_and_canonical_policy_id_never_enter_context(self):
        system = self.system()
        canonical = "task:SECRET_CANONICAL_POLICY:0"
        system.respond(self.query("Visible request", canonical))
        system.observe(
            Observation(
                content="Public result",
                instance_complete=True,
                metadata={"answer": "SECRET_OBSERVATION_METADATA", "reward": 12345},
            )
        )
        system.respond(self.query("Next visible request", "task:next", 1))
        contexts = json.dumps(self.model.requests)
        events = json.dumps(system.memory.state_dict())
        for forbidden in (
            "SECRET_QUERY_METADATA",
            "SECRET_OBSERVATION_METADATA",
            "SECRET_CANONICAL_POLICY",
            "12345",
        ):
            self.assertNotIn(forbidden, contexts)
            self.assertNotIn(forbidden, events)
        self.assertEqual(system.memory.events[0]["instance_id"], "experience_1")
        self.assertIn("Public result", contexts)
        # Canonical identity still deterministically seeds/logs the episode.
        self.assertEqual(self.model.requests[0][1], generation_seed(42, canonical, 1))
        records = [
            json.loads(line)
            for line in (self.output / "responses.jsonl").read_text().splitlines()
        ]
        self.assertEqual(records[0]["instance_id"], canonical)

    def test_boundary_resets_transcript_but_keeps_structured_public_experience(self):
        system = self.system()
        system.respond(self.query("OLD_RAW_QUERY_SENTINEL", "task:a"))
        system.observe(
            Observation(content="Remembered public observation", instance_complete=True)
        )
        system.respond(self.query("New task", "task:b", 1))
        messages = self.model.requests[-1][0]
        flattened = json.dumps(messages)
        self.assertNotIn("OLD_RAW_QUERY_SENTINEL", flattened)
        self.assertIn("Remembered public observation", flattened)
        self.assertFalse(any(message["role"] == "assistant" for message in messages))
        self.assertEqual(system.turn, 1)
        self.assertEqual(system.episodes_seen, 2)

    def test_within_episode_tool_feedback_is_retained_with_raw_interaction(self):
        system = self.system()
        system.respond(self.query("First question", "task:a"))
        system.observe(
            Observation(content="Column does not exist", instance_complete=False)
        )
        system.respond(self.query("Try another query", "task:a"))
        messages = self.model.requests[-1][0]
        flattened = json.dumps(messages)
        self.assertIn("First question", flattened)
        self.assertIn("Tool feedback:\\nColumn does not exist", flattened)
        self.assertTrue(any(message["role"] == "assistant" for message in messages))
        self.assertEqual(system.turn, 2)
        self.assertEqual(system.episodes_seen, 1)

    def test_independent_reset_drops_memory_and_transcript_but_keeps_usage_totals(self):
        system = self.system("independent")
        system.respond(self.query("Prior question", "task:a"))
        system.observe(
            Observation(content="PRIOR_EVIDENCE_SENTINEL", instance_complete=True)
        )
        self.assertEqual(system.memory.state_dict(), [])
        system.reset()
        system.respond(self.query("New question", "task:b", 1))
        messages = self.model.requests[-1][0]
        self.assertNotIn("Prior question", json.dumps(messages))
        self.assertNotIn("PRIOR_EVIDENCE_SENTINEL", json.dumps(messages))
        self.assertEqual(system.episodes_seen, 1)
        self.assertEqual(system.calls, 2)

    def test_outcome_metadata_is_logged_separately_from_writer_and_model(self):
        system = self.system()
        system.respond(self.query("Public question", "task:a"))
        recorder = Recorder(self.output, system, 1)
        recorder.sync_instance_outcomes(
            [
                InstanceOutcome(
                    instance_id="task:a",
                    instance_index=0,
                    reward=0.7,
                    metadata={"ground_truth": "SECRET_EVALUATOR_METADATA"},
                )
            ]
        )
        self.assertNotIn("SECRET_EVALUATOR_METADATA", json.dumps(self.model.requests))
        self.assertEqual(system.memory.state_dict(), [])
        self.assertIn(
            "SECRET_EVALUATOR_METADATA", (self.output / "progress.json").read_text()
        )

    def test_pairing_uses_canonical_ids_and_rejects_mismatches(self):
        result = comparison(
            {
                "independent": {
                    "outcomes": [
                        {"instance_id": "a", "reward": 0.2},
                        {"instance_id": "b", "reward": 0.6},
                    ]
                },
                "structured": {
                    "outcomes": [
                        {"instance_id": "b", "reward": 0.5},
                        {"instance_id": "a", "reward": 0.5},
                    ]
                },
            }
        )
        self.assertAlmostEqual(result["structured_minus_independent"], 0.1)
        self.assertEqual(
            (result["improved"], result["worse"], result["tied"]), (1, 1, 0)
        )
        with self.assertRaisesRegex(ValueError, "canonical instance"):
            comparison(
                {
                    "independent": {"outcomes": [{"instance_id": "a", "reward": 1}]},
                    "structured": {"outcomes": [{"instance_id": "b", "reward": 1}]},
                }
            )

    def test_display_ordinals_and_duplicate_brief_do_not_change_paired_inputs(self):
        self.args.num_instances = 12
        cases = [
            (
                "Question 1/1: inspect table",
                "Question 7/12: inspect table",
                "Question 7/12",
            ),
            ("Hand #1: choose action", "Hand #7: choose action", "Hand #7"),
            ("## Study 1/1: Example", "## Study 7/12: Example", "## Study 7/12"),
        ]
        for independent_prompt, structured_prompt, normalized in cases:
            for brief_on_independent in (False, True):
                with self.subTest(
                    normalized=normalized, brief_on_independent=brief_on_independent
                ):
                    independent = self.system("independent")
                    structured = self.system("structured")
                    independent.requested_canonical_index = 6
                    if brief_on_independent:
                        independent_prompt_with_brief = (
                            "Public task brief\n\n" + independent_prompt
                        )
                        structured_prompt_with_brief = structured_prompt
                    else:
                        independent_prompt_with_brief = independent_prompt
                        structured_prompt_with_brief = (
                            "Public task brief\n\n" + structured_prompt
                        )
                    # Isolated tasks may expose a local index zero; the requested
                    # canonical index overrides only its display ordinal.
                    independent.respond(
                        self.query(independent_prompt_with_brief, "canonical:same", 0)
                    )
                    independent_request = self.model.requests[-1]
                    structured.respond(
                        self.query(structured_prompt_with_brief, "canonical:same", 6)
                    )
                    structured_request = self.model.requests[-1]
                    self.assertEqual(independent_request, structured_request)
                    rendered = json.dumps(structured_request[0])
                    self.assertIn(normalized, rendered)
                    self.assertEqual(rendered.count("Public task brief"), 1)

    def test_independent_runner_uses_fresh_tasks_with_same_canonical_indices(self):
        constructed = []

        class Task:
            def __init__(task):
                task.selected = []
                constructed.append(task)

            def get_agent_brief(task):
                return None

            def reset_baseline_instance(task, index):
                task.selected.append(index)
                return self.query("Public task", f"canonical:{index}", index)

        def run(task, system, *, trace_recorder, initial_query, **kwargs):
            self.assertEqual(system.episodes_seen, 0)
            system.respond(initial_query)
            system.observe(Observation(content="Completed", instance_complete=True))
            outcome = InstanceOutcome(
                initial_query.instance_id, initial_query.instance_index, 0.5
            )
            trace_recorder.sync_instance_outcomes([outcome])
            return SimpleNamespace(instance_outcomes=[outcome])

        with (
            patch(
                "ttcl.structured_memory.run_benchmark.make_task",
                side_effect=lambda *args, **kwargs: Task(),
            ),
            patch("ttcl.structured_memory.run_benchmark.run_task", side_effect=run),
        ):
            metrics = run_mode(self.args, "independent", self.model)
        self.assertEqual([task.selected for task in constructed], [[], [0], [1]])
        self.assertEqual(metrics["completed_instances"], 2)
        self.assertEqual(
            [item["instance_id"] for item in metrics["outcomes"]],
            ["canonical:0", "canonical:1"],
        )
        self.assertEqual(
            [item["requested_canonical_index"] for item in metrics["outcomes"]], [0, 1]
        )


if __name__ == "__main__":
    unittest.main()
