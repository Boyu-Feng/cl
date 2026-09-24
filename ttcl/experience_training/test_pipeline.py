"""Exercise writer/reader separation without expensive model inference."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ttcl.experience_training import run


class FakeBackend:
    def __init__(self, args, adapter=None):
        self.repeat = 0
        self.writer = False

    def count(self, text):
        return len(text) // 4

    def generate(self, messages, seed, **kwargs):
        assert self.writer
        payload = json.loads(messages[1]["content"])
        assert payload["trajectory"]["episode"] == 1
        assert "NEXT_TASK_SENTINEL" not in str(messages)
        answer = copy.deepcopy(run.KEEP)
        return {
            "raw_response": json.dumps(answer),
            "finish_reason": "stop",
            "input_tokens": 10,
            "output_tokens": 20,
        }


def fake_episode(args, model, index, output, context):
    assert not model.writer
    assert index == 1
    assert context == ""
    output.mkdir(parents=True)
    row = {
        "status": "complete",
        "reward": 0.25,
        "instance_id": "NEXT_TASK_SENTINEL",
        "actor_calls": 1,
        "actor_input_tokens": 10,
        "actor_output_tokens": 20,
    }
    run.save(output / "metrics.json", row)
    return row, {}


class PipelineTests(unittest.TestCase):
    def test_same_next_instance_no_future_writer_access_and_ties_no_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run.save(
                root / "plan.json",
                {
                    "model": {},
                    "environment_seed": 42,
                    "source_indices": [0],
                    "candidate_temperatures": [0, 0.9],
                    "collection_repeats": [101, 202],
                    "writer_output_tokens": 100,
                    "bank": {},
                },
            )
            source = root / "data/test/episode_001"
            run.save(
                source / "bank_before.json",
                {"entries": [], "version": 0, "last_observed": 0},
            )
            run.save(
                source / "trajectory.json",
                {
                    "episode": 1,
                    "reward": 0,
                    "completed": True,
                    "steps": [
                        {
                            "step": 1,
                            "action": {"answer": "past"},
                            "public_feedback": "past feedback",
                        }
                    ],
                },
            )
            with (
                patch.object(run, "Backend", FakeBackend),
                patch.object(run, "run_episode", fake_episode),
                patch.object(run.os, "chdir"),
            ):
                run.collect(root, "test")
            result = run.read(root / "collection/test/prefix_000/result.json")
            self.assertIsNone(result["selected"])
            self.assertEqual(result["probe_index"], 1)
            self.assertEqual(len(result["records"]), 6)
            summary = run.read(root / "collection/test/summary.json")
            self.assertEqual(summary["actor_calls"], 2)  # identical contexts reused
            self.assertEqual(summary["writer_calls"], 2)
            self.assertEqual(summary["positive_update_labels"], 0)


if __name__ == "__main__":
    unittest.main()
