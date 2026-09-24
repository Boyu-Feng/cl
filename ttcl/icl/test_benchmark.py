"""Check complete causal history, matched controls, and explicit overflow failure."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from ttcl.icl.run_benchmark import ContextLimitError, check_context, run_mode
from src.interface import Query
from src.tasks.blind_spectrum_monitoring.task import ScanReport


class FakeTask:
    instances = []

    def __init__(self, **kwargs):
        self.count = kwargs["num_instances"]
        self.responses = []
        self.instances.append(self)

    def query(self, index):
        return Query(prompt=f"Original complete instructions and scan {index}",
                     response_schema=ScanReport, instance_id=f"scan-{index}")

    def reset(self):
        return self.query(1)

    def step(self, response):
        self.responses.append(response)
        index = len(self.responses)
        return SimpleNamespace(
            observation=SimpleNamespace(content=f"Public feedback {index}"),
            instance_outcome=SimpleNamespace(reward=0.987654321,
                                             metadata={"private": "HIDDEN_GROUND_TRUTH"}),
            done=index == self.count, next_query=self.query(index + 1))

    def evaluate(self):
        return SimpleNamespace(score=0.987654321)


class FakeModel:
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def generate(self, messages, seed):
        self.calls.append((copy.deepcopy(messages), seed))
        if len(self.calls) == self.fail_on:
            check_context(100, 20, 119)
        return {"raw_response": "invalid original answer" if len(self.calls) == 1 else '{"transmitters":[]}',
                "input_tokens": 100 * len(messages), "output_tokens": 5,
                "context_limit": 262144, "rendered_prompt_sha256": "fake"}


class ProtocolTest(unittest.TestCase):
    def args(self, directory):
        return SimpleNamespace(output_dir=Path(directory), data_path=Path("unused"),
                               num_scans=3, seed=42)

    def test_complete_history_and_matched_independent_control(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            models = {}
            for mode in ("full_history", "independent"):
                model = models[mode] = FakeModel()
                metrics = run_mode(args, mode, model, FakeTask)
                task = FakeTask.instances[-1]
                self.assertEqual(len(task.responses), 3)
                self.assertEqual(metrics["model_calls"], 3)
                self.assertEqual(metrics["reward_calls"], 3)
                self.assertEqual(metrics["invalid_reports"], 1)
                self.assertEqual(metrics["parameter_updates"], 0)
                self.assertFalse(metrics["history_truncated"])
                self.assertTrue(task.responses[0].metadata["latency_timeout"])
                out = Path(directory) / mode
                journal = [json.loads(line) for line in (out / "messages.jsonl").read_text().splitlines()]
                records = [json.loads(line) for line in (out / "responses.jsonl").read_text().splitlines()]
                for index, ((messages, _), record) in enumerate(zip(model.calls, records)):
                    reconstructed = [{"role": m["role"], "content": m["content"]}
                                     for m in journal[record["message_start"]:record["message_end"]]]
                    self.assertEqual(messages, reconstructed)
                    self.assertEqual(len(messages), 3 * index + 1 if mode == "full_history" else 1)
                    rendered = json.dumps(messages)
                    self.assertNotIn("0.987654321", rendered)
                    self.assertNotIn("HIDDEN_GROUND_TRUTH", rendered)
                    self.assertNotIn(f"Public feedback {index + 1}", rendered)
                    self.assertNotIn(f"instructions and scan {index + 2}", rendered)
                    self.assertIn("JSON object", messages[-1]["content"])
                if mode == "full_history":
                    last = model.calls[-1][0]
                    self.assertEqual(last[:1], model.calls[0][0])
                    self.assertEqual(last[1], {"role": "assistant", "content": "invalid original answer"})
                    self.assertEqual(last[2], {"role": "user", "content": "FEEDBACK: Public feedback 1"})
                self.assertEqual(json.loads((out / "metrics.json").read_text()), metrics)
            self.assertEqual([seed for _, seed in models["full_history"].calls],
                             [seed for _, seed in models["independent"].calls])
            self.assertEqual(models["full_history"].calls[0], models["independent"].calls[0])

    def test_overflow_stops_without_scoring_or_truncating_failed_query(self):
        check_context(100, 20, 120)
        with tempfile.TemporaryDirectory() as directory:
            model = FakeModel(fail_on=2)
            with self.assertRaises(ContextLimitError):
                run_mode(self.args(directory), "full_history", model, FakeTask)
            out = Path(directory) / "full_history"
            self.assertEqual(len(FakeTask.instances[-1].responses), 1)
            self.assertEqual(len(model.calls[-1][0]), 4)
            self.assertFalse((out / "metrics.json").exists())
            failure = json.loads((out / "failure.json").read_text())
            self.assertEqual(failure["scan"], 2)
            self.assertEqual(failure["completed"], 1)
            self.assertFalse(failure["history_truncated"])
            self.assertIn("No history was truncated", failure["error"])


if __name__ == "__main__":
    unittest.main()
