import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from ttcl.llm_memory.memory import Episode, ExperienceMemory, answer_messages
from ttcl.llm_memory.run_benchmark import run_mode
from src.interface import Query
from src.tasks.blind_spectrum_monitoring.task import ScanReport


class MemoryTest(unittest.TestCase):
    def test_general_text_tasks_and_no_reward_when_disabled(self):
        memory = ExperienceMemory()
        seen = []

        def generate(messages, seed, **kwargs):
            data = json.loads(messages[-1]["content"])
            seen.append(data)
            self.assertNotIn("reward", data["completed_interactions"][0])
            return {"raw_response": f"Useful evidence through episode {len(seen)}.", "finish_reason": "stop"}

        for i, (task, response, feedback) in enumerate([
            ("Write Python that preserves input order.", "sorted(items)", "Incorrect: this changes the order."),
            ("Translate this sentence into French.", "Bonjour", "The greeting is correct."),
        ], 1):
            memory.update(Episode(i, str(i), task, response, feedback, 0.314159), generate, 1)
        self.assertEqual(seen[1]["previous_memory"], "Useful evidence through episode 1.")
        self.assertEqual(seen[1]["completed_interactions"][0]["task"], "Translate this sentence into French.")
        self.assertEqual(memory.through_episode, 2)
        self.assertIn(memory.text, answer_messages("Next arbitrary task", memory=memory.text)[0]["content"])
        with self.assertRaises(ValueError):
            memory.update(Episode(2, "2", "Duplicate", "", ""), generate, 1)

    def test_truncated_memory_keeps_previous_version_and_retries_pending_evidence(self):
        memory = ExperienceMemory(include_reward=True)
        attempts = []

        def generate(messages, seed, **kwargs):
            attempts.append(json.loads(messages[-1]["content"]))
            return {"raw_response": "First good memory" if len(attempts) == 1 else "Next memory",
                    "finish_reason": "length" if len(attempts) == 2 else "stop"}

        for i in range(1, 4):
            update = memory.update(Episode(i, str(i), f"Task {i}", "Answer", "Feedback", i / 10), generate, 1)
            if i == 2:
                self.assertFalse(update["accepted"])
                self.assertEqual(memory.text, "First good memory")
                self.assertEqual(memory.through_episode, 1)
        pending = attempts[2]["completed_interactions"]
        self.assertEqual([e["index"] for e in pending], [2, 3])
        self.assertEqual([e["reward"] for e in pending], [0.2, 0.3])
        self.assertEqual(memory.pending, [])
        self.assertEqual(memory.rejected_updates, 1)


class FakeTask:
    instances = []

    def __init__(self, **kwargs):
        self.count = kwargs["num_instances"]
        self.responses = []
        self.instances.append(self)

    def query(self, index):
        return Query(prompt=f"Public task {index}", response_schema=ScanReport,
                     instance_id=f"id-{index}", metadata={"private": "SECRET_LABEL"})

    def reset(self):
        return self.query(1)

    def step(self, response):
        self.responses.append(response)
        index = len(self.responses)
        return SimpleNamespace(
            observation=SimpleNamespace(content=f"Feedback {index}", metadata={"private": "SECRET_LABEL"}),
            instance_outcome=SimpleNamespace(reward=0.314159 + index / 100, metadata={"private": "SECRET_LABEL"}),
            done=index == self.count, next_query=self.query(index + 1))

    def evaluate(self):
        return SimpleNamespace(score=0.334159)


class FakeModel:
    def __init__(self, fail_memory=False):
        self.fail_memory = fail_memory

    def generate(self, messages, seed, **kwargs):
        is_memory = messages[0]["role"] == "system"
        if is_memory and self.fail_memory:
            raise RuntimeError("deliberate model failure")
        text = ('{"transmitters":[]}' if not is_memory else
                "Remember episode " + str(json.loads(messages[-1]["content"])["completed_interactions"][-1]["index"]))
        return {"raw_response": text, "input_tokens": 100, "output_tokens": 10,
                "context_limit": 262144, "finish_reason": "stop", "rendered_prompt_sha256": "fake"}


class RunnerTest(unittest.TestCase):
    def args(self, directory):
        return SimpleNamespace(output_dir=Path(directory), data_path=Path("unused"), num_scans=3,
                               seed=42, memory_max_new_tokens=1024)

    def test_causal_updates_allowlist_budget_and_reward_ablation(self):
        with tempfile.TemporaryDirectory() as directory:
            traces = {}
            for mode in ("independent", "summary", "summary_reward"):
                metrics = run_mode(self.args(directory), mode, FakeModel(), FakeTask)
                out = Path(directory) / mode
                requests = [json.loads(s) for s in (out / "requests.jsonl").read_text().splitlines()]
                traces[mode] = requests
                self.assertEqual(metrics["reward_calls"], 3)
                self.assertEqual(len(FakeTask.instances[-1].responses), 3)
                self.assertEqual(metrics["answer_calls"], 3)
                self.assertEqual(metrics["memory_calls"], 0 if mode == "independent" else 3)
                self.assertEqual(metrics["model_calls"], 3 if mode == "independent" else 6)
                for request in requests:
                    rendered = json.dumps(request["messages"])
                    self.assertNotIn("SECRET_LABEL", rendered)
                    if request["purpose"] == "memory":
                        data = json.loads(request["messages"][-1]["content"])
                        episode = data["completed_interactions"][-1]
                        self.assertEqual(episode["index"], request["scan"])
                        self.assertEqual("reward" in episode, mode == "summary_reward")
                        self.assertNotIn(f"Public task {request['scan'] + 1}", rendered)
                        self.assertEqual(request["options"]["temperature"], 0.0)
                    elif request["scan"] > 1 and mode != "independent":
                        self.assertIn(f"Remember episode {request['scan'] - 1}", rendered)
                        self.assertNotIn(f"Remember episode {request['scan']}", rendered)
            firsts = [traces[mode][0] for mode in traces]
            self.assertEqual(firsts[0], firsts[1])
            self.assertEqual(firsts[0], firsts[2])
            seeds = [[r["seed"] for r in requests if r["purpose"] == "answer"] for requests in traces.values()]
            self.assertEqual(seeds[0], seeds[1])
            self.assertEqual(seeds[0], seeds[2])

    def test_failed_memory_update_does_not_reanswer_or_rescore(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                run_mode(self.args(directory), "summary", FakeModel(fail_memory=True), FakeTask)
            out = Path(directory) / "summary"
            self.assertEqual(len(FakeTask.instances[-1].responses), 1)
            self.assertFalse((out / "metrics.json").exists())
            failure = json.loads((out / "failure.json").read_text())
            self.assertEqual(failure["phase"], "memory_update")
            self.assertEqual(failure["completed_answers"], 1)
            self.assertEqual(failure["attempted_model_calls"], 2)


if __name__ == "__main__":
    unittest.main()
