"""CPU protocol tests for sampling, counterfactual budgets and causal memory."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from ttcl.ramp.run_reward_benchmark import RewardMemory, parse_args, public_id_mutations, run
from src.interface import Query
from src.tasks.blind_spectrum_monitoring.task import ScanReport


def report(center):
    return json.dumps({"transmitters": [{"center_freq": center, "bandwidth": 1.0,
                                       "currently_active": True, "estimated_power": -20.0}]})


class FakeTask:
    events = []
    live = None

    def __init__(self, num_instances, **kwargs):
        self.index = 0
        self.count = num_instances
        self.scores = []
        FakeTask.live = self
        FakeTask.events = []

    def query(self):
        return Query(prompt=f"public scan {self.index}", response_schema=ScanReport,
                     instance_id=f"scan-{self.index}")

    def reset(self):
        return self.query()

    def step(self, response):
        FakeTask.events.append(("score", self.index, self is FakeTask.live))
        score = (0.0 if response.metadata or not response.action.transmitters
                 else response.action.transmitters[0].center_freq / 100)
        self.scores.append(score)
        self.index += 1
        done = self.index == self.count
        return SimpleNamespace(instance_outcome=SimpleNamespace(reward=score),
                               done=done, next_query=None if done else self.query(),
                               observation="HIDDEN FEEDBACK MUST NEVER ENTER LEARNER")

    def evaluate(self):
        return SimpleNamespace(score=sum(self.scores) / len(self.scores))


class FakeMemory:
    latest = None
    invalid_first = False

    def __init__(self, args):
        self.args = args
        self.answers = 0
        self.updates = 0
        self.feedback = []
        self.groups = []
        FakeMemory.latest = self

    def respond(self, query):
        index = self.answers % self.args.candidates_per_scan
        scan = self.answers // self.args.candidates_per_scan
        FakeTask.events.append(("generate", scan, index))
        self.answers += 1
        text = "invalid" if self.invalid_first and index == 0 else report([10, 90, 50][index])
        return {"prompt": query.prompt, "response": text, "response_ids": [3, 4],
                "prompt_is_rendered": True}

    def observe(self, completion, reward, valid):
        assert set(completion) == {"prompt", "response", "response_ids", "prompt_is_rendered"}
        FakeTask.events.append(("feedback", FakeTask.live.index - 1, reward))
        self.feedback.append((reward, valid))
        return {"reward": reward}

    def observe_group(self, completions, rewards, valids):
        assert len({item["prompt"] for item in completions}) == 1
        self.groups.append(list(rewards))
        return [self.observe(item, reward, valid)
                for item, reward, valid in zip(completions, rewards, valids)]

    def adapt(self, output):
        FakeTask.events.append(("update", FakeTask.live.index - 1, None))
        self.updates += 1
        return {"accepted": True, "reason": "accepted"}

    def audit(self, output):
        pass


class SelectionTask(FakeTask):
    def query(self):
        shift = self.index * 2
        prompt = (f"--- Scan {self.index + 1}/{self.count} ---\n"
                  f"scan_id: scan-{self.index}\nDetected peaks:\n"
                  f"  - peak_id: p1 | freq: {10 + shift}.0 MHz | power: -20.0 dBm | width: 1.0 MHz\n"
                  f"  - peak_id: p2 | freq: {90 - shift}.0 MHz | power: -20.0 dBm | width: 1.0 MHz\n"
                  "Band: 0.0-100.0 MHz\n")
        return Query(prompt=prompt, response_schema=ScanReport, instance_id=f"scan-{self.index}")


class SelectionMemory(FakeMemory):
    def respond(self, query):
        completion = super().respond(query)
        index = (self.answers - 1) % self.args.candidates_per_scan
        action = [999] if self.invalid_first and index == 0 else [[1], [2], [1, 2]][index]
        completion["response"] = json.dumps({"include": action})
        return completion


class ProposalTokenizer:
    eos_token_id = 999

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]


class MutationMemory(FakeMemory):
    def __init__(self, args):
        super().__init__(args)
        self.tokenizer = ProposalTokenizer()
        self.calls = {}

    def respond(self, query):
        index = self.calls.get(query.instance_id, 0)
        self.calls[query.instance_id] = index + 1
        scan = int(query.instance_id.rsplit("-", 1)[1])
        FakeTask.events.append(("generate", scan, index))
        self.answers += 1
        included = [999] if self.invalid_first and index == 0 else [1, 2]
        return {"prompt": query.prompt, "response": json.dumps({"include": included}),
                "response_ids": [3, 4], "prompt_is_rendered": True}


class ProtocolTest(unittest.TestCase):
    def execute(self, root, *flags):
        data = root / "data.jsonl"
        data.write_text("{}\n{}\n")
        args = parse_args(["--model", "fake", "--output-dir", str(root / "output"),
                           "--data-path", str(data), "--num-scans", "2",
                           "--update-every", "1", "--candidates-per-scan", "3", *flags])
        return run(args, memory_factory=FakeMemory, task_factory=FakeTask)

    def setUp(self):
        FakeMemory.invalid_first = False

    def test_candidates_before_feedback_first_only_official_and_explicit_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = self.execute(root, "--advantage-mode", "group")
            self.assertAlmostEqual(metrics["score"], 0.1)
            self.assertAlmostEqual(metrics["oracle_mean_score"], 0.9)
            self.assertAlmostEqual(metrics["oracle_gain"], 0.8)
            self.assertEqual(metrics["reward_calls"], 6)
            self.assertEqual(metrics["official_reward_calls"], 2)
            self.assertEqual(metrics["diagnostic_reward_calls"], 4)
            self.assertEqual(metrics["learner_feedback_count"], 6)
            self.assertEqual(metrics["generation_tokens"], 12)
            self.assertEqual(metrics["num_updates"], 1)  # No final update.
            self.assertEqual(FakeTask.live.index, 2)  # Forks did not consume scans.
            self.assertEqual(FakeTask.live.scores, [0.1, 0.1])
            self.assertEqual(FakeMemory.latest.groups, [[0.1, 0.9, 0.5]] * 2)
            for scan in range(2):
                events = [event[0] for event in FakeTask.events if event[1] == scan]
                self.assertEqual(events[:3], ["generate"] * 3)
                self.assertEqual(events[3:6], ["score"] * 3)
                self.assertEqual(events[6:9], ["feedback"] * 3)
            candidates = [json.loads(line) for line in (root / "output/candidates.jsonl").read_text().splitlines()]
            self.assertEqual([item["official"] for item in candidates], [True, False, False] * 2)
            self.assertEqual(len({item["candidate_id"] for item in candidates}), 6)

    def test_invalid_official_is_zero_even_when_other_candidate_is_good(self):
        FakeMemory.invalid_first = True
        with tempfile.TemporaryDirectory() as tmp:
            metrics = self.execute(Path(tmp))
            self.assertEqual(metrics["score"], 0.0)
            self.assertEqual(metrics["invalid_reports"], 2)
            self.assertEqual(metrics["invalid_candidates"], 2)
            self.assertEqual(FakeMemory.latest.feedback[0], (0.0, False))
            self.assertAlmostEqual(metrics["oracle_mean_score"], 0.9)

    def test_frozen_has_same_reward_and_generation_budget_but_no_learning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = self.execute(root, "--method", "frozen")
            self.assertEqual(metrics["reward_calls"], 6)
            self.assertEqual(metrics["generation_tokens"], 12)
            self.assertEqual(metrics["learner_feedback_count"], 0)
            self.assertEqual(FakeMemory.latest.feedback, [])
            self.assertFalse((root / "output/experiences.jsonl").exists())

    def test_summary_is_strictly_past_and_each_scan_is_observed_once(self):
        # Use the real evaluator/public scan parser, but keep generation on CPU.
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args(["--model", "fake", "--output-dir", tmp, "--num-scans", "2",
                               "--candidates-per-scan", "3", "--memory-mode", "summary",
                               "--method", "frozen"])
            run(args, memory_factory=FakeMemory)
            rows = [json.loads(line) for line in (Path(tmp) / "responses.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["summary_context"], "")
            self.assertTrue(rows[1]["summary_context"])
            state = json.loads((Path(tmp) / "public_memory.json").read_text())
            self.assertEqual(state["observed_scans"], 2)


class SelectionTest(unittest.TestCase):
    def setUp(self):
        FakeMemory.invalid_first = False

    def execute(self, root):
        data = root / "data.jsonl"
        data.write_text("{}\n{}\n")
        args = parse_args(["--model", "fake", "--output-dir", str(root / "output"),
                           "--data-path", str(data), "--num-scans", "2",
                           "--update-every", "1", "--candidates-per-scan", "3",
                           "--advantage-mode", "group", "--memory-mode", "summary",
                           "--action-mode", "selection"])
        return run(args, memory_factory=SelectionMemory, task_factory=SelectionTask)

    def test_public_catalog_actions_score_reports_and_train_original_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = self.execute(root)
            out = root / "output"
            self.assertAlmostEqual(metrics["score"], 0.105)  # Candidate 0: 10, then mean(10,12).
            self.assertAlmostEqual(metrics["oracle_mean_score"], 0.895)
            self.assertEqual(metrics["action_mode"], "selection")
            self.assertEqual(metrics["mean_catalog_size"], 2)
            self.assertEqual(metrics["reward_calls"], 6)
            experiences = [json.loads(line) for line in (out / "experiences.jsonl").read_text().splitlines()]
            self.assertEqual([json.loads(row["response"]) for row in experiences],
                             [{"include": [1]}, {"include": [2]}, {"include": [1, 2]}] * 2)
            self.assertTrue(all(row["response_ids"] == [3, 4] for row in experiences))
            self.assertTrue(all("transmitters" not in row["response"] for row in experiences))
            rows = [json.loads(line) for line in (out / "responses.jsonl").read_text().splitlines()]
            self.assertEqual([row["report"]["transmitters"][0]["center_freq"] for row in rows], [10.0, 11.0])
            catalogs = [json.loads(line) for line in (out / "action_catalogs.jsonl").read_text().splitlines()]
            self.assertEqual(len(catalogs), 2)
            self.assertEqual([row["catalog"]["observed_scans"] for row in catalogs], [1, 2])
            self.assertEqual(json.loads((out / "public_memory.json").read_text())["observed_scans"], 2)
            self.assertIn('"include"', catalogs[0]["prompt"])
            self.assertNotIn("HIDDEN FEEDBACK", catalogs[0]["prompt"])

    def test_unknown_selection_id_scores_zero_without_choosing_better_candidate(self):
        FakeMemory.invalid_first = True
        with tempfile.TemporaryDirectory() as tmp:
            metrics = self.execute(Path(tmp))
            self.assertEqual(metrics["score"], 0)
            self.assertEqual(metrics["invalid_reports"], 2)
            self.assertEqual(metrics["invalid_candidates"], 2)
            self.assertEqual(FakeMemory.latest.feedback[0], (0.0, False))
            self.assertAlmostEqual(metrics["oracle_mean_score"], 0.895)

    def test_selection_requires_public_summary(self):
        with self.assertRaises(SystemExit):
            parse_args(["--model", "fake", "--output-dir", "/tmp/unused", "--action-mode", "selection"])


class MutationTest(unittest.TestCase):
    def setUp(self):
        FakeMemory.invalid_first = False

    def execute(self, root, *flags):
        data = root / "data.jsonl"
        data.write_text("{}\n{}\n")
        args = parse_args(["--model", "fake", "--output-dir", str(root / "output"),
                           "--data-path", str(data), "--num-scans", "2", "--update-every", "1",
                           "--candidates-per-scan", "3", "--advantage-mode", "group",
                           "--memory-mode", "summary", "--action-mode", "selection",
                           "--proposal-mode", "mutate", *flags])
        return run(args, memory_factory=MutationMemory, task_factory=SelectionTask)

    def test_prefeedback_mutations_have_valid_distinct_ids_and_honest_token_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = self.execute(root)
            out = root / "output"
            candidates = [json.loads(line) for line in (out / "candidates.jsonl").read_text().splitlines()]
            experiences = [json.loads(line) for line in (out / "experiences.jsonl").read_text().splitlines()]
            self.assertEqual(metrics["model_generated_candidates"], 2)
            self.assertEqual(metrics["mutation_count"], 4)
            self.assertEqual(metrics["generation_tokens"], 4)
            self.assertEqual(metrics["reward_calls"], 6)
            self.assertEqual(metrics["proposal_fallback_candidates"], 0)
            self.assertGreater(metrics["proposed_action_tokens"], 0)
            self.assertAlmostEqual(metrics["score"], 0.105)  # No oracle selection.
            self.assertAlmostEqual(metrics["oracle_mean_score"], 0.895)
            self.assertEqual([row["proposal_source"] for row in candidates],
                             ["model", "public_id_mutation", "public_id_mutation"] * 2)
            for scan in (1, 2):
                group = [row for row in candidates if row["scan"] == scan]
                self.assertEqual(len({tuple(json.loads(row["raw_response"])["include"]) for row in group}), 3)
                self.assertTrue(all(row["parse_error"] is None for row in group))
                events = [event[0] for event in FakeTask.events if event[1] == scan - 1]
                self.assertEqual(events[:4], ["generate", "score", "score", "score"])
            for candidate, experience in zip(candidates, experiences):
                if candidate["proposal_source"] == "public_id_mutation":
                    self.assertIsNone(candidate["generation_seed"])
                    self.assertIsNotNone(candidate["mutation_seed"])
                    self.assertEqual(candidate["generation_tokens"], 0)
                    expected = ProposalTokenizer().encode(experience["response"]) + [999]
                    self.assertEqual(experience["response_ids"], expected)
                    self.assertEqual(candidate["proposed_action_tokens"], len(expected))
            self.assertEqual(metrics["proposed_action_tokens"],
                             sum(row["proposed_action_tokens"] for row in candidates))

    def test_frozen_has_identical_proposals_and_feedback_budget(self):
        with tempfile.TemporaryDirectory() as online, tempfile.TemporaryDirectory() as frozen:
            a = self.execute(Path(online))
            b = self.execute(Path(frozen), "--method", "frozen")
            def records(root):
                return [json.loads(line) for line in (Path(root) / "output/candidates.jsonl").read_text().splitlines()]
            for left, right in zip(records(online), records(frozen)):
                for key in ("raw_response", "proposal_source", "mutation_seed", "generation_seed", "reward"):
                    self.assertEqual(left[key], right[key])
            for key in ("reward_calls", "generation_tokens", "proposed_action_tokens", "mutation_count"):
                self.assertEqual(a[key], b[key])
            self.assertEqual(b["learner_feedback_count"], 0)

    def test_invalid_official_uses_fresh_model_fallbacks_and_correct_generation_indices(self):
        FakeMemory.invalid_first = True
        with tempfile.TemporaryDirectory() as tmp:
            metrics = self.execute(Path(tmp))
            candidates = [json.loads(line) for line in (Path(tmp) / "output/candidates.jsonl").read_text().splitlines()]
            self.assertEqual(metrics["model_generated_candidates"], 6)
            self.assertEqual(metrics["mutation_count"], 0)
            self.assertEqual(metrics["generation_tokens"], 12)
            self.assertEqual(metrics["proposed_action_tokens"], 0)
            self.assertEqual(metrics["proposal_fallback_candidates"], 4)
            self.assertEqual(metrics["score"], 0)
            self.assertEqual([row["model_generation_index"] for row in candidates], [0, 1, 2] * 2)
            self.assertTrue(all(row["generation_seed"] is not None for row in candidates))

    def test_mutation_order_varies_public_ids_only_and_exhausts_finite_space(self):
        from ttcl.ramp.spectrum_actions import build_candidate_catalog, decode_selection
        from ttcl.ramp.spectrum_memory import SpectrumMemory
        from ttcl.ramp.tests.test_spectrum_memory import observation

        catalog = build_candidate_catalog(SpectrumMemory(), observation(1, [(10, 1), (50, 1), (90, 1)]))
        for included in ([], [1], [1, 2, 3]):
            raw = json.dumps({"include": included})
            actions = public_id_mutations(raw, catalog, 2, 42)
            self.assertEqual(actions, public_id_mutations(raw, catalog, 2, 42))
            self.assertEqual(len(set(actions)), 2)
            lengths = [len(json.loads(action)["include"]) for action in actions]
            self.assertEqual(lengths, [0, 2] if included == [1] else ([1, 1] if not included else [2, 2]))
            for action in actions:
                decode_selection(action, catalog)
        single = build_candidate_catalog(SpectrumMemory(), observation(1, [(10, 1)]))
        self.assertEqual(public_id_mutations('{"include":[1]}', single, 5, 42), ['{"include":[]}'])
        self.assertEqual(public_id_mutations('```json\n{"include":[1]}\n```', single, 5, 42), ['{"include":[]}'])
        ordered = public_id_mutations('{"include":[3,1]}', catalog, 2, 42)
        self.assertIn(json.loads(ordered[0])["include"], ([3], [1]))
        self.assertEqual(json.loads(ordered[1])["include"], [3, 1, 2])
        with self.assertRaises(ValueError):
            public_id_mutations('{"include":[1,1]}', catalog, 2, 42)

    def test_mutations_require_selection_and_multiple_candidates(self):
        for flags in (["--proposal-mode", "mutate"],
                      ["--proposal-mode", "mutate", "--action-mode", "selection", "--memory-mode", "summary"]):
            with self.assertRaises(SystemExit):
                parse_args(["--model", "fake", "--output-dir", "/tmp/unused", *flags])


class TensorBatch(dict):
    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, device):
        return self


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def __call__(self, text, **kwargs):
        return TensorBatch(input_ids=torch.tensor([[1, 2]]))

    def decode(self, ids, **kwargs):
        return str(ids)


class FakeModel:
    def eval(self):
        return self

    def generate(self, input_ids, **kwargs):
        self.options = kwargs
        ids = torch.randint(3, 10000, (1, 8)) if kwargs["do_sample"] else torch.ones((1, 8), dtype=torch.long)
        return torch.cat((input_ids, ids), dim=1)


class SamplingTest(unittest.TestCase):
    def memory(self, sampled=True):
        args = parse_args(["--model", "fake", "--output-dir", "/tmp/unused", "--device", "cpu",
                           *(["--do-sample"] if sampled else [])])
        memory = RewardMemory.__new__(RewardMemory)
        memory.args = args
        memory.model = FakeModel()
        memory.tokenizer = FakeTokenizer()
        memory.generation_counts = {}
        return memory

    def test_training_rng_and_candidate_count_do_not_change_next_first_candidate(self):
        first = Query(prompt="first", response_schema=ScanReport, instance_id="first")
        second = Query(prompt="second", response_schema=ScanReport, instance_id="second")
        one, three = self.memory(), self.memory()
        torch.manual_seed(321)
        state = torch.random.get_rng_state()
        a = one.respond(first)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        torch.rand(123)  # A different amount of stochastic training/model setup.
        self.assertEqual(a, three.respond(first))
        self.assertNotEqual(a["response_ids"], three.respond(first)["response_ids"])
        three.respond(first)
        torch.rand(999)
        self.assertEqual(one.respond(second), three.respond(second))
        self.assertEqual(one.model.options["temperature"], 0.7)
        self.assertEqual(one.model.options["top_p"], 0.9)
        self.assertEqual(one.model.options["top_k"], 0)

    def test_greedy_default_and_sampling_validation(self):
        memory = self.memory(sampled=False)
        memory.respond(Query(prompt="x", response_schema=ScanReport))
        self.assertFalse(memory.model.options["do_sample"])
        self.assertNotIn("temperature", memory.model.options)
        for flag, value in [("--temperature", "0"), ("--top-p", "1.5"),
                            ("--top-k", "-1"), ("--candidates-per-scan", "0")]:
            with self.assertRaises(SystemExit):
                parse_args(["--model", "fake", "--output-dir", "/tmp/unused", flag, value])

    def test_selection_prompt_does_not_receive_conflicting_report_schema(self):
        memory = self.memory()
        memory.args.action_mode = "selection"
        prompt = 'Select public candidate IDs. Return {"include":[integer IDs]}.'
        completion = memory.respond(Query(prompt=prompt, response_schema=ScanReport))
        self.assertEqual(completion["prompt"], prompt)
        self.assertNotIn("transmitters", completion["prompt"])


if __name__ == "__main__":
    unittest.main()
