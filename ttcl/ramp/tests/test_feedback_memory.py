"""Reward attribution, bounded history, and causal feedback integration."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from ttcl.ramp.feedback_memory import FeedbackMemory
from ttcl.ramp.run_reward_benchmark import parse_args, run
from ttcl.ramp.spectrum_actions import build_candidate_catalog, decode_selection
from ttcl.ramp.spectrum_memory import SpectrumMemory
from ttcl.ramp.tests.test_reward_benchmark import FakeMemory, FakeTask, MutationMemory, SelectionTask


def catalog():
    obs = {"scan_number": 1, "band_mhz": [0, 100], "detected_peaks": [
        {"freq_mhz": 10, "width_mhz": 2, "power_dbm": -20},
        {"freq_mhz": 50, "width_mhz": 2, "power_dbm": -20}]}
    return build_candidate_catalog(SpectrumMemory(), obs), obs


def observe(memory, scan, ids, rewards):
    cat, obs = catalog()
    from src.tasks.blind_spectrum_monitoring.task import ScanReport

    raw = [json.dumps({"include": values}) for values in ids]
    return memory.observe(scan=scan, current=obs, catalog=cat, responses=raw,
                          reports=[decode_selection(x, cat, ScanReport) for x in raw],
                          rewards=rewards)


class FeedbackMemoryTest(unittest.TestCase):
    def test_single_feedback_never_manufactures_preference_across_scans(self):
        memory = FeedbackMemory()
        observe(memory, 1, [[1]], [0.1])
        observe(memory, 2, [[2]], [0.9])
        self.assertTrue(all(e["preference"] is None and not e["local_effects"]
                            for e in memory.episodes))
        self.assertNotIn("SAME-SCAN preference:", memory.render())
        self.assertIn("NOT correct/incorrect labels", memory.render())

    def test_exact_single_change_attribution_and_conflicting_evidence(self):
        memory = FeedbackMemory()
        first = observe(memory, 1, [[1, 2], [1]], [0.6, 0.2])
        self.assertEqual(first["local_effects"][0]["candidate_id"], 2)
        self.assertAlmostEqual(first["local_effects"][0]["include_minus_exclude"], 0.4)
        observe(memory, 2, [[1, 2], [1]], [0.1, 0.3])
        cat, _ = catalog()
        text = memory.render(cat)
        self.assertIn("including helped in 1 comparisons, hurt in 1", text)
        self.assertIn("source scans=[1, 2]", text)
        # Do not transfer a singleton lesson to a now-repeated candidate.
        changed = replace(cat, candidates=tuple(replace(r, scan_count=2) for r in cat.candidates))
        self.assertNotIn("including helped", memory.render(changed))

    def test_multiple_changes_rank_whole_answers_without_local_credit(self):
        memory = FeedbackMemory()
        event = observe(memory, 1, [[1], [2]], [0.2, 0.8])
        self.assertEqual(event["preference"]["preferred"], 1)
        self.assertEqual(event["local_effects"], [])
        self.assertIn("SAME-SCAN preference", memory.render())

    def test_equal_scores_invalid_answers_and_nonfinite_rewards(self):
        memory = FeedbackMemory()
        event = observe(memory, 1, [[1], [1, 2]], [0.2, 0.2])
        self.assertIsNone(event["preference"])
        self.assertEqual(event["local_effects"], [])
        event = memory.observe(scan=2, current=None, catalog=None,
                               responses=["bad"], reports=[None], rewards=[0])
        self.assertFalse(event["actions"][0]["valid"])
        self.assertIsNone(event["preference"])
        with self.assertRaises(ValueError):
            observe(memory, 3, [[1]], [float("nan")])
        self.assertEqual(memory.observed_scans, 2)

    def test_bounded_history_duplicates_and_snapshot_isolation(self):
        memory = FeedbackMemory(window=2)
        for scan in range(1, 4):
            observe(memory, scan, [[1]], [0.2])
        state = memory.state_dict()
        self.assertEqual([e["scan"] for e in state["episodes"]], [2, 3])
        state["episodes"].clear()
        self.assertEqual(len(memory.episodes), 2)
        with self.assertRaises(ValueError):
            observe(memory, 3, [[1]], [0.2])
        self.assertEqual(memory.reward_count, 3)

    def test_duplicate_actions_do_not_multiply_local_evidence(self):
        event = observe(FeedbackMemory(), 1, [[1], [1], [1, 2]], [0.2, 0.2, 0.6])
        self.assertEqual(len(event["local_effects"]), 1)


class FeedbackProtocolTest(unittest.TestCase):
    def setUp(self):
        FakeMemory.invalid_first = False

    def test_next_prompt_only_same_training_context_and_unchanged_budget(self):
        for method in ("frozen", "ramp"):
            for selection in (False, True):
                with self.subTest(method=method, selection=selection), tempfile.TemporaryDirectory() as tmp:
                    flags = (["--action-mode", "selection", "--proposal-mode", "mutate",
                              "--memory-mode", "summary"] if selection else [])
                    args = parse_args(["--model", "fake", "--output-dir", tmp,
                                       "--num-scans", "2", "--method", method,
                                       "--candidates-per-scan", "3", "--feedback-memory", "summary",
                                       *flags])
                    result = run(args, memory_factory=MutationMemory if selection else FakeMemory,
                                 task_factory=SelectionTask if selection else FakeTask)
                    root = Path(tmp)
                    rows = [json.loads(x) for x in (root / "responses.jsonl").read_text().splitlines()]
                    self.assertEqual(rows[0]["feedback_context"], "")
                    self.assertIn("through scan 1", rows[1]["feedback_context"])
                    self.assertNotIn("Scan 2 observed", rows[1]["feedback_context"])
                    self.assertNotIn("HIDDEN FEEDBACK", rows[1]["feedback_context"])
                    self.assertEqual(result["reward_calls"], 6)
                    self.assertEqual(result["feedback_memory_reward_count"], 6)
                    self.assertEqual(result["model_generated_candidates"], 2 if selection else 6)
                    state = json.loads((root / "feedback_memory.json").read_text())
                    self.assertEqual(state["observed_scans"], 2)
                    if method == "ramp":
                        examples = [json.loads(x) for x in (root / "experiences.jsonl").read_text().splitlines()]
                        self.assertNotIn("Past scalar-reward experience", examples[0]["prompt"])
                        self.assertIn(rows[1]["feedback_context"], examples[3]["prompt"])

    def test_one_answer_protocol_stays_one_reward(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args(["--model", "fake", "--output-dir", tmp, "--num-scans", "2",
                               "--method", "frozen", "--feedback-memory", "summary"])
            result = run(args, memory_factory=FakeMemory, task_factory=FakeTask)
            self.assertEqual(result["reward_calls"], 2)
            self.assertEqual(result["diagnostic_reward_calls"], 0)
            state = json.loads((Path(tmp) / "feedback_memory.json").read_text())
            self.assertTrue(all(e["preference"] is None for e in state["episodes"]))


if __name__ == "__main__":
    unittest.main()
