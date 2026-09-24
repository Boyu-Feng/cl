"""CPU report checks for matched controls, feedback budgets and failed runs."""

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from ttcl.ramp.analyze_ramp_experiments import summarize


class ExperimentReportTest(unittest.TestCase):
    def write_run(self, root, name, rewards, *, config_changes=None, updates=(), oracle=0.99):
        path = root / name
        path.mkdir()
        config = {
            "model": "/models/local-model",
            "adapter_path": None,
            "data_path": "/data/scans.jsonl",
            "num_scans": len(rewards),
            "seed": 42,
            "generation_seed": 73,
            "method": "frozen" if name.endswith("frozen") else "ramp",
            "memory_mode": "summary" if name.startswith("summary") else "none",
            "candidates_per_scan": 3 if "group" in name else 1,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 0,
            "max_new_tokens": 1536,
            "max_input_tokens": 8192,
            "dtype": "bfloat16",
        }
        config.update(config_changes or {})
        (path / "config.json").write_text(json.dumps(config))
        count = config["candidates_per_scan"]
        metrics = {
            "mean_score": sum(rewards) / len(rewards),
            "oracle_mean_score": oracle,
            "reward_calls": count * len(rewards),
            "learner_feedback_count": count * len(rewards) if config["method"] == "ramp" else 0,
        }
        (path / "metrics.json").write_text(json.dumps(metrics))
        rows = [{"instance_id": f"scan-{index}", "reward": reward,
                 "candidate_index": 0, "oracle_score": oracle, "parse_error": None}
                for index, reward in enumerate(rewards, 1)]
        candidates = [
            {"instance_id": row["instance_id"], "candidate_index": index,
             "official": index == 0, "reward": row["reward"] if index == 0 else oracle}
            for row in rows for index in range(count)
        ]
        for filename, entries in (("responses.jsonl", rows), ("candidates.jsonl", candidates),
                                  ("updates.jsonl", updates)):
            (path / filename).write_text("".join(json.dumps(row) + "\n" for row in entries))
        return path

    def report(self, root):
        with redirect_stdout(io.StringIO()):
            return summarize(root)

    def test_official_score_and_control_delta_never_use_best_of_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_run(root, "group_frozen", [0.25, 0.25], oracle=0.8)
            self.write_run(root, "group_ramp", [0.2, 0.4], oracle=0.99)
            result = self.report(root)
            learned = result["runs"]["group_ramp"]
            self.assertAlmostEqual(learned["mean_score"], 0.3)
            self.assertAlmostEqual(learned["delta_vs_control"], 0.05)
            self.assertAlmostEqual(learned["oracle_mean_score"], 0.99)
            self.assertEqual(learned["reward_calls"], 6)
            self.assertEqual(learned["learner_feedback_count"], 6)
            self.assertEqual(result["runs"]["group_frozen"]["learner_feedback_count"], 0)
            self.assertEqual(result["comparisons"]["group_ramp"]["wins"], 1)
            self.assertEqual(result["comparisons"]["group_ramp"]["losses"], 1)
            # Verify exported artifacts, not just the in-memory result.
            saved = json.loads((root / "comparison.json").read_text())
            self.assertEqual(saved, result)
            with (root / "comparison.csv").open(newline="") as handle:
                csv_rows = {row["variant"]: row for row in csv.DictReader(handle)}
            self.assertAlmostEqual(float(csv_rows["group_ramp"]["mean_score"]), 0.3)
            report = (root / "report.md").read_text()
            self.assertIn("30.000%", report)
            self.assertIn("+5.000", report)
            self.assertIn("99.000%", report)
            self.assertIn("best-of-K IoU（仅诊断）", report)

    def test_mismatched_inputs_or_sampling_are_not_matched_controls(self):
        cases = {
            "model": "/models/other-model",
            "adapter_path": "/models/different-initial-adapter",
            "data_path": "/data/different-scans.jsonl",
            "seed": 43,
            "generation_seed": 74,
            "memory_mode": "summary",
            "candidates_per_scan": 3,
            "temperature": 1.0,
            "top_p": 0.8,
            "top_k": 20,
            "do_sample": False,
            "max_new_tokens": 1024,
            "max_input_tokens": 4096,
            "dtype": "float32",
            "feedback_memory": "summary",
        }
        for key, changed in cases.items():
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.write_run(root, "sample_frozen", [0.2, 0.3])
                self.write_run(root, "sample_ramp", [0.4, 0.5], config_changes={key: changed})
                with self.assertRaisesRegex(ValueError, "Mismatched configurations.*" + key):
                    self.report(root)
                self.assertFalse((root / "comparison.json").exists())

    def test_feedback_memory_ablation_is_separate_from_parameter_learning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_run(root, "summary_frozen", [0.2, 0.3])
            config = {"memory_mode": "summary", "feedback_memory": "summary"}
            self.write_run(root, "feedback_frozen", [0.2, 0.5], config_changes=config)
            self.write_run(root, "feedback_ramp", [0.2, 0.4], config_changes=config)
            result = self.report(root)
            self.assertAlmostEqual(result["feedback_memory_comparisons"]["feedback_frozen"]["mean_delta"], 0.1)
            self.assertAlmostEqual(result["comparisons"]["feedback_ramp"]["mean_delta"], -0.05)

    def test_same_config_with_different_scan_order_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_run(root, "sample_frozen", [0.2, 0.3])
            trial = self.write_run(root, "sample_ramp", [0.4, 0.5])
            path = trial / "responses.jsonl"
            lines = path.read_text().splitlines()
            path.write_text("\n".join(reversed(lines)) + "\n")
            with self.assertRaisesRegex(ValueError, "Mismatched scan order"):
                self.report(root)

    def test_skipped_updates_are_distinct_from_rollbacks_and_step_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            updates = [
                {"accepted": False, "reason": "no_reward_signal", "optimizer_steps": 0},
                {"accepted": False, "reason": "no_new_data", "optimizer_steps": 0},
                {"accepted": False, "reason": "drift_limit", "optimizer_steps": 2,
                 "retained_optimizer_steps": 0},
                {"accepted": True, "reason": "accepted", "optimizer_steps": 3,
                 "retained_optimizer_steps": 3},
            ]
            self.write_run(root, "group_ramp", [0.2] * 20, updates=updates)
            trial = self.report(root)["runs"]["group_ramp"]
            self.assertEqual(trial["accepted_updates"], 1)
            self.assertEqual(trial["rejected_updates"], 1)
            self.assertEqual(trial["skipped_updates"], 2)
            self.assertEqual(trial["optimizer_steps"], 5)
            self.assertEqual(trial["retained_optimizer_steps"], 3)
            self.assertIn("| 1/1/2 |", (root / "report.md").read_text())

    def test_legacy_update_logs_have_conservative_step_count_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            updates = [
                {"accepted": True, "reason": "accepted", "losses": [0.5, 0.4]},
                {"accepted": False, "reason": "drift_limit", "losses": [0.9]},
                {"accepted": False, "reason": "no_reward_signal"},
            ]
            self.write_run(root, "sample_ramp", [0.2] * 16, updates=updates)
            trial = self.report(root)["runs"]["sample_ramp"]
            self.assertEqual(trial["optimizer_steps"], 3)
            self.assertEqual(trial["retained_optimizer_steps"], 2)

    def test_unfinished_variants_remain_visible_without_fabricated_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_run(root, "sample_frozen", [0.2, 0.3])
            plan = {
                "variants": ["sample_frozen", "sample_ramp", "group_ramp",
                             "summary_ramp", "summary_group_ramp"],
                "runs": {
                    "sample_frozen": {"status": "complete"},
                    "sample_ramp": {"status": "failed", "exit_code": 1},
                    "group_ramp": {"status": "running"},
                    "summary_ramp": {"status": "interrupted", "exit_code": -15},
                },
            }
            (root / "experiment_plan.json").write_text(json.dumps(plan))
            result = self.report(root)
            expected = {"sample_ramp": "failed", "group_ramp": "running",
                        "summary_ramp": "interrupted", "summary_group_ramp": "not_started"}
            self.assertEqual(result["unfinished_runs"], expected)
            self.assertEqual(set(result["runs"]), {"sample_frozen"})
            self.assertEqual(result["comparisons"], {})
            report = (root / "report.md").read_text()
            for name, status in expected.items():
                self.assertIn(f"{name}: {status}", report)


if __name__ == "__main__":
    unittest.main()
