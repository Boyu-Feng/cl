import gzip
import io
import json
import logging
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.interface import EvalMetrics, TaskResult, TaskResults
from src.cli import (
    _archive_existing_run_all_task_outputs,
    _build_display_task_params,
    _extract_run_all_headline,
    _final_task_artifact_exists,
    _load_json_file,
    main,
)
from src.tasks.exploitable_poker.task import Poker


class CLITests(unittest.TestCase):
    def _task_results(self, run_group_id="logtest"):
        result = TaskResult(
            metrics={"total_reward": 1.0},
            summary="ok",
            eval_metrics=EvalMetrics(
                loss_curve=[],
                optimal_performance=1.0,
                actual_performance=1.0,
            ),
        )
        return TaskResults(
            run_group_id=run_group_id,
            results=[result],
            execution_summaries=[
                {"status": "completed", "instance_outcomes": [], "usage": {}}
            ],
        )

    def test_cli_rejects_invalid_numeric_parameter(self):
        stderr = io.StringIO()
        with patch(
            "sys.argv",
            [
                "clbench",
                "run",
                "exploitable_poker",
                "--system",
                "human",
                "--task.num-instances",
                "abc",
            ],
        ):
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as cm:
                    main()

        self.assertEqual(cm.exception.code, 2)
        self.assertIn("Invalid task parameter 'num_instances'", stderr.getvalue())

    def test_inspect_task_prints_parameters_and_variants(self):
        stdout = io.StringIO()
        with patch("sys.argv", ["clbench", "inspect", "task", "exploitable_poker"]):
            with redirect_stdout(stdout):
                main()

        output = stdout.getvalue()
        self.assertIn("Task: exploitable_poker", output)
        self.assertIn("Parameters:", output)
        self.assertIn("variant", output)
        self.assertIn("schedule", output)
        self.assertIn("Named variants (reusable presets):", output)
        self.assertIn("Named schedules:", output)
        self.assertIn("calling_station", output)
        self.assertIn("quick_test", output)

    def test_inspect_task_shows_schedules(self):
        stdout = io.StringIO()
        with patch("sys.argv", ["clbench", "inspect", "task", "exploitable_poker"]):
            with redirect_stdout(stdout):
                main()

        self.assertIn("quick_test", stdout.getvalue())
        self.assertIn("Calling Station", stdout.getvalue())

    def test_inspect_system_prints_parameters(self):
        stdout = io.StringIO()
        with patch("sys.argv", ["clbench", "inspect", "system", "human"]):
            with redirect_stdout(stdout):
                main()

        output = stdout.getvalue()
        self.assertIn("System: human", output)
        self.assertIn("Parameters:", output)
        self.assertIn("name", output)

    def test_dry_run_instantiates_without_running_benchmark(self):
        stdout = io.StringIO()
        with patch(
            "src.cli.run_benchmark",
            side_effect=AssertionError("run_benchmark should not be called in dry-run"),
        ):
            with patch(
                "sys.argv",
                [
                    "clbench",
                    "run",
                    "exploitable_poker",
                    "--system",
                    "human",
                    "--schedule",
                    "quick_test",
                    "--dry-run",
                ],
            ):
                with redirect_stdout(stdout):
                    with self.assertRaises(SystemExit) as cm:
                        main()

        self.assertEqual(cm.exception.code, 0)
        output = stdout.getvalue()
        self.assertIn("Dry run successful.", output)
        self.assertIn("Resolved system params", output)
        self.assertIn("Resolved task params", output)

    def test_dry_run_enables_live_dashboard_by_default(self):
        stdout = io.StringIO()
        with patch(
            "sys.argv",
            [
                "clbench",
                "run",
                "exploitable_poker",
                "--system",
                "human",
                "--schedule",
                "quick_test",
                "--dry-run",
            ],
        ):
            with redirect_stdout(stdout):
                with self.assertRaises(SystemExit) as cm:
                    main()

        self.assertEqual(cm.exception.code, 0)
        self.assertIn("Live dashboard: True", stdout.getvalue())

    def test_dry_run_allows_disabling_live_dashboard(self):
        stdout = io.StringIO()
        with patch(
            "sys.argv",
            [
                "clbench",
                "run",
                "exploitable_poker",
                "--system",
                "human",
                "--schedule",
                "quick_test",
                "--dry-run",
                "--no-live-dashboard",
            ],
        ):
            with redirect_stdout(stdout):
                with self.assertRaises(SystemExit) as cm:
                    main()

        self.assertEqual(cm.exception.code, 0)
        self.assertIn("Live dashboard: False", stdout.getvalue())

    def test_run_all_requires_name(self):
        stderr = io.StringIO()
        with patch("sys.argv", ["clbench", "run-all", "--system", "human"]):
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as cm:
                    main()

        self.assertEqual(cm.exception.code, 2)
        self.assertIn("--name", stderr.getvalue())

    def test_load_json_file_reads_gzipped_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "artifact.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as fh:
                json.dump({"kind": "viewer_artifact"}, fh)

            self.assertEqual(_load_json_file(path), {"kind": "viewer_artifact"})

    def test_run_all_archive_moves_existing_task_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            task_path = run_dir / "tasks" / "codebase_adaptation.json"
            gz_task_path = run_dir / "tasks" / "codebase_adaptation.json.gz"
            log_path = run_dir / "logs" / "codebase_adaptation.log"
            task_path.parent.mkdir(parents=True)
            log_path.parent.mkdir(parents=True)
            task_path.write_text('{"old": true}')
            gz_task_path.write_bytes(gzip.compress(b'{"old": true}'))
            log_path.write_text("old log")

            _archive_existing_run_all_task_outputs(
                final_run_dir=run_dir,
                task_name="codebase_adaptation",
                batch_id="batch123",
            )

            archived = run_dir / "archive" / "batch123" / "codebase_adaptation"
            self.assertFalse(task_path.exists())
            self.assertFalse(gz_task_path.exists())
            self.assertFalse(log_path.exists())
            self.assertEqual(
                (archived / "codebase_adaptation.json").read_text(), '{"old": true}'
            )
            self.assertEqual(
                gzip.decompress(
                    (archived / "codebase_adaptation.json.gz").read_bytes()
                ),
                b'{"old": true}',
            )
            self.assertEqual(
                (archived / "codebase_adaptation.log").read_text(), "old log"
            )

    def test_final_task_artifact_exists_accepts_gzipped_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            task_path = run_dir / "tasks" / "exploitable_poker.json.gz"
            task_path.parent.mkdir(parents=True)
            task_path.write_bytes(gzip.compress(b"{}"))

            self.assertTrue(
                _final_task_artifact_exists(
                    final_run_dir=run_dir,
                    task_name="exploitable_poker",
                    task_entry={"status": "ok"},
                )
            )

    def test_run_all_headline_uses_cumulative_gain_total(self):
        headline = _extract_run_all_headline(
            {
                "aggregate": {
                    "mean_gain_by_index": [0.25, None, -0.1, 0.35],
                    "final_cumulative_mean_gain": 0.166667,
                },
                "runs": [
                    {"aggregate": {"total_reward": 10.0, "total_cost": 1.5}},
                    {"aggregate": {"total_reward": 14.0, "total_cost": 2.5}},
                ],
            }
        )

        self.assertEqual(headline["cumulative_gain"], 0.5)
        self.assertEqual(headline["total_reward"], 12.0)

    def test_run_all_task_parallelism_aliases_short_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stdout = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "clbench",
                    "run-all",
                    "--name",
                    "tp_test",
                    "--system",
                    "human",
                    "--task",
                    "exploitable_poker",
                    "-j",
                    "2",
                    "--dry-run",
                    "--final-results-dir",
                    tmpdir,
                    "--no-live-dashboard",
                ],
            ):
                with redirect_stdout(stdout):
                    main()

            output = stdout.getvalue()
            self.assertIn("Dry run successful.", output)
            self.assertIn("with up to 2 parallel task subprocess(es)", output)
            self.assertIn("(per-task workers from schedule)", output)

    def test_run_all_per_task_parallelism_forwarded_to_inner_run(self):
        captured_cmds: list[list[str]] = []

        class _FakeProc:
            returncode = 0

        def _fake_subprocess_run(cmd, *args, **kwargs):
            captured_cmds.append(list(cmd))
            return _FakeProc()

        with tempfile.TemporaryDirectory() as tmpdir:
            stdout = io.StringIO()
            with patch("subprocess.run", side_effect=_fake_subprocess_run):
                with patch("src.cli._find_clbench_cmd", return_value=["clbench"]):
                    with patch(
                        "sys.argv",
                        [
                            "clbench",
                            "run-all",
                            "--name",
                            "ptp_test",
                            "--system",
                            "human",
                            "--task",
                            "exploitable_poker",
                            "--per-task-parallelism",
                            "4",
                            "--final-results-dir",
                            tmpdir,
                            "--no-live-dashboard",
                        ],
                    ):
                        with redirect_stdout(stdout):
                            main()

        self.assertEqual(len(captured_cmds), 1)
        cmd = captured_cmds[0]
        self.assertIn("--max-workers", cmd)
        max_workers_index = cmd.index("--max-workers")
        self.assertEqual(cmd[max_workers_index + 1], "4")
        self.assertEqual(cmd[1], "run")
        self.assertEqual(cmd[2], "exploitable_poker")
        self.assertIn("up to 4 worker(s) per task", stdout.getvalue())

    def test_run_all_missing_only_dry_run_skips_existing_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "nightly"
            task_path = run_dir / "tasks" / "exploitable_poker.json"
            task_path.parent.mkdir(parents=True)
            task_path.write_text("{}")
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "kind": "run_all_manifest",
                        "system": "human",
                        "system_args": [],
                        "tasks": [
                            {
                                "task": "exploitable_poker",
                                "status": "ok",
                                "final_artifact_path": str(task_path),
                            }
                        ],
                    }
                )
            )

            stdout = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "clbench",
                    "run-all",
                    "--name",
                    "nightly",
                    "--system",
                    "human",
                    "--task",
                    "exploitable_poker",
                    "--missing-only",
                    "--dry-run",
                    "--final-results-dir",
                    str(root),
                ],
            ):
                with redirect_stdout(stdout):
                    main()

            output = stdout.getvalue()
            self.assertIn("No tasks to run.", output)
            self.assertIn("already present in final results", output)

    def test_run_all_missing_only_replaces_stale_task_manifest_entry(self):
        class _FakeProc:
            returncode = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "nightly"
            completed_task_path = run_dir / "tasks" / "sales_prediction.json.gz"
            completed_task_path.parent.mkdir(parents=True)
            completed_task_path.write_bytes(gzip.compress(b"{}"))
            run_dir.mkdir(exist_ok=True)
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "kind": "run_all_manifest",
                        "run_name": "nightly",
                        "run_slug": "nightly",
                        "run_group_id": "old-batch_run-all",
                        "system": "human",
                        "system_args": [],
                        "current_batch_id": "old-batch_run-all",
                        "tasks": [
                            {
                                "task": "exploitable_poker",
                                "status": "failed",
                                "batch_id": "old-batch_run-all",
                                "live_manifest_path": "results/exploitable_poker/live/old-batch_run-all/manifest.json",
                                "final_artifact_path": "final_results/runs/nightly/tasks/exploitable_poker.json.gz",
                            },
                            {
                                "task": "sales_prediction",
                                "status": "ok",
                                "batch_id": "old-batch_run-all",
                                "live_manifest_path": "results/sales_prediction/live/old-batch_run-all/manifest.json",
                                "final_artifact_path": str(completed_task_path),
                            },
                        ],
                    }
                )
            )

            stdout = io.StringIO()
            with patch(
                "src.cli.new_timestamp_run_id", return_value="new-batch_run-all"
            ):
                with patch("src.cli._find_clbench_cmd", return_value=["clbench"]):
                    with patch("subprocess.run", return_value=_FakeProc()):
                        with patch(
                            "sys.argv",
                            [
                                "clbench",
                                "run-all",
                                "--name",
                                "nightly",
                                "--system",
                                "human",
                                "--task",
                                "exploitable_poker",
                                "--missing-only",
                                "--task-parallelism",
                                "1",
                                "--final-results-dir",
                                str(root),
                                "--no-live-dashboard",
                            ],
                        ):
                            with redirect_stdout(stdout):
                                main()

            manifest = _load_json_file(run_dir / "manifest.json")
            task_names = [entry["task"] for entry in manifest["tasks"]]
            self.assertEqual(task_names.count("exploitable_poker"), 1)
            self.assertEqual(task_names.count("sales_prediction"), 1)

            retried_entry = next(
                entry
                for entry in manifest["tasks"]
                if entry["task"] == "exploitable_poker"
            )
            self.assertEqual(retried_entry["status"], "ok")
            self.assertEqual(retried_entry["batch_id"], "new-batch_run-all")
            self.assertEqual(
                retried_entry["live_manifest_path"],
                "results/exploitable_poker/live/new-batch_run-all/manifest.json",
            )

    def test_run_with_config_and_json_params_does_not_consume_flag_values_as_task(self):
        stdout = io.StringIO()
        with patch(
            "sys.argv",
            [
                "clbench",
                "run",
                "--config",
                "configs/exploitable_poker/exploitable_poker_human.json",
                "--system-params",
                '{"name": "cli_override"}',
                "--dry-run",
            ],
        ):
            with redirect_stdout(stdout):
                with self.assertRaises(SystemExit) as cm:
                    main()

        self.assertEqual(cm.exception.code, 0)
        output = stdout.getvalue()
        self.assertIn("Dry run successful.", output)
        self.assertIn("Task: exploitable_poker", output)
        self.assertIn("System: human", output)
        self.assertIn("'name': 'cli_override'", output)

    def test_run_unknown_task_reports_next_steps(self):
        stderr = io.StringIO()
        with patch(
            "sys.argv",
            [
                "clbench",
                "run",
                "exploitable_pokre",
                "--system",
                "human",
                "--schedule",
                "quick_test",
                "--dry-run",
            ],
        ):
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as cm:
                    main()

        self.assertEqual(cm.exception.code, 1)
        output = stderr.getvalue()
        self.assertIn("Unknown task 'exploitable_pokre'", output)
        self.assertIn("Closest matches: exploitable_poker", output)
        self.assertIn("clbench list tasks", output)

    def test_run_execution_failure_surfaces_traceback_by_default(self):
        stderr = io.StringIO()
        with patch("src.cli.run_benchmark", side_effect=RuntimeError("boom")):
            with patch(
                "sys.argv",
                [
                    "clbench",
                    "run",
                    "exploitable_poker",
                    "--system",
                    "human",
                    "--schedule",
                    "quick_test",
                    "--no-live-dashboard",
                ],
            ):
                with redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as cm:
                        main()

        self.assertEqual(cm.exception.code, 1)
        output = stderr.getvalue()
        self.assertIn("Benchmark execution failed.", output)
        self.assertIn("Task: exploitable_poker", output)
        self.assertIn("System: human", output)
        self.assertIn("RuntimeError: boom", output)
        self.assertIn("Traceback", output)
        self.assertIn("Add --debug for extra CLI setup logs.", output)

    def test_run_debug_mode_emits_traceback(self):
        stderr = io.StringIO()
        with patch("src.cli.run_benchmark", side_effect=RuntimeError("boom")):
            with patch(
                "sys.argv",
                [
                    "clbench",
                    "run",
                    "exploitable_poker",
                    "--system",
                    "human",
                    "--schedule",
                    "quick_test",
                    "--no-live-dashboard",
                    "--debug",
                ],
            ):
                with redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as cm:
                        main()

        self.assertEqual(cm.exception.code, 1)
        output = stderr.getvalue()
        self.assertIn("Benchmark execution failed.", output)
        self.assertIn("Traceback", output)
        self.assertIn("RuntimeError: boom", output)

    def test_run_writes_jsonl_log_file_without_debug_console(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_cwd = os.getcwd()
            stderr = io.StringIO()

            def fake_run_benchmark(**kwargs):
                logging.getLogger("tests.cli").debug("default_debug_visible")
                logging.getLogger("tests.cli").info(
                    "benchmark_configured",
                    extra={"payload": "x" * 3000},
                )
                return (
                    None,
                    self._task_results("logtest"),
                    [
                        {
                            "status": "completed",
                            "interactions": [],
                            "execution": {"run_index": 0},
                        }
                    ],
                )

            try:
                os.chdir(tmpdir)
                with patch("src.cli.run_benchmark", side_effect=fake_run_benchmark):
                    with patch(
                        "sys.argv",
                        [
                            "clbench",
                            "run",
                            "exploitable_poker",
                            "--system",
                            "human",
                            "--schedule",
                            "quick_test",
                            "--no-live-dashboard",
                            "--run-group-id",
                            "logtest",
                            "--runs",
                            "1",
                        ],
                    ):
                        with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
                            main()
            finally:
                os.chdir(orig_cwd)

            self.assertEqual(stderr.getvalue(), "")
            log_path = (
                Path(tmpdir)
                / "results"
                / "exploitable_poker"
                / "live"
                / "logtest"
                / "run.log"
            )
            records = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(any(r["event"] == "default_debug_visible" for r in records))
            record = next(r for r in records if r["event"] == "benchmark_configured")
            self.assertEqual(record["event"], "benchmark_configured")
            self.assertEqual(record["level"], "INFO")
            self.assertLess(len(record["payload"]), 1200)

    def test_run_autogenerated_group_uses_timestamp_in_artifact_paths(self):
        run_id = "2026-04-29T15-30-12.123456Z"
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_cwd = os.getcwd()
            captured: dict[str, object] = {}

            def fake_run_benchmark(**kwargs):
                captured.update(kwargs)
                return (
                    None,
                    self._task_results(str(kwargs["run_group_id"])),
                    [
                        {
                            "status": "completed",
                            "interactions": [],
                            "execution": {"run_index": 0},
                        }
                    ],
                )

            try:
                os.chdir(tmpdir)
                with patch("src.cli.new_timestamp_run_id", return_value=run_id):
                    with patch("src.cli.run_benchmark", side_effect=fake_run_benchmark):
                        with patch(
                            "sys.argv",
                            [
                                "clbench",
                                "run",
                                "exploitable_poker",
                                "--system",
                                "human",
                                "--schedule",
                                "quick_test",
                                "--no-live-dashboard",
                                "--runs",
                                "1",
                            ],
                        ):
                            with (
                                redirect_stderr(io.StringIO()),
                                redirect_stdout(io.StringIO()),
                            ):
                                main()
            finally:
                os.chdir(orig_cwd)

            self.assertEqual(captured["run_group_id"], run_id)
            self.assertTrue(
                (
                    Path(tmpdir)
                    / "results"
                    / "exploitable_poker"
                    / "live"
                    / run_id
                    / "run.log"
                ).exists()
            )
            self.assertTrue(
                (
                    Path(tmpdir)
                    / "results"
                    / "exploitable_poker"
                    / "traces"
                    / run_id
                    / "run_0000.json"
                ).exists()
            )
            viewer_artifacts = list(
                (Path(tmpdir) / "results" / "exploitable_poker").glob(
                    f"viewer_artifact_{run_id}_*.json.gz"
                )
            )
            self.assertEqual(len(viewer_artifacts), 1)

    def test_debug_console_mirrors_enabled_verbosity_as_human_readable_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_cwd = os.getcwd()
            stderr = io.StringIO()

            def fake_run_benchmark(**kwargs):
                logging.getLogger("tests.cli").info("console_info")
                return (
                    None,
                    self._task_results("logtest"),
                    [
                        {
                            "status": "completed",
                            "interactions": [],
                            "execution": {"run_index": 0},
                        }
                    ],
                )

            try:
                os.chdir(tmpdir)
                with patch("src.cli.run_benchmark", side_effect=fake_run_benchmark):
                    with patch(
                        "sys.argv",
                        [
                            "clbench",
                            "run",
                            "exploitable_poker",
                            "--system",
                            "human",
                            "--schedule",
                            "quick_test",
                            "--no-live-dashboard",
                            "--run-group-id",
                            "logtest",
                            "--runs",
                            "1",
                            "--debug",
                            "-v",
                        ],
                    ):
                        with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
                            main()
            finally:
                os.chdir(orig_cwd)

            console_output = stderr.getvalue()
            self.assertIn("INFO", console_output)
            self.assertIn("console_info", console_output)
            self.assertFalse(console_output.lstrip().startswith("{"))

    def test_display_task_params_include_schedule_resolved_instance_count(self):
        display_params = _build_display_task_params(
            Poker,
            {
                "num_instances": 50,
                "starting_chips": 1000,
                "small_blind": 5,
                "big_blind": 10,
                "seed": 42,
                "schedule": "calling_station_then_fit_or_fold",
            },
        )

        self.assertEqual(display_params["expected_num_instances"], 40)

    def test_no_subcommand_exits_nonzero(self):
        stdout = io.StringIO()
        with patch("sys.argv", ["clbench"]):
            with redirect_stdout(stdout):
                with self.assertRaises(SystemExit) as cm:
                    main()

        # Shows help, exits 1
        self.assertNotEqual(cm.exception.code, 0)
        self.assertIn("clbench", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
