import io
import json
import logging
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from src.logging_utils import (
    RUN_LOG_LEVEL,
    bind_logging_context,
    build_run_log_path,
    configure_logging,
    reset_logging_for_tests,
)


class LoggingUtilsTests(unittest.TestCase):
    def tearDown(self):
        reset_logging_for_tests()

    def test_jsonl_file_logs_include_context_and_cap_large_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "run.log"
            configure_logging(
                verbosity=1,
                debug_console=False,
                log_file_path=log_path,
            )

            logger = logging.getLogger("tests.logging_utils")
            with bind_logging_context(
                run_group_id="abc123",
                task="task_a",
                system="system_b",
                phase="run",
            ):
                logger.info(
                    "interaction_recorded",
                    extra={"prompt_preview": "x" * 3000, "step": 2},
                )

            record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])

            self.assertEqual(record["event"], "interaction_recorded")
            self.assertEqual(record["level"], "INFO")
            self.assertEqual(record["logger"], "tests.logging_utils")
            self.assertEqual(record["run_group_id"], "abc123")
            self.assertEqual(record["task"], "task_a")
            self.assertEqual(record["system"], "system_b")
            self.assertEqual(record["phase"], "run")
            self.assertEqual(record["step"], 2)
            self.assertLess(len(record["prompt_preview"]), 1200)
            self.assertTrue(record["prompt_preview"].endswith("...[truncated]"))

    def test_console_logging_requires_debug_console(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "run.log"
            stderr = io.StringIO()
            configure_logging(
                verbosity=1,
                debug_console=False,
                log_file_path=log_path,
            )
            with redirect_stderr(stderr):
                logging.getLogger("tests.logging_utils").warning("file_only")

            self.assertEqual(stderr.getvalue(), "")
            self.assertIn("file_only", log_path.read_text(encoding="utf-8"))

            reset_logging_for_tests()
            stderr = io.StringIO()
            configure_logging(
                verbosity=1,
                debug_console=True,
                log_file_path=log_path,
            )
            with redirect_stderr(stderr):
                logging.getLogger("tests.logging_utils").warning("console_visible")

            console_output = stderr.getvalue()
            self.assertIn("WARNING", console_output)
            self.assertIn("console_visible", console_output)
            self.assertFalse(console_output.lstrip().startswith("{"))

    def test_default_verbosity_records_all_file_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "run.log"
            configure_logging(
                verbosity=0,
                debug_console=False,
                log_file_path=log_path,
            )

            logger = logging.getLogger("tests.logging_utils")
            logger.debug("debug_visible")
            logger.info("info_visible")
            logger.log(RUN_LOG_LEVEL, "run_visible")

            records = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [record["event"] for record in records],
                ["debug_visible", "info_visible", "run_visible"],
            )
            self.assertEqual(
                [record["level"] for record in records], ["DEBUG", "INFO", "RUN"]
            )

    def test_build_run_log_path_uses_live_run_group_directory(self):
        self.assertEqual(
            build_run_log_path("task_a", "abcdef123456"),
            Path("results/task_a/live/abcdef123456/run.log"),
        )


if __name__ == "__main__":
    unittest.main()
