"""Public-evidence, scoping, and schema-drift checks for domain summaries."""

import json
import unittest

from ttcl.structured_memory.codebase_memory import CodebaseMemory
from ttcl.structured_memory.database_memory import DatabaseMemory


class DatabaseMemoryTests(unittest.TestCase):
    def test_schema_extraction_and_model_claims_are_not_evidence(self):
        memory = DatabaseMemory()
        memory.observe(
            "Question 1/12\n\nHow many items?\n\nYou have access",
            {
                "action": "QUERY",
                "content": "PRAGMA table_info(items)",
                "thought": "SECRET_GOLD total is 999",
            },
            "Query result (1/15 queries used, 14 remaining):\n\ncid | name | type | notnull | dflt_value | pk\n----+------+------+---------+------------+---\n0 | price_cents | INTEGER | 0 | NULL | 0",
            instance_id="q1",
            instance_complete=False,
        )
        record = json.loads(memory.context())["records"][0]
        self.assertEqual(record["table"], "items")
        self.assertEqual(record["columns"][0]["name"], "price_cents")
        self.assertNotIn("SECRET_GOLD", memory.context())
        self.assertNotIn("999", memory.context())

    def test_errors_do_not_become_schema_and_public_corrections_are_scoped(self):
        memory = DatabaseMemory()
        memory.observe(
            "How many?",
            {"action": "QUERY", "content": "SELECT fake FROM goods"},
            "Query result (1/15 queries used, 14 remaining):\n\nERROR: no such column: fake",
            instance_id="q1",
            instance_complete=False,
        )
        memory.observe(
            "How many?",
            {"action": "ANSWER", "content": "10"},
            "Question 1: INCORRECT.\nYour answer: 10\nCorrect answer: 12\nExploratory queries used: 1",
            instance_id="q1",
            instance_complete=True,
        )
        records = json.loads(memory.context())["records"]
        self.assertEqual({r["kind"] for r in records}, {"sql_error", "answer_feedback"})
        feedback = next(r for r in records if r["kind"] == "answer_feedback")
        self.assertEqual(feedback["publicly_revealed_correct_answer"], "12")
        self.assertEqual(feedback["instances"], ["q1"])

    def test_sql_catalog_preserves_all_visible_table_names(self):
        memory = DatabaseMemory()
        memory.observe(
            "Explore",
            {
                "action": "QUERY",
                "content": "SELECT name FROM sqlite_master WHERE type='table'",
            },
            "Query result (1/15 queries used, 14 remaining):\n\nname\n------\n"
            + "\n".join(f"table_{i}" for i in range(10)),
            instance_id="q1",
            instance_complete=False,
        )
        record = json.loads(memory.context())["records"][0]
        self.assertEqual(len(record["catalog_objects"]), 10)
        self.assertEqual(record["catalog_objects"][-1]["name"], "table_9")

    def test_drift_marks_old_evidence_stale_before_first_new_action(self):
        memory = DatabaseMemory()
        memory.observe(
            "before",
            {"action": "QUERY", "content": ".tables"},
            "Query result (1/15 queries used, 14 remaining):\n\nitems",
            instance_id="q1",
            instance_complete=False,
        )
        notice = "NOTICE: The live database schema or contents may have changed since your earlier exploration."
        self.assertTrue(
            json.loads(memory.context(notice))["records"][0]["may_be_stale"]
        )
        memory.observe(
            notice,
            {"action": "QUERY", "content": ".tables"},
            "Query result (1/15 queries used, 14 remaining):\n\nproducts",
            instance_id="q2",
            instance_complete=False,
        )
        entries = json.loads(memory.context())["records"]
        self.assertEqual(entries[0]["tables"], ["products"])
        self.assertFalse(entries[0]["may_be_stale"])
        self.assertTrue(entries[1]["may_be_stale"])


class CodebaseMemoryTests(unittest.TestCase):
    def test_repo_scope_observed_paths_and_status(self):
        memory = CodebaseMemory()
        memory.observe(
            "Repository: owner/one",
            {
                "command": "python -m pytest tests/test_a.py",
                "thought": "I solved the issue in hidden.py",
            },
            "<returncode>1</returncode>\n<output>tests/test_a.py:12: AssertionError\n1 failed</output>",
            instance_id="a",
            instance_complete=False,
        )
        self.assertEqual(memory.context("Repository: owner/two"), "")
        result = json.loads(memory.context("Repository: owner/one"))
        record = result["records"][0]
        self.assertEqual(record["kind"], "test_execution")
        self.assertEqual(record["status"], "execution_failed")
        self.assertIn("tests/test_a.py", record["observed_paths"])
        self.assertNotIn("hidden.py", json.dumps(memory.state_dict()))

    def test_zero_exit_is_not_submission_success(self):
        memory = CodebaseMemory()
        memory.observe(
            "Repository: owner/one",
            {"command": "echo PASSED"},
            "<returncode>0</returncode>\n<output>PASSED</output>",
            instance_id="a",
            instance_complete=False,
        )
        self.assertEqual(
            json.loads(memory.context("Repository: owner/one"))["records"][0]["status"],
            "execution_succeeded",
        )
        memory.observe(
            "Repository: owner/one",
            {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"},
            "Submission FAILED (eval status: test_failure).",
            instance_id="a",
            instance_complete=True,
        )
        statuses = {
            r["status"]
            for r in json.loads(memory.context("Repository: owner/one"))["records"]
        }
        self.assertEqual(statuses, {"execution_succeeded", "submission_failed"})

    def test_context_is_bounded_json_and_state_is_defensive_copy(self):
        memory = CodebaseMemory(max_chars=4000)
        for i in range(30):
            memory.observe(
                "Repository: owner/one",
                {"command": f"cat file{i}.py"},
                "<returncode>0</returncode>\n" + "x" * 2000,
                instance_id=str(i),
                instance_complete=True,
            )
        context = memory.context("Repository: owner/one")
        self.assertLessEqual(len(context), 4000)
        self.assertGreater(json.loads(context)["omitted_records"], 0)
        state = memory.state_dict()
        state["repositories"].clear()
        self.assertTrue(memory.state_dict()["repositories"])


if __name__ == "__main__":
    unittest.main()
