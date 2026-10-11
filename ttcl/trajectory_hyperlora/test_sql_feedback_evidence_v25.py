import unittest

from ttcl.trajectory_hyperlora.general_evidence_v22 import Evidence
from ttcl.trajectory_hyperlora.sql_feedback_evidence_v25 import (
    apply, candidate, fit, missing_identifiers,
)


class SqlFeedbackEvidenceTests(unittest.TestCase):
    def source(self, index=0):
        return Evidence(index, 2, "source_hash", "action_hash", "feedback_hash",
            "question", {"action": "QUERY", "content":
                "SELECT name FROM sqlite_master WHERE type='table';"},
            "Query result (1/15 queries used, 14 remaining):\n\nname\nitems_g1",
            0.0)

    def test_query_copy_needs_prior_error_and_prior_valid_source(self):
        current = {"action": "QUERY", "content": "SELECT * FROM products_office;"}
        history = [self.source()]
        self.assertIsNone(candidate(current, history, 0, {"products_office"}))
        self.assertIsNone(candidate(current, history, 1, set()))
        chosen = candidate(current, history, 1, {"products_office"})
        self.assertEqual(chosen.source.source_index, 0)
        self.assertEqual(chosen.action["content"], history[0].action["content"])
        self.assertEqual(missing_identifiers("ERROR: no such table: products_office"),
                         {"products_office"})
        action, info = apply(current, history, 1, {"products_office"},
            {"enabled": True, "operator": "sql_feedback_schema_query_copy"})
        self.assertEqual(action["action"], "QUERY")
        self.assertEqual(info["source_feedback_sha256"], "feedback_hash")

    def test_feedback_gate_groups_repeated_steps_by_task(self):
        rows = ([{"split": "train", "target_index": i,
                  "feedback_delta": 1} for i in range(4)] +
                [{"split": "dev", "target_index": i,
                  "feedback_delta": 1} for i in range(4, 7)])
        self.assertTrue(fit(rows)["enabled"])
        self.assertEqual(fit(rows)["train_targets"], 4)


if __name__ == "__main__":
    unittest.main()
