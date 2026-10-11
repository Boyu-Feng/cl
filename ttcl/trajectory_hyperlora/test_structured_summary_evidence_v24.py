"""Source and schema checks for model-distilled typed evidence."""

import json
import unittest

from ttcl.trajectory_hyperlora.structured_summary_evidence_v24 import (
    build_summary, digest, render, sample_steps, select, writer_input,
)


class StructuredSummaryEvidenceTest(unittest.TestCase):
    def test_summary_binds_original_typed_step(self):
        records = [{"observation": "Current tables", "action": {
            "action": "QUERY", "content": ".tables"},
            "feedback": "items_g1"}]
        value = writer_input(records, source_index=2, reward=-26.0,
                             initial_query="List tables")
        raw = json.dumps({"observation_condition": "Schema unknown",
            "action_pattern": "Inspect tables first",
            "feedback_effect": "Table list shown",
            "applicability": "New database",
            "caution": "Do not reuse source table names",
            "supporting_step_indices": [0]})
        summary = build_summary(raw, value, writer_prompt_sha256="a",
                                writer_response_sha256=digest(raw))
        self.assertEqual(summary["official_reward"], -26.0)
        self.assertEqual(summary["field_types"]["content"], "text")
        self.assertEqual(summary["source_excerpts"][0]["feedback"], "items_g1")
        self.assertIn(summary["source_trajectory_sha256"], render([summary]))
        self.assertEqual(select([summary], target_index=2,
                                target_query="New database"), [])

    def test_unobserved_step_rejected_and_sampling_spans_trajectory(self):
        records = [{"observation": str(i), "action": i,
                    "feedback": "ok"} for i in range(20)]
        self.assertEqual(sample_steps(records)[0], 0)
        self.assertEqual(sample_steps(records)[-1], 19)
        value = writer_input(records, source_index=0, reward=0.5,
                             initial_query="count")
        raw = json.dumps({"observation_condition": "Count needed",
            "action_pattern": "Count observations",
            "feedback_effect": "ok", "applicability": "count task",
            "caution": "verify", "supporting_step_indices": [1]})
        with self.assertRaises(ValueError):
            build_summary(raw, value, writer_prompt_sha256="a",
                          writer_response_sha256=digest(raw))


if __name__ == "__main__":
    unittest.main()
