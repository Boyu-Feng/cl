"""Checks for source binding and causal typed prompt selection."""

import unittest

from ttcl.trajectory_hyperlora.typed_prompt_evidence_v23 import (
    extract_records, render, select, verify,
)


class TypedPromptEvidenceTest(unittest.TestCase):
    def test_negative_official_reward_preserved_and_source_bound(self):
        records = [{"observation": "Current tables", "action": {
            "action": "QUERY", "content": ".tables"}, "feedback": "items_g1"}]
        item = extract_records(records, source_index=2, reward=-26.0)[0]
        self.assertEqual(item.reward, -26.0)
        self.assertTrue(verify(item, records))
        self.assertFalse(verify(item, [{**records[0], "feedback": "other"}]))

    def test_future_sources_excluded_and_types_retained(self):
        records = [{"observation": "Find reviewers", "action": {
            "action": "QUERY", "content": ".tables", "limit": 5},
            "feedback": "found"}]
        old = extract_records(records, source_index=0, reward=1.0)
        future = extract_records(records, source_index=3, reward=10.0)
        chosen = select(old + future, target_index=2,
                        target_query="Find reviewers")
        self.assertEqual([x.source_index for x in chosen], [0])
        content = render(chosen)
        self.assertIn('"limit":"number"', content)
        self.assertIn(old[0].source_sha256, content)


if __name__ == "__main__":
    unittest.main()
