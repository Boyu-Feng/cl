"""Checks provenance and generic schema-path evidence behavior."""

import unittest

from ttcl.trajectory_hyperlora.structured_evidence_gate_v16 import (
    candidate, extract, features,
)


class EvidenceTest(unittest.TestCase):
    def test_extract_and_merge_without_field_name_rules(self):
        source = {"items/name": [{"id": 1}], "probability": .2}
        records = extract(source, source_index=2,
            trajectory_sha256="trace", feedback={"ok": True},
            reward=.4, query="find item 1")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].schema_path, "/items~1name")
        self.assertEqual(records[0].source_trajectory_sha256, "trace")
        current = {"items/name": [{"id": 2}], "probability": .9}
        merged = candidate(current, records[0])
        self.assertEqual(merged["items/name"], [{"id": 2}, {"id": 1}])
        self.assertEqual(merged["probability"], .9)
        self.assertEqual(current["items/name"], [{"id": 2}])
        self.assertEqual(len(features(current, records[0], target_index=3,
                                      target_query="find item 2")), 8)

    def test_non_array_field_has_no_evidence(self):
        records = extract({"forecast": .5}, source_index=1,
            trajectory_sha256="trace", feedback=None,
            reward=.1, query="forecast")
        self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
