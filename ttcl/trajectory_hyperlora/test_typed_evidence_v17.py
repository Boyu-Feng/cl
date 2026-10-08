"""Behavioral checks for typed evidence, provenance, and safe abstention."""

from __future__ import annotations

import unittest

import numpy as np

from ttcl.trajectory_hyperlora.typed_evidence_v17 import (
    FEATURES, extract, features, fit, proposals, route, verified,
)


class TypedEvidenceTest(unittest.TestCase):
    def test_nested_json_and_text_have_source_bindings(self):
        action = {"items": [{"id": 1}], "prediction": {"value": .3}, "note": "ok"}
        rows = extract(action, source_index=2, trajectory_sha256="trajectory",
            feedback={"ok": True}, reward=.4, query="find record")
        self.assertEqual({x.path for x in rows},
                         {"/items", "/prediction/value", "/note"})
        self.assertTrue(all(verified(x, source_action=action,
            source_feedback={"ok": True}, trajectory_sha256="trajectory")
            for x in rows))
        self.assertFalse(verified(rows[0], source_action=action,
            source_feedback={"ok": False}, trajectory_sha256="trajectory"))
        command = extract("take mug", source_index=3,
            trajectory_sha256="other", feedback="picked up", reward=1,
            query="get mug")
        self.assertEqual(command[0].kind, "text")
        self.assertEqual(proposals("look", command[0])[0].operator, "text_hint")

    def test_schema_blocks_invalid_scalar_and_keeps_valid_array(self):
        source = {"items": [{"id": 1}], "prediction": 0}
        rows = extract(source, source_index=0, trajectory_sha256="t",
            feedback="ok", reward=1, query="q")
        current = {"items": [{"id": 2}], "prediction": 20}
        schema = {"type": "object", "properties": {
            "items": {"type": "array", "items": {"type": "object"}},
            "prediction": {"type": "integer", "minimum": 15}}}
        candidates = [x for row in rows for x in proposals(current, row, schema)]
        self.assertEqual({x.operator for x in candidates}, {"array_union"})
        self.assertEqual(candidates[0].action["items"], [{"id": 2}, {"id": 1}])

    def test_untrained_operator_abstains_then_uses_reviewed_positive_utility(self):
        source = {"items": [{"id": 1}]}
        evidence = extract(source, source_index=0, trajectory_sha256="t",
            feedback="ok", reward=1, query="q")[0]
        current = {"items": []}
        candidate = proposals(current, evidence)[0]
        empty = {"features": FEATURES, "operators": {}}
        self.assertEqual(route([candidate], current, empty,
            target_index=1, target_query="q")[0], "lora")
        x = features(candidate, current, target_index=1, target_query="q")
        trained = fit([("array_union", x + np.ones(len(x)) * i * .001, .5)
            for i in range(8)], min_examples=8)
        self.assertEqual(route([candidate], current, trained,
            target_index=1, target_query="q")[0], "array_union")
        self.assertEqual(route([], current, trained,
            target_index=1, target_query="q")[0], "lora")


if __name__ == "__main__":
    unittest.main()
