"""Provenance and non-JSON hybrid-read checks."""

import unittest

from ttcl.trajectory_hyperlora.general_evidence_v22 import (
    compose_structured, digest, extract_records, select_text, text_provider, verify,
)


class GeneralEvidenceTest(unittest.TestCase):
    def test_text_evidence_requires_past_source_and_public_command(self):
        records = [{"observation": "A pillow is on bed 1.",
                    "action": "take pillow 1 from bed 1",
                    "feedback": "You pick up the pillow."}]
        source = digest(records)
        item = extract_records(records, source_index=4,
                               source_sha256=source, reward=1.)[0]
        self.assertTrue(verify(item, records, source))
        self.assertFalse(verify(item, [{**records[0], "feedback": "failed"}], source))
        self.assertEqual(select_text([item], target_index=4, task="pillow",
                         observation="bed", available=[records[0]["action"]]), [])
        hint = text_provider([item], target_index=5)(
            "put pillow in armchair", "A pillow is on bed 1.",
            ["take pillow 2 from bed 1"])
        self.assertIn("You pick up the pillow", hint)
        self.assertIn("Choose only from the current available commands", hint)
        self.assertEqual(text_provider([item], target_index=5)(
            "pillow", "bed", ["look"]), "")

    def test_structured_evidence_can_modify_existing_parameter_action(self):
        from ttcl.trajectory_hyperlora.typed_evidence_v20 import (
            extract as typed_extract, features, fit, proposals,
        )

        action = {"events": [{"id": 2}]}
        record = [{"observation": "same topic", "action": {"events": [{"id": 1}]},
                   "feedback": "accepted"}]
        item = extract_records(record, source_index=0,
            source_sha256=digest(record), reward=1.)[0]
        schema = {"type": "object", "properties": {"events": {"type": "array"}},
                  "required": ["events"]}
        typed = typed_extract(item.action, source_index=0,
            trajectory_sha256=item.source_sha256, feedback=item.feedback,
            reward=1., query=item.observation)[0]
        candidate = proposals(action, typed, schema)[0]
        vector = features(candidate, action, target_index=1,
                          target_query="same topic")
        gate = fit([("array_union", vector, 1.) for _ in range(8)])
        composed, decision = compose_structured(action, [item], schema=schema,
            gate_model=gate, target_index=1, target_query="same topic")
        self.assertEqual(composed, {"events": [{"id": 2}, {"id": 1}]})
        self.assertEqual(decision["decision"], "array_union")
        self.assertEqual(action, {"events": [{"id": 2}]})
        unchanged, _ = compose_structured(action, [item], schema=schema,
            gate_model=gate, target_index=0, target_query="same topic")
        self.assertEqual(unchanged, action)


if __name__ == "__main__":
    unittest.main()
