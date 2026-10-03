from __future__ import annotations

import hashlib
import json
import unittest

from .structured_evidence import structured_field


class StructuredEvidenceTests(unittest.TestCase):
    def test_field_uses_public_events_and_keeps_source_binding(self):
        trace = json.dumps([{'action': {'thinking': 'private guess', 'value': 3},
                             'public_feedback': 'Observed 3', 'step': 1}])
        field = structured_field(trace, {'thought': 'guess', 'answer': 3})
        self.assertEqual(field['public_trace_sha256'], hashlib.sha256(trace.encode()).hexdigest())
        self.assertEqual(field['events'][0]['action'], {'value': 3})
        self.assertEqual(field['raw_final_action'], {'answer': 3})
        self.assertNotIn('private guess', json.dumps(field))

    def test_same_schema_handles_alfworld_string_actions(self):
        trace = json.dumps({'trajectory': [
            {'action': 'go to desk', 'observation': 'You arrive.'},
            {'action': 'take mug', 'observation': 'You take it.'}]})
        field = structured_field(trace)
        self.assertEqual(field['event_count'], 2)
        self.assertEqual(field['raw_final_action'], 'take mug')
        self.assertEqual(field['events'][0]['feedback'], 'You arrive.')


if __name__ == '__main__':
    unittest.main()
