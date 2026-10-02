from __future__ import annotations

import unittest

from .general_evidence import GeneralEvidenceMemory


class GeneralEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.memory = object.__new__(GeneralEvidenceMemory)
        self.memory.raw_actions = []

    def test_numeric_action_uses_prior_raw_reports_only(self):
        first = {f'field_{i}': float(i) for i in range(12)}
        second = {f'field_{i}': float(i + 2) for i in range(12)}
        current = {f'field_{i}': float(i + 100) for i in range(12)}
        self.assertEqual(self.memory._project(first), (first, 'numeric_first'))
        self.memory.raw_actions = [first, second]
        final, operator = self.memory._project(current)
        self.assertEqual(operator, 'numeric_mean')
        self.assertEqual(final['field_0'], 1.)
        self.assertEqual(final['field_11'], 12.)
        self.assertEqual(current['field_0'], 100.)

    def test_entity_recurrence_counts_distinct_prior_actions(self):
        a = {'entities': [dict(center_freq=10., bandwidth=3., currently_active=True)]}
        b = {'entities': [dict(center_freq=10.5, bandwidth=3., currently_active=True)]}
        self.memory.raw_actions = [a]
        self.assertEqual(self.memory._project({'entities': []})[0], {'entities': []})
        self.memory.raw_actions.append(b)
        final, operator = self.memory._project({'entities': []})
        self.assertEqual(operator, 'recurrent_records')
        self.assertEqual(len(final['entities']), 1)
        self.assertAlmostEqual(final['entities'][0]['center_freq'], 10.25)
        self.assertFalse(final['entities'][0]['currently_active'])
        self.assertEqual(len(a['entities']), 1)

    def test_unstructured_actions_pass_through(self):
        action = {'command': 'go to desk 1'}
        self.assertEqual(self.memory._project(action), (action, 'unchanged'))


if __name__ == '__main__':
    unittest.main()
