from __future__ import annotations

import unittest

from .typed_projection import numeric_consensus, recurring_records, typed_candidates


class TypedProjectionTests(unittest.TestCase):
    def test_repeated_numeric_vector_preserves_every_field(self):
        events = [dict(episode=i, source_sha256=f'source-{i}',
                       submitted_action={f'field_{j}': i / 10 + j / 1000
                                         for j in range(108)})
                  for i in (1, 2, 3)]
        current = {f'field_{j}': 0. for j in range(108)}
        candidate = numeric_consensus(events, current)
        self.assertEqual(candidate['sample_count'], 3)
        self.assertEqual(len(candidate['values']), 108)
        self.assertAlmostEqual(candidate['values']['field_0'], .2)
        self.assertEqual(candidate['status'], 'unverified_submission_aggregate')
        self.assertIsNone(numeric_consensus(events, {'action': 'CALL'}))

    def test_recurring_records_learn_stable_numeric_identity_field(self):
        events = []
        for episode in range(1, 5):
            events.append(dict(episode=episode, source_sha256=f'source-{episode}',
                submitted_action={'records': [
                    {'coordinate': 10 + .2 * episode, 'width': 3 + episode,
                     'strength': -40 - 5 * episode},
                    {'coordinate': 80 + .1 * episode, 'width': 8 + episode,
                     'strength': -60 + 4 * episode}]}))
        current = {'records': [{'coordinate': 10.8, 'width': 6,
                                'strength': -60}]}
        candidates = recurring_records(events, current)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]['identity_field'], 'coordinate')
        self.assertEqual(len(candidates[0]['records']), 2)
        self.assertTrue(all(item['seen_in_episodes'] == 4
                            for item in candidates[0]['records']))
        self.assertEqual(typed_candidates(events, current), candidates)


if __name__ == '__main__':
    unittest.main()
