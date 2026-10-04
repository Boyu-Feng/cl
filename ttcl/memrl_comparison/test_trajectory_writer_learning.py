"""Evidence, update, and leakage checks for trajectory writer targets."""
from __future__ import annotations

import unittest

from .trajectory_experience_teacher import claims_from_trajectory, validate_claims
from .trajectory_writer_learning import ClaimMemory, teacher_operations


def episode(trajectory):
    return {'trajectory': trajectory}


class TrajectoryWriterLearningTests(unittest.TestCase):
    def test_successful_preparation_requires_exact_feedback(self):
        trace = [dict(action='heat mug 1 with microwave 1', observation='Nothing happens.'),
                 dict(action='heat mug 1 with microwave 1',
                      observation='You heat the mug 1 using the microwave 1.')]
        claims = claims_from_trajectory(trace)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]['evidence_indices'], [1])
        validate_claims(trace, claims)
        memory = ClaimMemory()
        bad = dict(op='add', key='verified_preparation_tool:heat:microwave',
                   evidence_indices=[0], evidence_row_sha256=claims[0]['evidence_row_sha256'])
        self.assertEqual(memory.apply([bad], episode(trace), 'a' * 64)['applied'], [])

    def test_full_recovery_sequence_and_cross_game_reinforcement(self):
        trace = [dict(action='move mug 1 to shelf 1', observation='Nothing happens.'),
                 dict(action='go to shelf 1', observation='You arrive at shelf 1.'),
                 dict(action='move mug 1 to shelf 1',
                      observation='You move the mug 1 to the shelf 1.')]
        claims = claims_from_trajectory(trace)
        self.assertEqual([claim['kind'] for claim in claims], ['move_retry_after_navigation'])
        memory = ClaimMemory()
        first = teacher_operations(memory, episode(trace))
        self.assertEqual(memory.apply(first, episode(trace), 'a' * 64)['after'][0]['supporting_games'], 1)
        second = teacher_operations(memory, episode(trace))
        self.assertEqual(second[0]['op'], 'reinforce')
        self.assertEqual(memory.apply(second, episode(trace), 'b' * 64)['after'][0]['supporting_games'], 2)
        self.assertNotIn('mug 1', memory.experience_text())


if __name__ == '__main__':
    unittest.main()
