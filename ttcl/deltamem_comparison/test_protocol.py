import unittest
import tempfile
from pathlib import Path

from .worker import previous_state, final_public_input
from .run import aggregate


class ProtocolTests(unittest.TestCase):
    def test_no_cross_task_leak_in_reset_and_base(self):
        self.assertIsNone(previous_state('base', 1, 'prior_task.pt', 'prior_attempt.pt'))
        self.assertIsNone(previous_state('deltamem_reset', 0, 'prior_task.pt', None))
        self.assertEqual(previous_state('deltamem_reset', 1, 'prior_task.pt', 'prior_attempt.pt'), 'prior_attempt.pt')
        self.assertEqual(previous_state('deltamem_online', 0, 'prior_task.pt', None), 'prior_task.pt')

    def test_unscored_and_missing_pairs_are_not_failures(self):
        rows = [dict(arm=a, task=1, reward=v) for a, v in
                [('base', 0), ('deltamem_reset', None), ('deltamem_online', 1)]]
        rows.append(dict(arm='base', task=2, reward=1))
        result = aggregate(rows, lambda r: r['task'], 'reward')
        self.assertEqual(result['paired_records'], 1)
        self.assertEqual(result['paired_scored'], 0)
        self.assertEqual(result['paired_unscored'], 1)
        self.assertIsNone(result['means']['base'])

    def test_zero_action_terminal_hand_needs_no_model_log(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(final_public_input(Path(directory),
                {'actor_calls': 0, 'status': 'complete', 'reward': 0.5}), ('', []))

    def test_missing_log_after_actor_call_remains_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Missing actor log'):
                final_public_input(Path(directory), {'actor_calls': 1, 'status': 'complete'})
            with self.assertRaisesRegex(ValueError, 'Missing actor log'):
                final_public_input(Path(directory), {'actor_calls': 0, 'status': 'failed'})

    def test_zero_reward_is_scored(self):
        rows = [dict(arm=a, task=1, reward=0) for a in ('base', 'deltamem_reset', 'deltamem_online')]
        result = aggregate(rows, lambda r: r['task'], 'reward')
        self.assertEqual(result['paired_scored'], 1)
        self.assertEqual(result['means']['base'], 0)


if __name__ == '__main__':
    unittest.main()
