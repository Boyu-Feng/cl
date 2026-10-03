"""The final-retry gate preserves every earlier native attempt."""
from __future__ import annotations

import unittest

from .late_retry_trigger import should_guide_final_retry


def episode(invalid: int, reward: int = 0) -> dict:
    return dict(reward=reward, trajectory=[dict(valid_command=index >= invalid)
                                           for index in range(50)])


class LateRetryTests(unittest.TestCase):
    def test_only_two_failed_attempts_can_trigger(self):
        high, low = episode(21), episode(5)
        self.assertFalse(should_guide_final_retry([]))
        self.assertFalse(should_guide_final_retry([high]))
        self.assertTrue(should_guide_final_retry([high, low]))
        self.assertTrue(should_guide_final_retry([low, high]))
        self.assertFalse(should_guide_final_retry([low, low]))
        self.assertFalse(should_guide_final_retry([high, episode(0, reward=1)]))
        with self.assertRaisesRegex(ValueError, 'exactly two'):
            should_guide_final_retry([low, low, low])


if __name__ == '__main__':
    unittest.main()
