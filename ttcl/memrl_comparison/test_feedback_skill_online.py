"""Check test-time updates require confirmed effects in distinct games."""
from __future__ import annotations

import unittest

from .probe_feedback_skill_online import SkillStore


def episode(obj: str, observation: str) -> dict:
    return {'game': 'json_2.1.1/valid_unseen/example/game.tw-pddl',
            'trajectory': [{'action': f'clean {obj} with sinkbasin 1',
                            'observation': observation, 'valid_command': True}]}


class OnlineSkillStoreTests(unittest.TestCase):
    def test_updates_after_two_distinct_input_hashes(self):
        store = SkillStore()
        store.observe('a' * 64, episode('mug 1', 'You clean the mug 1 using the sinkbasin 1.'))
        store.observe('a' * 64, episode('mug 2', 'You clean the mug 2 using the sinkbasin 1.'))
        self.assertEqual(store.snapshot()['skills'], {})
        store.observe('b' * 64, episode('mug 3', 'You clean the mug 3 using the sinkbasin 1.'))
        self.assertEqual(store.snapshot()['skills']['clean'][0]['positive_games'], 2)

    def test_failed_feedback_does_not_activate_skill(self):
        store = SkillStore()
        store.observe('a' * 64, episode('mug 1', 'Nothing happens.'))
        store.observe('b' * 64, episode('mug 2', 'You clean the mug 9 using the sinkbasin 1.'))
        self.assertEqual(store.snapshot()['skills'], {})


if __name__ == '__main__':
    unittest.main()
