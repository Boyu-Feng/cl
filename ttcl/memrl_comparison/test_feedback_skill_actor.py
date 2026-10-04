"""Focused tests for feedback-based credit and cross-task skill use."""
from __future__ import annotations

import unittest

from .feedback_skill_actor import choose_action, effect, public_state


class FeedbackSkillActorTests(unittest.TestCase):
    def test_effect_requires_exact_action_feedback_alignment(self):
        self.assertEqual(effect('heat mug 1 with microwave 1',
                                'You heat the mug 1 using the microwave 1.'),
                         ('heat', 'mug 1', 'microwave 1'))
        self.assertIsNone(effect('heat mug 1 with microwave 1', 'Nothing happens.'))
        self.assertIsNone(effect('heat mug 1 with microwave 1',
                                 'You heat the mug 2 using the microwave 1.'))
        self.assertIsNone(effect('heat mug 1 with microwave 1',
                                 'You heat the mug 1 using the stoveburner 1.'))

    def test_failed_preparation_does_not_finish_goal(self):
        messages = [
            {'role': 'system', 'content': 'x'},
            {'role': 'user', 'content': 'room\nAvailable commands:\ntake mug 1 from table 1'},
            {'role': 'assistant', 'content': 'Action: take mug 1 from table 1'},
            {'role': 'user', 'content': 'You pick up the mug 1 from the table 1.\nAvailable commands:\nheat mug 1 with microwave 1\ngo to microwave 1'},
            {'role': 'assistant', 'content': 'heat mug 1 with microwave 1'},
            {'role': 'user', 'content': 'Nothing happens.\nAvailable commands:\ngo to microwave 1'},
        ]
        state = public_state('put a hot mug in shelf.', messages)
        self.assertEqual(state['held'], ['mug 1'])
        self.assertEqual(state['prepared'], [])
        table = {'skills': {'heat': [{'tool': 'microwave'}]}}
        self.assertEqual(choose_action(state, ['go to microwave 1'], table),
                         ('go to microwave 1', 'learned_tool_navigation'))

    def test_empty_table_cannot_invoke_cross_task_skill(self):
        state = dict(kind='heat', target='mug', destination='shelf',
                     held=['mug 1'], moved=[], prepared=[], current='table 1')
        self.assertEqual(choose_action(state, ['go to microwave 1'], {'skills': {}}),
                         (None, 'actor_search'))


if __name__ == '__main__':
    unittest.main()
