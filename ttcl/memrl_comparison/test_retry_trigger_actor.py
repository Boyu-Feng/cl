"""Failed-attempt routing preserves native generation below the frozen gate."""
from __future__ import annotations

import unittest

from .retry_trigger_actor import RetryTriggeredActor, should_guide_retry
from .test_adaptive_guided_actor import Client


class RetryTriggerTests(unittest.TestCase):
    def test_only_invalid_heavy_failed_attempt_triggers(self):
        def episode(invalid, reward=0):
            return dict(reward=reward, trajectory=[
                dict(valid_command=index >= invalid) for index in range(50)])
        self.assertFalse(should_guide_retry(None))
        self.assertFalse(should_guide_retry(episode(20)))
        self.assertTrue(should_guide_retry(episode(21)))
        self.assertFalse(should_guide_retry(episode(21, reward=1)))
        with self.assertRaisesRegex(ValueError, 'unaudited'):
            should_guide_retry(dict(reward=0, trajectory=[dict(action='look')]))

    def test_one_completion_in_each_mode(self):
        client = Client()
        actor = RetryTriggeredActor(dict(actor_url='http://actor',
                                       actor_max_tokens=32,
                                       actor_temperature=.7), client)
        messages = [dict(role='system', content='Do task'),
                    dict(role='user', content='Room\nAvailable commands:\nlook\ninventory')]
        try:
            native = actor.generate(messages, 123)
            actor.guide_retry = True
            guided = actor.generate(messages, 124)
        finally:
            actor.pool.shutdown(wait=True)
        self.assertFalse(native['guided_choice_used'])
        self.assertTrue(guided['guided_choice_used'])
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(client.session.calls), 1)
        self.assertEqual(client.session.calls[0]['guided_choice'], ['look', 'inventory'])


if __name__ == '__main__':
    unittest.main()
