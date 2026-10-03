"""Adaptive decoder only constrains the turn after an invalid public command."""
from __future__ import annotations

import unittest

from .adaptive_guided_actor import AdaptiveGuidedActor
from .test_guided_choice_actor import Session, Tokenizer


class Client:
    def __init__(self):
        self.calls = []
        self.session = Session()
        self.tokenizer = Tokenizer()
        self.plan = {'context': 1024}

    def complete(self, messages, seed, **kwargs):
        self.calls.append((messages, seed, kwargs))
        return dict(raw_response='invented action', finish_reason='stop',
                    input_tokens=5, output_tokens=2,
                    rendered_prompt_sha256='prompt', seconds=0.)


class AdaptiveGuidedTests(unittest.TestCase):
    def test_native_then_guide_after_invalid_then_native_after_valid(self):
        client = Client()
        actor = AdaptiveGuidedActor(dict(actor_url='http://actor',
                                         actor_max_tokens=32,
                                         actor_temperature=.7), client)
        try:
            system = dict(role='system', content='Do task')
            first = dict(role='user', content='Room\nAvailable commands:\nlook\ninventory')
            a = actor.generate([system, first], 123)
            self.assertFalse(a['guided_choice_used'])
            second = dict(role='user', content='Still here\nAvailable commands:\nlook\ninventory')
            b = actor.generate([system, first,
                                dict(role='assistant', content=a['text']), second], 124)
            self.assertTrue(b['guided_choice_used'])
            third = dict(role='user', content='New room\nAvailable commands:\nlook\ninventory')
            c = actor.generate([system, first,
                                dict(role='assistant', content=a['text']), second,
                                dict(role='assistant', content=b['text']), third], 125)
            self.assertFalse(c['guided_choice_used'])
        finally:
            actor.pool.shutdown(wait=True)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(len(client.session.calls), 1)
        self.assertEqual(client.session.calls[0]['guided_choice'], ['look', 'inventory'])


if __name__ == '__main__':
    unittest.main()
