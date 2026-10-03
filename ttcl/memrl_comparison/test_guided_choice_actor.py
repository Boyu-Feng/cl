"""Public-command binding and one-call guided decoding checks."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

from .guided_choice_actor import GuidedChoiceActor, advertised_commands


class Tokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return '\n'.join(item['content'] for item in messages) + '\nassistant:'

    def encode(self, text, add_special_tokens=False):
        return list(text)


class Response:
    def __init__(self, count):
        self.count = count

    def raise_for_status(self):
        pass

    def json(self):
        return dict(usage=dict(prompt_tokens=self.count, completion_tokens=1),
                    choices=[dict(text='look', finish_reason='stop')])


class Session:
    def __init__(self):
        self.calls = []

    def post(self, url, timeout, json):
        self.calls.append(json)
        return Response(len(json['prompt']))


class GuidedChoiceTests(unittest.TestCase):
    def test_only_public_advertised_commands_are_exposed(self):
        messages = [dict(role='system', content='Do the task'),
                    dict(role='user', content='Room\nAvailable commands:\nlook\ninventory')]
        self.assertEqual(advertised_commands(messages), ['look', 'inventory'])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            advertised_commands([dict(role='user', content=
                'Room\nAvailable commands:\nlook\nlook')])

    def test_one_completion_constrained_to_public_commands(self):
        session = Session()
        client = SimpleNamespace(tokenizer=Tokenizer(), session=session,
                                 plan={'context': 1024})
        actor = GuidedChoiceActor(dict(actor_url='http://actor',
                                      actor_max_tokens=32,
                                      actor_temperature=.7), client)
        try:
            result = actor.generate([dict(role='system', content='Do the task'),
                dict(role='user', content='Room\nAvailable commands:\nlook\ninventory')],
                123)
        finally:
            actor.pool.shutdown(wait=True)
        self.assertEqual(result['text'], 'look')
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(session.calls[0]['guided_choice'], ['look', 'inventory'])
        self.assertEqual(session.calls[0]['max_tokens'], 32)


if __name__ == '__main__':
    unittest.main()
