"""Cross-episode feedback qualification and structured readout checks."""
from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .grounded_evidence_v27 import TypedGroundedMemory as V27Memory
from .grounded_evidence_v30 import TypedGroundedMemory
from .recurrent_feedback import stable_feedback


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)

    def decode(self, tokens, skip_special_tokens=True):
        return ''.join(tokens)


def event(episode, action='inspect', feedback='name | type\n---- | ----\nid | INTEGER'):
    return dict(episode=episode, instance_complete=False,
                submitted_action={'command': action},
                observed_feedback='Tool result (different counter):\n\n' + feedback,
                public_input='Question about the available data and fields',
                source_sha256=f'{episode:064x}',
                public_trace_sha256=f'{episode + 100:064x}')


class RecurrentFeedbackTests(unittest.TestCase):
    def test_only_identical_public_feedback_from_distinct_episodes(self):
        rows = [event(1), event(1), event(2),
                event(3, feedback='Error: invalid query with repeated words'),
                dict(event(4), instance_complete=True),
                event(5, action='different')]
        groups = stable_feedback(rows)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]['episode_count'], 2)
        self.assertEqual(groups[0]['source_sha256'], [f'{1:064x}', f'{2:064x}'])

    def test_structured_context_keeps_native_ids_and_bound(self):
        memory = object.__new__(TypedGroundedMemory)
        memory._text_retry = False
        memory.public_events = [event(1), event(2)]
        memory.client = SimpleNamespace(tokenizer=Tokenizer())
        memory.service = SimpleNamespace(embedding_provider=SimpleNamespace(
            embed=lambda texts: [[1.0, 0.0] for _ in texts]))
        memory.plan = {'memory_tokens': 1000}
        memory._tokens = len
        native = dict(context='Native MemRL', ids=['memory_1'], tokens=12,
                      context_sha256='prior')
        with patch.object(V27Memory, 'retrieve', return_value=native):
            selected = memory.retrieve('Current question about the available data')
        self.assertEqual(selected['ids'], ['memory_1'])
        self.assertIn('Repeated observed feedback', selected['context'])
        self.assertEqual(len(selected['recurrent_feedback']), 1)
        self.assertLessEqual(selected['tokens'], memory.plan['memory_tokens'])
        self.assertEqual(memory.selected_public_context, '')

    def test_text_retry_preserves_native_retrieval(self):
        memory = object.__new__(TypedGroundedMemory)
        memory._text_retry = True
        native = dict(context='native', ids=['memory_1'])
        with patch.object(V27Memory, 'retrieve', return_value=native):
            self.assertIs(memory.retrieve('task'), native)


if __name__ == '__main__':
    unittest.main()
