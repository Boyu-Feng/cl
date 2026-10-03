"""Task-local retry notes cannot gain credit as unshown native memories."""
from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

from .grounded_evidence_v27 import TypedGroundedMemory as V27Memory
from .grounded_evidence_v29 import TypedGroundedMemory


class LocalReflectionTest(TestCase):
    def test_retry_uses_current_task_note_and_suppresses_native_ids(self):
        memory = TypedGroundedMemory.__new__(TypedGroundedMemory)
        memory._text_retry = True
        memory._text_attempt = 2
        memory.plan = {'memory_tokens': 2048}
        memory._tokens = lambda text: len(text.split())
        memory.local_reflections = [dict(text='check the observed shelf', attempt=1,
                                         public_trace_sha256='bound-trace')]
        native = dict(context='old task', ids=['old-memory'], tokens=2)
        with patch.object(V27Memory, 'retrieve', return_value=native):
            result = memory.retrieve('current task')
        self.assertEqual(result['ids'], [])
        self.assertEqual(result['suppressed_ids'], ['old-memory'])
        self.assertIn('check the observed shelf', result['context'])
        self.assertEqual(result['local_source_sha256'], ['bound-trace'])

    def test_failed_update_records_bound_native_writer_text(self):
        memory = TypedGroundedMemory.__new__(TypedGroundedMemory)
        memory._text_retry = True
        memory._text_attempt = 1
        memory.local_reflections = []

        def native_update(instance, *args):
            instance.abstracts = ['Try checking the shelf first.']
            return {'new_memory_id': 'new-memory'}

        with patch.object(V27Memory, 'update', native_update):
            result = memory.update('task', 'public trace', -1, False, {},
                                   {'input_sha256': 'bound-input'})
        self.assertEqual(result['new_memory_id'], 'new-memory')
        self.assertEqual(memory.local_reflections[0]['attempt'], 1)
        self.assertEqual(memory.local_reflections[0]['input_binding'],
                         {'input_sha256': 'bound-input'})

    def test_new_task_clears_notes(self):
        memory = TypedGroundedMemory.__new__(TypedGroundedMemory)
        memory.local_reflections = [{'text': 'old'}]
        with patch.object(V27Memory, 'begin_task'):
            memory.begin_task('retryable_text_command', 'new-input')
        self.assertEqual(memory.local_reflections, [])
