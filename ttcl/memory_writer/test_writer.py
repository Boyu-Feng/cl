from contextlib import contextmanager
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from ttcl.memory_writer.core import apply_update, messages, parse_update, state_score
from ttcl.memory_writer.evaluate import WriterBackend, update
from ttcl.memory_writer.prepare_data import put, retention_stream, synthetic_row
from ttcl.memory_writer.train import encode_example


class CoreTests(unittest.TestCase):
    def test_atomic_updates_preserve_unmentioned_facts_and_separate_contexts(self):
        initial = [put('A', 'entity', 'owner', 'old', 'e0'), put('B', 'entity', 'owner', 'other', 'e0')]
        memory = apply_update([], {'operations': initial}, {'e0'})
        original = copy.deepcopy(memory)
        changed = apply_update(memory, {'operations': [put('A', 'entity', 'owner', 'new', 'e1')]}, {'e1'})
        self.assertEqual(memory, original)
        self.assertEqual(changed[1], memory[1])
        self.assertEqual(changed[0]['value'], ['new'])
        bad = {'operations': [put('A', 'entity', 'owner', 'new', 'e1'), put('C', 'x', 'owner', 'bad', 'future')]}
        with self.assertRaises(ValueError):
            apply_update(memory, bad, {'e1'})
        self.assertEqual(memory, original)

    def test_generated_training_targets_execute_and_uncertain_claims_do_not_override(self):
        for split in ('train', 'dev', 'test'):
            for i in range(16):
                row = synthetic_row(i, split, random.Random(i))
                expected = apply_update(row['memory_before'], row['target'], {e['id'] for e in row['events']})
                self.assertEqual(expected, row['expected_after'])
                self.assertTrue(state_score(expected, row['expected_after'])['exact_state'])
                payload = json.loads(messages(row['memory_before'], row['events'], row['schema'])[-1]['content'])
                self.assertEqual(set(payload), {'memory', 'field_descriptions', 'events'})
                if row['family'] == 'uncertain':
                    self.assertEqual(expected, row['memory_before'])
        stream = retention_stream(0)['rows']
        self.assertEqual(len(stream), 12)
        first_old = [r for r in stream[0]['expected_after'] if r['entity'] == 'record-0'][0]
        last = [r for r in stream[-1]['expected_after'] if r['entity'] == 'record-0'][0]
        self.assertNotEqual(first_old['value'], last['value'])
        self.assertEqual(len(stream[-1]['expected_after']), 11)

    def test_invalid_or_truncated_model_output_retains_previous_memory(self):
        row = synthetic_row(1, 'test', random.Random(1))

        class Model:
            def call(self, prompt, identity):
                self.prompt = prompt
                return {'raw_response': '{"operations":[]}', 'finish_reason': 'length'}

        model = Model()
        after, _, error = update(model, row['memory_before'], row['events'], None, 'test')
        self.assertIsNotNone(error)
        self.assertEqual(after, row['memory_before'])
        self.assertEqual(parse_update('```json\n{"operations": []}\n```'), {'operations': []})
        with self.assertRaises(ValueError):
            parse_update('{"operations":[],"extra":true}')

    def test_reader_disables_writer_adapter_and_restores_it(self):
        class Adapter:
            enabled = True

            @contextmanager
            def disable_adapter(self):
                self.enabled = False
                try:
                    yield
                finally:
                    self.enabled = True

        with tempfile.TemporaryDirectory() as directory:
            backend = WriterBackend.__new__(WriterBackend)
            backend.model, backend.adapter = Adapter(), Path('adapter')
            backend.out, backend.call_count = Path(directory), 0
            enabled = []

            def generate(this, prompt, seed, **kwargs):
                enabled.append(this.model.enabled)
                return {'raw_response': 'ok'}

            with patch('ttcl.common.local_qwen.LocalQwen.generate', generate):
                backend.call([{'role': 'user', 'content': 'answer'}], 'q1', reader=True)
                backend.call([{'role': 'user', 'content': 'write'}], 'q1', reader=False)
            self.assertEqual(enabled, [False, True])
            self.assertTrue(backend.model.enabled)

    def test_only_assistant_tokens_are_supervised(self):
        from transformers import AutoTokenizer

        model = Path(__file__).resolve().parents[2] / 'current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        row = synthetic_row(0, 'test', random.Random(1))
        encoded = encode_example(tokenizer, row, 4096)
        self.assertIsNotNone(encoded)
        prefix = tokenizer.apply_chat_template(messages(row['memory_before'], row['events'], row['schema']),
                                               tokenize=True, add_generation_prompt=True)
        self.assertTrue(all(x == -100 for x in encoded['labels'][:len(prefix)]))
        self.assertEqual(encoded['labels'][len(prefix):], encoded['input_ids'][len(prefix):])
        self.assertGreater(len(encoded['input_ids']) - len(prefix), 0)
        self.assertIsNone(encode_example(tokenizer, row, 16))


if __name__ == '__main__':
    unittest.main()
