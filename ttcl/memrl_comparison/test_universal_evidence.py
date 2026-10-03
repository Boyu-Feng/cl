from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest

from .test_protocol import Client, Embeddings
from .universal_evidence import UniversalEvidenceMemory, public_events, _operations


class NotebookClient(Client):
    tokenizer = SimpleNamespace(encode=lambda value, **kwargs: value.split(),
                                decode=lambda pieces, **kwargs: ' '.join(pieces))

    def complete(self, messages, *args, **kwargs):
        if 'cross-episode evidence notebook' in messages[0]['content']:
            self.messages.append(messages)
            event = json.loads(messages[1]['content'])['new_public_events'][0]['id']
            return dict(raw_response=json.dumps(dict(operations=[dict(
                op='add', claim='A public observation may recur in later instances.',
                support=[event])])), input_tokens=20, output_tokens=25,
                finish_reason='stop')
        return super().complete(messages, *args, **kwargs)


class UniversalEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.plan = dict(upstream=str(Path(__file__).resolve().parents[2] / 'current_work/MemRL'),
                         writer_max_tokens=768, memory_tokens=2048,
                         rl_config=dict(epsilon=0., alpha=.3, gamma=0., topk=3))
        self.client = NotebookClient()
        self.memory = UniversalEvidenceMemory(self.plan, self.client, self.tmp.name,
            dict(threshold=.5, mean=.5, std=.1), Embeddings())

    def test_same_public_event_path_for_different_action_shapes(self):
        for action in ({'tool': 'query', 'content': 'SELECT 1'},
                       {'transmitters': [{'center_freq': 42.}]},
                       {'field__s12': .7}, {'action': 'CALL'}):
            trace = json.dumps([dict(query='public input', action=action,
                                     public_feedback='public measurement')])
            retrieval = self.memory.retrieve('public input')
            self.memory.update('public input', trace, 1., True, retrieval, {})
        self.assertEqual(self.memory.episode_count, 4)
        self.assertEqual(len(self.memory.entries), 4)
        self.assertEqual(len(self.memory.events), 4)
        retrieved = self.memory.retrieve('public input')
        self.assertEqual(retrieved['projection'], 'public_evidence_notebook_v2')
        self.assertIn('public observation', retrieved['context'])
        self.assertIn('Task: public input', retrieved['context'])
        self.assertIn('Experience: Inspect the current evidence before acting.', retrieved['context'])
        path = Path(self.tmp.name) / 'state.json'
        self.memory.snapshot(path)
        restored = UniversalEvidenceMemory(self.plan, self.client,
            Path(self.tmp.name) / 'other', dict(threshold=.5, mean=.5, std=.1), Embeddings())
        restored.restore(path)
        self.assertEqual(restored.entries, self.memory.entries)
        self.assertEqual(restored.events, self.memory.events)

    def test_unbound_or_old_only_citations_are_rejected(self):
        trace = json.dumps([dict(query='public', action={'action': 'CALL'},
                                 public_feedback='observed')])
        fresh = public_events(trace, 1)
        result = self.memory._apply_operations([dict(op='add', claim='bad',
                                                      support=['e_unknown'])], fresh)
        self.assertEqual(result[0], {})
        self.assertEqual(result[3][0]['reason'], 'Notebook citation is missing or unbound')
        self.memory.events[fresh[0]['id']] = fresh[0]
        newer = public_events(trace, 2)
        result = self.memory._apply_operations([dict(op='add', claim='stale',
                                                      support=[fresh[0]['id']])], newer)
        self.assertEqual(result[0], {})
        self.assertEqual(result[3][0]['reason'], 'Notebook citation is missing or unbound')

    def test_complete_operations_survive_truncated_response(self):
        raw = '{"operations":[{"op":"add","claim":"observed","support":["e1"]}, {"op":"add"'
        operations, truncated = _operations(raw)
        self.assertTrue(truncated)
        self.assertEqual(len(operations), 1)

    def test_reasoning_and_hidden_reward_are_not_in_notebook_prompt(self):
        trace = json.dumps([dict(query='public input', action={'thinking': 'secret thought',
                                                               'action': 'CALL'},
                                 public_feedback='observed')])
        self.memory.update('public input', trace, -12345., False,
                           self.memory.retrieve('public input'), {})
        prompt = self.client.messages[0]
        self.assertNotIn('secret thought', str(prompt))
        self.assertNotIn('-12345', str(prompt))
        self.assertEqual(self.memory.notebook_output_tokens, 25)


if __name__ == '__main__':
    unittest.main()
