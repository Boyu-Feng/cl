import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ttcl.memory_writer.core import apply_update, messages
from ttcl.memory_writer.preference_train import orient_pairs, preference_loss
from ttcl.memory_writer.utility import probe_correct, probe_memory, probe_prompt, select_pair
from ttcl.memory_writer.utility_data import episode


class UtilityTests(unittest.TestCase):
    def test_memory_prompt_is_independent_of_python_hash_seed_and_field_order(self):
        script = '''
import json
from ttcl.memory_writer.core import apply_update, messages
from ttcl.memory_writer.prepare_data import put
from ttcl.memory_writer.utility import probe_prompt
op = put('scope', 'object', 'location', 'lab', 'e0')
memory = apply_update([], {'operations': [op]}, {'e0'})
print(json.dumps(memory))
print(json.dumps(messages(memory, [{'id':'e0', 'text':'observed'}])))
print(json.dumps(probe_prompt(memory, {'scope':'scope', 'entity':'object', 'attribute':'location'})))
'''
        results = [subprocess.check_output([sys.executable, '-c', script],
                    env=dict(os.environ, PYTHONHASHSEED=str(seed)), text=True) for seed in (1, 2, 42)]
        self.assertEqual(len(set(results)), 1)
        row = episode(0, 'test')['rows'][0]
        memory = row['expected_after']
        reversed_fields = [dict(reversed(list(record.items()))) for record in memory]
        self.assertEqual(probe_prompt(memory, row['probes'][0]), probe_prompt(reversed_fields, row['probes'][0]))
        self.assertEqual(messages(memory, row['events']), messages(reversed_fields, row['events']))

    def test_public_curriculum_handles_empty_state_nested_feedback_and_retractions(self):
        seen = set()
        for split in ('bridge', 'preference', 'dev', 'test'):
            sequence = episode(0, split)
            memory = []
            self.assertEqual(sequence['rows'][0]['memory_before'], [])
            for row in sequence['rows']:
                self.assertEqual(row['memory_before'], memory)
                memory = apply_update(memory, row['target'], {e['id'] for e in row['events']})
                self.assertEqual(memory, row['expected_after'])
                rendered = messages(row['memory_before'], row['events'], row['schema'])
                payload = json.loads(rendered[-1]['content'])
                self.assertEqual(set(payload), {'memory', 'field_descriptions', 'events'})
                self.assertNotIn('expected_after', payload)
                self.assertNotIn('probes', payload)
                event = json.dumps(row['events'])
                self.assertNotIn(event, seen)
                seen.add(event)
            self.assertEqual(len(sequence['rows'][0]['target']['operations']), 2)
            self.assertEqual(sequence['rows'][4]['expected_after'], sequence['rows'][3]['expected_after'])
            self.assertEqual(len(sequence['rows'][-1]['expected_after']), len(sequence['rows'][-2]['expected_after']) - 1)

    def test_followup_candidates_have_same_seed_identity_without_gold_answer_in_prompt(self):
        row = episode(0, 'preference')['rows'][0]
        with tempfile.TemporaryDirectory() as tmp:
            class Backend:
                out = Path(tmp)
                calls = []

                def call(self, prompt, identity, **options):
                    self.calls.append((prompt, identity, options))
                    return {'raw_response': '{"value": [], "status": "unknown"}', 'finish_reason': 'stop'}

            model = Backend()
            probe_memory(model, [], row, 'empty')
            probe_memory(model, row['expected_after'], row, 'written')
            self.assertEqual([c[1] for c in model.calls[:2]], [c[1] for c in model.calls[2:]])
            self.assertTrue(all(c[2]['reader'] for c in model.calls))
            # Expected answers only belong to the scorer, not reader question construction.
            probe = copy.deepcopy(row['probes'][0])
            before = probe_prompt([], probe)
            probe['expected']['value'] = ['FORBIDDEN_GOLD_SENTINEL']
            self.assertEqual(before, probe_prompt([], probe))

    def test_pairs_require_real_utility_gap_validity_and_supported_chosen_state(self):
        low = dict(error=None, state={'precision': 1.0}, response='empty', utility=0.0)
        high = dict(error=None, state={'precision': 1.0}, response='good', utility=1.0)
        bad = dict(error=None, state={'precision': 0.2}, response='hallucinated', utility=1.0)
        self.assertEqual(select_pair([low, high]), (high, low))
        self.assertIsNone(select_pair([low, bad]))
        self.assertIsNone(select_pair([high, dict(high, response='tie')]))
        self.assertIsNone(select_pair([low, dict(high, error='invalid JSON')]))

    def test_shuffled_control_changes_only_half_the_labels_and_preserves_prefix_pairs(self):
        pairs = [dict(id=str(i), chosen=f'good-{i}', rejected=f'bad-{i}', events=['public']) for i in range(12)]
        original = copy.deepcopy(pairs)
        randomized = orient_pairs(pairs, True, 42)
        self.assertEqual(pairs, original)
        self.assertEqual(sum(r['label_flipped'] for r in randomized), 6)
        self.assertEqual(randomized, orient_pairs(pairs, True, 42))
        for before, after in zip(pairs, randomized):
            self.assertEqual(before['id'], after['id'])
            self.assertEqual(before['events'], after['events'])
            self.assertEqual({before['chosen'], before['rejected']}, {after['chosen'], after['rejected']})

    def test_dpo_gradient_increases_chosen_relative_to_rejected(self):
        import math
        import torch
        chosen = torch.tensor(-3.0, requires_grad=True)
        rejected = torch.tensor(-4.0, requires_grad=True)
        loss = preference_loss(chosen, rejected, -3.0, -4.0, 0.1)
        self.assertAlmostEqual(float(loss.detach()), math.log(2), places=6)
        loss.backward()
        self.assertLess(float(chosen.grad), 0)
        self.assertGreater(float(rejected.grad), 0)
        changed = preference_loss(chosen.detach() - chosen.grad, rejected.detach() - rejected.grad, -3.0, -4.0, 0.1)
        self.assertLess(float(changed), float(loss.detach()))

    def test_probe_scores_reject_wrong_status_and_non_string_values(self):
        target = {'value': ['12'], 'status': 'observed'}
        self.assertTrue(probe_correct(json.dumps(target), target))
        self.assertFalse(probe_correct('{"value":[12],"status":"observed"}', target))
        self.assertFalse(probe_correct('{"value":["12"],"status":"hypothesis"}', target))
        self.assertFalse(probe_correct('not JSON', target))


if __name__ == '__main__':
    unittest.main()
