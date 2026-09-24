"""No-GPU checks for ExpeL source isolation, freezing and whole-evidence budgets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from ttcl.experience_evolution.core import FAMILIES, save, workspace_root
from . import bank


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class ExpeLBankTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='alf-expel-bank-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'original'
        self.data = self.root / 'games'
        self.out = self.root / 'bank'
        self.upstream = self.root / 'upstream'
        (self.upstream / 'expel').mkdir(parents=True)
        repo = workspace_root() / 'current_work/ExpeL'
        for src, dest in [('agent/expel.py', 'expel.py'),
                          ('prompts/templates/human.py', 'human.py'),
                          ('prompts/alfworld.py', 'alfworld.py')]:
            shutil.copy2(repo / src, self.upstream / 'expel' / dest)
        self.games = [f'json_2.1.1/train/{FAMILIES[0]}-Object-{i}/trial/game.tw-pddl'
                      for i in range(3)]
        hashes = {}
        for game in self.games:
            path = self.data / game
            path.parent.mkdir(parents=True)
            path.write_text('{}')
            hashes[game] = hashlib.sha256(path.read_bytes()).hexdigest()
        save(self.source / 'plan.json', {'data_root': str(self.data), 'batch_sequences': 4,
                                        'training': [{'games': self.games}]})
        save(self.source / 'data_hashes.json', hashes)
        self.episode_paths = []
        for position, game in enumerate(self.games):
            for branch in (['task'] if position == 0 else ['task', 'baseline']):
                reward = int(position == 0 or (position == 1 and branch == 'baseline'))
                episode = {'game': game, 'seed': 10 + position,
                           'initial_observation': f'Find object {position}.',
                           'initial_commands_sha256': 'initial-reset',
                           'trajectory': [{'action': 'look', 'observation': 'A room.'}],
                           'reward': reward, 'steps': 1, 'status': 'complete',
                           'actor_adapter_enabled': False,
                           'private_not_for_prompt': 'SECRET_INTERNAL_STATE'}
                path = (self.source / 'training/delta/batch_000/seq_0'
                        / f'{branch}_{position}' / 'episode.json')
                save(path, episode)
                self.episode_paths.append(path)
        self.tokenizer = _Tokenizer()
        self.client = SimpleNamespace(tokenizer=self.tokenizer, context=32768)

    def generate(self, client, messages, path, random_seed):
        # Persist a minimal cache as the actual cached_generation helper does.
        value = {'raw_response': 'ADD 1: Inspect objects before moving them.',
                 'input_tokens': 10, 'output_tokens': 8, 'seconds': 0.01,
                 'finish_reason': 'stop'}
        save(path, value)
        return value

    def test_build_uses_train_and_same_task_contrast_then_freezes(self):
        with patch.object(bank, '_generate', side_effect=self.generate) as call:
            state = bank.build_bank(self.client, self.source, self.out, self.upstream)
        self.assertEqual(state['source_episodes'], 5)
        self.assertEqual(len(state['training_games']), 3)
        self.assertEqual(len(state['successful_examples']), 2)
        self.assertEqual(state['num_compare_calls'], 1)
        self.assertEqual(state['num_success_calls'], 1)
        self.assertEqual(state['development_cost']['new_environment_episodes'], 0)
        self.assertFalse(state['test_feedback_used'])
        prompts = json.dumps([args.args[1] for args in call.call_args_list])
        self.assertNotIn('SECRET_INTERNAL_STATE', prompts)
        self.assertNotIn('game.tw-pddl', prompts)
        self.assertIn('household environment', prompts)
        with patch.object(bank, '_generate', side_effect=AssertionError('Must reuse frozen bank')):
            reused = bank.build_bank(self.client, self.source, self.out, self.upstream)
        self.assertEqual(state['bank_id'], reused['bank_id'])

    def test_frozen_source_and_bank_tampering_are_rejected(self):
        with patch.object(bank, '_generate', side_effect=self.generate):
            bank.build_bank(self.client, self.source, self.out, self.upstream)
        path = self.episode_paths[0]
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'hashes changed'):
            bank.build_bank(self.client, self.source, self.out, self.upstream)
        path.write_text(path.read_text()[:-1])
        state_path = self.out / 'state.json'
        state_path.write_text(state_path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'integrity mismatch'):
            bank.build_bank(self.client, self.source, self.out, self.upstream)

    def test_nontrain_and_seed_mismatch_rejected(self):
        plan_path = self.source / 'plan.json'
        plan = json.loads(plan_path.read_text())
        plan['training'][0]['games'][0] = self.games[0].replace('/train/', '/valid_unseen/')
        save(plan_path, plan)
        with self.assertRaisesRegex(ValueError, 'official train'):
            bank._training_sources(self.source)
        plan['training'][0]['games'][0] = self.games[0]
        save(plan_path, plan)
        baseline = next(p for p in self.episode_paths if p.parent.name == 'baseline_1')
        episode = json.loads(baseline.read_text()); episode['seed'] += 1; save(baseline, episode)
        with self.assertRaisesRegex(ValueError, 'mismatched resets/seeds'):
            bank._training_sources(self.source)

    def test_changed_game_hash_rejected(self):
        (self.data / self.games[0]).write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, 'game hash mismatch'):
            bank._training_sources(self.source)

    def test_official_scores_and_hard_rule_cap(self):
        upstream = bank._OfficialExpeL(self.upstream)
        rules = [(f'Old rule number {i}.', 5) for i in range(19)]
        output = ('EDIT 0: Invalid rule.\nADD 20: New first rule.\n'
                  'ADD 21: New second rule.\nADD 22: New third rule.\nADD 23: New fourth rule.')
        updated, operations, rejected, dropped = upstream.update(rules, output)
        self.assertEqual(len(updated), 20)
        self.assertEqual(len(dropped), 3)
        self.assertEqual(rejected, [('EDIT 0', 'Invalid rule.')])
        self.assertTrue(all(score == 5 for _, score in updated[:19]))
        self.assertEqual(updated[-1], ('New first rule.', 2))

    def test_retrieval_uses_same_family_and_never_slices_evidence(self):
        family = FAMILIES[0]
        def document(index, text, query='past task', domain=family):
            return {'index': index, 'game': f'train/{index}', 'family': domain,
                    'query': query, 'trajectory': text, 'source': f'source/{index}',
                    'source_sha256': str(index)}
        docs = [document(0, 'wrong family', domain=FAMILIES[1]),
                document(1, 'same observation', query='target'),
                document(2, 'oversized ' * 100), document(3, 'short complete example'),
                document(4, 'second complete example')]
        state = {'rules': [('oversized ' * 100, 10), ('Use observed commands.', 5)],
                 'rule_template': 'Rules:\n{rules}', 'successful_examples': docs}
        seen = []
        def rank(query, candidates):
            seen.extend(candidates)
            return [(i, 1 - i / 10) for i in range(len(candidates))]
        context, audit = bank.query_context(state, 'target', family,
            SimpleNamespace(rank=rank), self.tokenizer, budget=30)
        self.assertEqual([x['index'] for x in seen], [1, 2, 3, 4])
        self.assertEqual([x['training_index'] for x in audit['selected']], [3, 4])
        self.assertEqual(audit['skipped_rules'][0]['index'], 1)
        self.assertIn('short complete example', context)
        self.assertNotIn('oversized', context)
        self.assertLessEqual(audit['tokens'], 30)
        self.assertEqual(audit['other_family_examples_excluded'], 1)


if __name__ == '__main__':
    unittest.main()
