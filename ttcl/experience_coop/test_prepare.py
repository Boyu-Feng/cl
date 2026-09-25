"""Fresh-server preparation keeps provenance and correction choices explicit."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ttcl.experience_evolution.core import read, save
from ttcl.experience_lab.prepare import audit_lineage
from ttcl.experience_v2.common import sha_file
from . import prepare as preparation


class FreshPreparationTests(unittest.TestCase):
    def test_fresh_corrected_run_freezes_settings_without_failed_origin(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            model = workspace/'models/base'
            save(model/'config.json', {'num_hidden_layers': 36})
            for name in ['experience_coop','experience_lab','experience_evolution','experience_v2',
                         'alfworld_comparison','experience_feedback','experience_repair','reflexion_expel',
                         'common','structured_memory','llm_memory']:
                p = workspace/'ttcl'/name/'__init__.py'
                p.parent.mkdir(parents=True); p.write_text('')
            (workspace/'ttcl/paths.py').write_text('')
            def splits(root, **kwargs):
                self.assertTrue(kwargs['fresh_lineage'])
                save(root/'adapters/original_delta/adapter_config.json', {'r': 8})
                parent = {k: [] for k in ['development','final_sequences','development_seeds',
                                         'final_seeds','clbench_splits']}
                parent.update(model=str(model), data_root=str(workspace/'data'), context=65536,
                    actor_max_tokens=64, max_steps=50, writer_tokens=512, memory_tokens=2048,
                    environment_seed=42, already_exposed_test='fixture',
                    initial_adapter=str(root/'adapters/original_delta'), rounds=[])
                for ri in range(2):
                    parent['rounds'].append({'histories': [
                        {'id': f'{ri}_{family}_{i}', 'domain': f'group_{family}'}
                        for family in range(10) for i in range(8)]})
                save(root/'plan.json', parent)
                save(root/'split_audit.json', {'full_scene_hash_disjoint': True})
                save(root/'input_hashes.json', {str(root/'plan.json'): sha_file(root/'plan.json')})
            with patch.object(preparation, 'WORKSPACE', workspace), \
                 patch.object(preparation, 'BENCH', workspace/'bench'), \
                 patch.object(preparation, 'prepare_splits', splits):
                root = workspace/'new_run'
                result = preparation.prepare(root, workspace/'missing.json', gpu=0,
                    rollout_correction='decoupled_token_is', fresh_lineage=True)
            self.assertEqual(result['blocks_per_arm'], 16)
            plan = read(root/'plan.json')
            self.assertEqual(plan['training']['rollout_correction'], 'decoupled_token_is')
            self.assertEqual(plan['training']['max_behavior_logp_mae'], .15)
            self.assertEqual(plan['training']['reject_token_ratio'], 4.)
            self.assertNotIn('reused_initial_rollouts', plan)
            self.assertFalse(read(root/'split_audit.json')['shared_with_experience_lab'])
            self.assertEqual(read(root/'input_hashes.json')[str(root/'plan.json')], sha_file(root/'plan.json'))

    def test_invalid_mode_or_reused_split_does_not_create_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'run'; split = Path(temp)/'existing.json'
            split.write_text('{}')
            for options in [{'rollout_correction': 'typo'}, {'fresh_lineage': True}]:
                with self.assertRaises(ValueError):
                    preparation.prepare(root, split, **options)
                self.assertFalse(root.exists())

    def test_fresh_lineage_excludes_initial_tasks_and_checks_frozen_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            results = Path(temp); old = results/'experience_evolution/initial'
            game = 'json_2.1.1/train/pick_and_place_simple-X/trial/game.tw-pddl'
            save(old/'plan.json', {'training': [{'game': game}]})
            save(old/'data_hashes.json', {game: 'taskhash'})
            save(old/'freeze.json', {'plan_sha256': sha_file(old/'plan.json'),
                                    'data_hashes_sha256': sha_file(old/'data_hashes.json')})
            save(old/'training/delta/status.json', {'phase': 'complete', 'audit': {'base_unchanged': True}})
            adapter = old/'training/delta/adapter'; adapter.mkdir()
            save(adapter/'adapter_config.json', {'r': 8})
            (adapter/'adapter_model.safetensors').write_bytes(b'fixture')
            inventory = {game: {'sha256': 'taskhash'}}
            # Old-machine mode still demands the full historic experiment lineage.
            with self.assertRaises(FileNotFoundError):
                audit_lineage(results, old, inventory)
            train, _, _, audit, hashes = audit_lineage(results, old, inventory, True)
            self.assertIn(game, train)
            self.assertEqual(audit['mode'], 'fresh_local_lineage')
            self.assertIn('experience_repair', audit['missing_historical_projects'])
            self.assertIn(str(old/'freeze.json'), hashes)
            with self.assertRaisesRegex(ValueError, 'task content changed'):
                audit_lineage(results, old, {game: {'sha256': 'different'}}, True)
            save(old/'plan.json', {'training': []})
            with self.assertRaisesRegex(ValueError, 'provenance changed'):
                audit_lineage(results, old, inventory, True)

    def test_unknown_prior_task_role_still_fails_in_fresh_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            results = Path(temp)
            save(results/'experience_v2/run/plan.json', {
                'unknown_role': 'json_2.1.1/train/pick_and_place_simple-X/trial/game.tw-pddl'})
            with self.assertRaisesRegex(ValueError, 'Unclassified'):
                audit_lineage(results, results/'missing', {}, True)


if __name__ == '__main__': unittest.main()
