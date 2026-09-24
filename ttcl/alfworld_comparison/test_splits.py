import json
from pathlib import Path
import tempfile
import unittest

from .splits import build_manifest, write_manifest


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        self.results = self.root / 'results'
        self.train = self.game('train', 'training', 'train-world')
        self.game('valid_seen', 'seen', 'seen-world')
        self.fresh = self.game('valid_unseen', 'fresh', 'fresh-world')
        self.old = self.game('valid_unseen', 'old', 'old-world')
        self.alias = self.game('valid_unseen', 'alias', 'train-world')
        self.leaked = self.game('valid_unseen', 'leaked', 'leaked-world')
        self.plan('experience_evolution', 'plan.json',
                  {'training': [{'games': [self.train]}], 'evaluation': [{'games': [self.old]}]})
        self.plan('experience_v2', 'training_plan.json',
                  {'screening': [{'game': self.train}], 'evaluation': [{'games': [self.old]}]})
        self.plan('experience_repair', 'plan.json', {'probes': {'h0': [self.leaked]}})
        self.plan('experience_design', 'dataset.json', [{'domain': 'alfworld', 'source_game': self.train}])

    def game(self, split, name, content_id):
        path = self.data / 'json_2.1.1' / split / ('pick_and_place_simple-' + name) / 'trial_1' / 'game.tw-pddl'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'solvable': True, 'content_id': content_id}))
        return path.relative_to(self.data).as_posix()

    def plan(self, project, name, value):
        path = self.results / project / 'run' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def build(self):
        return build_manifest(self.data, self.results)

    def test_excludes_training_path_and_identical_content_but_flags_prior_test(self):
        result = self.build()
        self.assertEqual({x['path'] for x in result['tasks']}, {self.fresh, self.old})
        self.assertEqual({x['path'] for x in result['excluded']}, {self.alias, self.leaked})
        self.assertEqual([x['path'] for x in result['tasks'] if x['never_evaluated']], [self.fresh])
        self.assertEqual(result['counts']['never_evaluated'], 1)
        self.assertTrue(result['checks_passed'])
        self.assertFalse(any(result['intersections'].values()))
        self.assertTrue(next(x for x in result['tasks'] if x['path'] == self.old)['past_evaluation_sources'])

    def test_prior_test_content_alias_is_not_fresh(self):
        alias = self.game('valid_unseen', 'old_alias', 'old-world')
        result = self.build()
        item = next(x for x in result['tasks'] if x['path'] == alias)
        self.assertFalse(item['never_evaluated'])
        self.assertEqual(item['past_evaluation_content_matches'], [self.old])

    def test_same_world_goal_excluded_but_shared_goal_template_allowed(self):
        train = self.game('train', 'semantic_training', 'semantic-train')
        duplicate = self.game('valid_unseen', 'semantic_duplicate', 'different-json')
        shared_goal = self.game('valid_unseen', 'shared_goal', 'different-world')
        for name, problem_id, room in [(train, 'original', 'room1'),
                                      (duplicate, 'renamed', 'room1'),
                                      (shared_goal, 'other', 'room2')]:
            path = self.data / name
            record = json.loads(path.read_text())
            record.update(pddl_domain='(define (domain household))',
                pddl_problem=f'(define (problem {problem_id}) (:domain household) '
                             f'(:init (at agent {room})) (:goal (and (at object target))))')
            path.write_text(json.dumps(record))
        result = self.build()
        excluded = next(x for x in result['excluded'] if x['path'] == duplicate)
        self.assertIn('training_normalized_pddl_problem', {r['kind'] for r in excluded['reasons']})
        allowed = next(x for x in result['tasks'] if x['path'] == shared_goal)
        self.assertTrue(allowed['goal_template_seen_in_training'])
        self.assertTrue(allowed['never_evaluated'])

    def test_unknown_reference_role_fails_closed(self):
        self.plan('experience_design', 'plan.json', {'future_custom_stage': [self.fresh]})
        with self.assertRaisesRegex(ValueError, 'Unclassified'):
            self.build()

    def test_missing_history_and_missing_game_fail_closed(self):
        path = self.results / 'experience_repair/run/plan.json'
        path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'Historical experiment manifests'):
            self.build()
        self.plan('experience_repair', 'plan.json', {'probes': [self.leaked]})
        (self.data / self.leaked).unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'Historical task files absent'):
            self.build()

    def test_train_split_forbidden_and_manifest_cannot_be_overwritten(self):
        with self.assertRaisesRegex(ValueError, 'never train'):
            build_manifest(self.data, self.results, test_split='train')
        target = self.root / 'manifest.json'
        first = write_manifest(target, data_root=self.data, results_root=self.results)
        with self.assertRaises(FileExistsError):
            write_manifest(target, data_root=self.data, results_root=self.results)
        self.assertEqual(first['task_selection_sha256'], json.loads(target.read_text())['task_selection_sha256'])


if __name__ == '__main__':
    unittest.main()
