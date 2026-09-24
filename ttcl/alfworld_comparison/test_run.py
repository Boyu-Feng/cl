"""Pure runner/report tests; no server, model download, or ALFWorld execution."""
from concurrent.futures import CancelledError
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ttcl.experience_evolution.core import digest, read, save
from . import run
from .report import paired_interval, report, summarize


class RunnerTests(unittest.TestCase):
    def episode(self, job, reward=0.0):
        return {'game': job['game'], 'seed': job['seed'], 'memory': job['memory'],
                'memory_sha256': digest(job['memory']), 'initial_observation': 'Public task',
                'initial_commands_sha256': 'commands', 'status': 'complete',
                'actor_adapter_enabled': False, 'reward': reward, 'steps': 1,
                'trajectory': [{'action': 'look', 'observation': 'A room', 'valid_command': True}],
                'generations': [{'usage': {'prompt_tokens': 5, 'completion_tokens': 2}, 'seconds': .1}]}

    def actor(self):
        def execute(jobs):
            for job in jobs:
                save(Path(job['output'])/'episode.json', self.episode(job))
        return SimpleNamespace(run_many=Mock(side_effect=execute))

    def test_empty_jobs_never_call_actor(self):
        actor = self.actor()
        self.assertEqual(run.run_jobs(actor, []), [])
        actor.run_many.assert_not_called()

    def test_reuse_identical_jobs_and_resume_without_new_execution(self):
        actor = self.actor()
        with tempfile.TemporaryDirectory() as temp:
            jobs = [dict(game='game', seed=42, memory='', output=str(Path(temp)/arm))
                    for arm in ['none', 'delta']]
            episodes = run.run_jobs(actor, jobs)
            self.assertEqual(len(actor.run_many.call_args.args[0]), 1)
            self.assertIn('reused_from', episodes[1])
            new_job = dict(jobs[0], output=str(Path(temp)/'reflexion'))
            replay = run.run_jobs(actor, jobs + [new_job])
            self.assertEqual(actor.run_many.call_count, 1)
            self.assertIn('reused_from', replay[-1])
            with self.assertRaises(ValueError):
                run.run_jobs(actor, [dict(jobs[0], memory='changed')])
            self.assertEqual(actor.run_many.call_count, 1)

    def test_invalid_cached_outcome_is_not_success_or_silent_zero(self):
        actor = self.actor()
        with tempfile.TemporaryDirectory() as temp:
            job = dict(game='game', seed=42, memory='', output=temp)
            save(Path(temp)/'episode.json', self.episode(job, reward=.5))
            with self.assertRaises(ValueError):
                run.run_jobs(actor, [job])
            actor.run_many.assert_not_called()

    def test_cancellation_prevents_next_actor_request(self):
        event = threading.Event()
        actor = run.CancellableActor({}, event)
        try:
            event.set()
            with patch.object(run.Actor, 'generate') as generate:
                with self.assertRaises(CancelledError):
                    actor.generate([], 1)
                generate.assert_not_called()
        finally:
            actor.pool.shutdown()

    def test_all_success_stops_after_one_attempt_and_still_updates_delta(self):
        tokenizer = SimpleNamespace(encode=lambda text, **kwargs: text.split())
        client = SimpleNamespace(tokenizer=tokenizer, session=Mock())
        actor = SimpleNamespace(pool=Mock())
        environment = SimpleNamespace(reset=lambda: {'feedback': 'Public task'}, close=Mock())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = {'actor_url': 'unused', 'context': 1000, 'data_root': temp,
                    'memory_tokens': 2048, 'max_attempts': 3}
            save(root/'plan.json', plan)
            sequence = {'repeat': 1, 'family': 'pick_and_place_simple',
                        'tasks': [{'path': 'train/game', 'never_evaluated': False}]}
            def update(*args, **kwargs):
                return {'raw_response': 'updated memory', 'input_tokens': 4,
                        'output_tokens': 2, 'seconds': .1}
            with patch.object(run, 'Client', return_value=client), \
                 patch.object(run, 'CancellableActor', return_value=actor), \
                 patch.object(run, 'make_env', return_value=environment), \
                 patch.object(run, 'report'), \
                 patch.object(run, 'run_jobs', side_effect=lambda actor, jobs: [self.episode(j, 1) for j in jobs]) as execute, \
                 patch('ttcl.alfworld_comparison.bank.query_context', return_value=('', {})), \
                 patch('ttcl.alfworld_comparison.methods.delta_update', side_effect=update) as writer, \
                 patch('ttcl.alfworld_comparison.methods.reflexion_update') as reflection:
                run.chain(root, sequence, {}, None)
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(writer.call_count, 2)
            reflection.assert_not_called()
            rows = [read(p) for p in (root/'evaluation').glob('*/*/*/*/result.json')]
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(row['attempts'] == 1 and row['success'] == 1 for row in rows))
            client.session.close.assert_called_once()


class ReportTests(unittest.TestCase):
    def row(self, arm, game='game', repeat=1, success=1):
        return {'status': 'complete', 'arm': arm, 'game': game, 'repeat': repeat,
                'family': 'family', 'position': 0, 'never_evaluated': True,
                'success': success, 'first_success': success, 'attempts': 1,
                'actor_calls': 1, 'physical_actor_calls': 1}

    def test_bootstrap_mean_matches_paired_cells_with_unequal_seed_counts(self):
        rows = []
        for repeat in [1, 2, 3]:
            rows += [self.row('delta', 'game_a', repeat, 1), self.row('retry_none', 'game_a', repeat, 0)]
        rows += [self.row('delta', 'game_b', 1, 0), self.row('retry_none', 'game_b', 1, 0)]
        value = paired_interval(rows, 'delta', repeats=50)
        self.assertEqual(value['mean_difference'], .75)
        self.assertEqual(value['paired_count'], 4)
        self.assertEqual(value['task_clusters'], 2)
        with self.assertRaises(ValueError):
            summarize(rows + [rows[0]], ['retry_none', 'delta'])

    def test_report_reads_exact_grid_and_rejects_premature_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = {'arms': ['retry_none', 'delta'], 'max_attempts': 3, 'expected_cells': 2,
                    'sequences': [{'family': 'family', 'repeat': 1,
                                   'tasks': [{'path': 'game', 'never_evaluated': True}]}]}
            save(root/'plan.json', plan)
            first = root/'evaluation/family/1/task_000/retry_none/result.json'
            second = root/'evaluation/family/1/task_000/delta/result.json'
            save(first, self.row('retry_none'))
            self.assertEqual(report(root)['completed_cells'], 1)
            with self.assertRaises(RuntimeError):
                report(root, final=True)
            save(second, self.row('delta'))
            complete = report(root, final=True)
            self.assertEqual(complete['completed_cells'], 2)
            self.assertEqual(complete['all']['single_attempt_none']['successes'], 1)
            self.assertEqual(complete['all']['arms']['delta']['physical_actor_calls'], 1)
            save(second, self.row('delta', game='wrong game'))
            with self.assertRaises(ValueError):
                report(root)


if __name__ == '__main__':
    unittest.main()
