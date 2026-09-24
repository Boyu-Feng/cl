from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from . import environment


class EnvironmentTests(unittest.TestCase):
    def test_factory_reset_step_close_share_one_lock(self):
        activity_lock = threading.Lock()
        activity = {'current': 0, 'peak': 0, 'calls': 0}
        def touch():
            with activity_lock:
                activity['current'] += 1
                activity['peak'] = max(activity['peak'], activity['current'])
                activity['calls'] += 1
            time.sleep(.002)
            with activity_lock:
                activity['current'] -= 1
        class FakeEnvironment:
            def reset(self):
                touch()
                return {'feedback': 'room', 'admissible_commands': ['look'], 'won': False}
            def step(self, command):
                touch()
                return self.reset(), 0, False
            def close(self):
                touch()
        def factory(game):
            touch()
            return FakeEnvironment()
        barrier = threading.Barrier(4)
        def worker(index):
            barrier.wait()
            env = environment.make_env('game')
            try:
                env.reset()
                env.step('look')
            finally:
                env.close()
                env.close()  # Cleanup is safe after an earlier successful close.
        with patch.object(environment, 'base_make_env', side_effect=factory):
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(worker, range(4)))
        self.assertEqual(activity['peak'], 1)
        self.assertEqual(activity['calls'], 20)

    def test_actor_generation_stays_parallel_and_matches_original_schema(self):
        generation_barrier = threading.Barrier(2)
        class FakeEnvironment:
            def reset(self):
                return {'feedback': 'room', 'admissible_commands': ['look'], 'won': False}
            def step(self, command):
                return {'feedback': 'success', 'admissible_commands': ['look'], 'won': True}, 1, True
            def close(self):
                pass
        class ParallelActor(environment.Actor):
            def generate(self, messages, random_seed):
                generation_barrier.wait(timeout=3)
                return {'text': 'look', 'seed': random_seed, 'finish_reason': 'stop',
                        'usage': {'prompt_tokens': 1, 'completion_tokens': 1}, 'seconds': .01}
        actor = ParallelActor({'data_root': '.', 'max_steps': 3})
        try:
            with tempfile.TemporaryDirectory() as temp, patch.object(
                    environment, 'base_make_env', return_value=FakeEnvironment()):
                jobs = [{'game': 'game', 'seed': i, 'memory': '',
                         'output': str(Path(temp)/str(i))} for i in range(2)]
                episodes = actor.run_many(jobs)
                self.assertEqual([episode['reward'] for episode in episodes], [1.0, 1.0])
                self.assertTrue(all(episode['steps'] == 1 for episode in episodes))
                self.assertTrue(all(episode['actor_adapter_enabled'] is False for episode in episodes))
                self.assertEqual(episodes[0]['initial_observation'], 'room')
                self.assertEqual(episodes[0]['trajectory'],
                    [{'action': 'look', 'observation': 'success', 'valid_command': True}])
        finally:
            actor.pool.shutdown(wait=True)

    def test_preflight_rejects_evaluation_data_and_covers_training_families(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = []
            for index in range(4):
                path = root/f'json_2.1.1/train/family{index}-object/trial/game.tw-pddl'
                path.parent.mkdir(parents=True)
                path.touch()
                paths.append(str(path.relative_to(root)))
            with patch.object(environment, 'stress_test', return_value={'passed': True}) as stress:
                report = environment.concurrent_preflight(root, paths + paths, workers=3, repeats=1)
                self.assertEqual(len(stress.call_args.args[0]), 4)
                self.assertEqual(stress.call_args.kwargs['repeats'], 2)
                self.assertEqual(len(report['training_games']), 4)
            evaluation = root/'json_2.1.1/valid_unseen/family0-object/trial/game.tw-pddl'
            evaluation.parent.mkdir(parents=True)
            evaluation.touch()
            with self.assertRaisesRegex(ValueError, 'train games only'):
                environment.concurrent_preflight(root, [evaluation])

    def test_environment_closed_even_when_actor_generation_fails(self):
        closed = []
        class FakeEnvironment:
            def reset(self):
                return {'feedback': 'room', 'admissible_commands': ['look'], 'won': False}
            def close(self):
                closed.append(True)
        actor = environment.Actor({'data_root': '.', 'max_steps': 1})
        try:
            with tempfile.TemporaryDirectory() as temp, \
                 patch.object(environment, 'base_make_env', return_value=FakeEnvironment()), \
                 patch.object(actor, 'generate', side_effect=RuntimeError('transport failure')):
                with self.assertRaisesRegex(RuntimeError, 'transport failure'):
                    actor.run_many([{'game': 'game', 'seed': 1, 'memory': '', 'output': temp}])
                self.assertEqual(closed, [True])
                self.assertFalse((Path(temp)/'episode.json').exists())
        finally:
            actor.pool.shutdown(wait=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--real-train', action='store_true')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=2)
    args, extra = parser.parse_known_args()
    if args.real_train:
        from ttcl.experience_evolution.core import ROOT, read
        plan = read(ROOT/'ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json')
        games = [group['games'][0] for group in plan['training']]
        print(json.dumps(environment.concurrent_preflight(
            plan['data_root'], games, args.workers, args.repeats, steps=3), indent=2))
    else:
        unittest.main(argv=[__file__, *extra])
