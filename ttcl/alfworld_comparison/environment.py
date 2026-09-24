"""Locally serialize TextWorld environment calls without serializing inference.

TextWorld's PDDL, logic, and text parsers share mutable process-global parser
instances. Separate environments are therefore not safe to construct or step
concurrently in threads. This adapter owns one lock for all environment calls;
HTTP actor generation remains parallel. No third-party or historical module is
patched. The command policy and episode schema match experience_evolution.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time

from ttcl.experience_evolution.core import digest, save, seed
from ttcl.experience_evolution.environment import (
    ACTOR_SYSTEM, Actor as BaseActor, clean_command, make_env as base_make_env,
)


ENVIRONMENT_LOCK = threading.RLock()


class Environment:
    def __init__(self, environment):
        self.environment = environment
        self.closed = False

    def reset(self):
        with ENVIRONMENT_LOCK:
            if self.closed:
                raise RuntimeError('Cannot reset a closed environment')
            return self.environment.reset()

    def step(self, command):
        with ENVIRONMENT_LOCK:
            if self.closed:
                raise RuntimeError('Cannot step a closed environment')
            return self.environment.step(command)

    def close(self):
        with ENVIRONMENT_LOCK:
            if not self.closed:
                self.environment.close()
                self.closed = True


def make_env(game):
    """Load a fresh original environment under the shared parser lock."""
    with ENVIRONMENT_LOCK:
        return Environment(base_make_env(game))


class Actor(BaseActor):
    """Original actor generation with a locally bound safe environment factory."""
    def run_many(self, jobs):
        active, results, opened = [], [None] * len(jobs), []
        try:
            for index, job in enumerate(jobs):
                path = Path(job['output'])
                if (path/'episode.json').exists():
                    raise FileExistsError(path)
                game = Path(self.plan['data_root'])/job['game']
                environment = make_env(game)
                opened.append(environment)
                state = environment.reset()
                initial = str(state['feedback'])
                available = list(state['admissible_commands'])
                system = ACTOR_SYSTEM
                if job['memory']:
                    system += '\n\nPast experience:\n' + job['memory']
                messages = [{'role': 'system', 'content': system}]
                active.append({'index': index, 'job': job, 'environment': environment,
                    'state': state, 'messages': messages,
                    'episode': {'game': job['game'], 'memory': job['memory'],
                        'memory_sha256': digest(job['memory']), 'seed': job['seed'],
                        'initial_observation': initial,
                        'initial_commands_sha256': digest(json.dumps(available)),
                        'trajectory': [], 'generations': [], 'reward': 0.0, 'steps': 0,
                        'actor_adapter_enabled': False}})
            for turn in range(self.plan['max_steps']):
                if not active:
                    break
                pending = []
                for item in active:
                    feedback = str(item['state']['feedback'])
                    available = list(item['state']['admissible_commands'])
                    item['messages'].append({'role': 'user', 'content': feedback +
                        '\nAvailable commands:\n' + '\n'.join(available)})
                    pending.append(self.pool.submit(self.generate, item['messages'],
                                                    seed(item['job']['seed'], turn)))
                remaining = []
                for item, future in zip(active, pending):
                    completion = future.result()
                    command = clean_command(completion['text'], item['state']['admissible_commands'])
                    was_valid = command in item['state']['admissible_commands']
                    state, score, done = item['environment'].step(command)
                    item['messages'].append({'role': 'assistant', 'content': completion['text']})
                    episode = item['episode']
                    episode['generations'].append(completion)
                    episode['trajectory'].append({'action': command,
                        'observation': str(state['feedback']), 'valid_command': was_valid})
                    episode.update(reward=float(bool(state['won'])), steps=turn + 1)
                    item['state'] = state
                    if done or state['won'] or turn + 1 == self.plan['max_steps']:
                        episode['status'] = 'complete'
                        episode['termination'] = 'success' if state['won'] else 'budget_or_environment_done'
                        save(Path(item['job']['output'])/'episode.json', episode)
                        results[item['index']] = episode
                        item['environment'].close()
                    else:
                        remaining.append(item)
                active = remaining
        finally:
            for environment in opened:
                environment.close()
        if any(result is None for result in results):
            raise RuntimeError('Incomplete actor execution')
        return results


def environment_trace(game, steps=3, factory=None):
    """Run deterministic public commands and hash the complete observed trace."""
    environment = (factory or make_env)(game)
    try:
        state = environment.reset()
        trace = [{'observation': str(state['feedback']),
                  'commands': list(state['admissible_commands']), 'won': bool(state['won'])}]
        for turn in range(steps):
            available = list(state['admissible_commands'])
            if not available or state['won']:
                break
            command = available[turn % len(available)]
            state, reward, done = environment.step(command)
            trace.append({'action': command, 'observation': str(state['feedback']),
                          'commands': list(state['admissible_commands']),
                          'reward': reward, 'won': bool(state['won']), 'done': done})
            if done:
                break
        return digest(json.dumps(trace, ensure_ascii=False, sort_keys=True))
    finally:
        environment.close()


def stress_test(games, workers=3, repeats=2, steps=3):
    """Compare serial and concurrent real TRAIN environments without a model/GPU.

    Returns a JSON-serializable audit. Uses ``workers * repeats`` concurrent runs
    and rejects evaluation paths so runtime preflight cannot consume test tasks.
    """
    games = [Path(game) for game in games]
    if not games or workers < 1 or repeats < 1 or steps < 1:
        raise ValueError('Stress test needs training games and positive budgets')
    if any('train' not in game.parts or not game.is_file() for game in games):
        raise ValueError('Environment stress test accepts existing official train games only')
    started = time.monotonic()
    # Compare to the unmodified historical factory, run serially under our lock.
    with ENVIRONMENT_LOCK:
        baseline = {str(game): environment_trace(game, steps, base_make_env) for game in games}
    barrier = threading.Barrier(workers)
    def trial(index):
        game = games[index % len(games)]
        barrier.wait(timeout=120)
        actual = environment_trace(game, steps)
        if actual != baseline[str(game)]:
            raise AssertionError(f'Concurrent environment changed public state: {game}')
        return {'game': str(game), 'trace_sha256': actual}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(trial, range(workers * repeats)))
    return {'passed': True, 'workers': workers, 'rounds': repeats,
            'concurrent_runs': len(results), 'distinct_training_games': len(set(games)),
            'steps_per_run': steps, 'sequential_trace_hashes': baseline,
            'baseline_factory': 'experience_evolution.environment.make_env',
            'all_concurrent_traces_match': True, 'test_data_used': False,
            'gpu_used': False, 'seconds': time.monotonic()-started}


def concurrent_preflight(data_root, games, workers=3, repeats=2, steps=2):
    """CPU-only train preflight comparing original serial and safe concurrent calls.

    Accept relative paths under ``data_root`` or absolute paths. Select at most
    one game per supplied training family, and ensure every selected game gets a
    concurrent trial. The report contains paths and public-state hashes only.
    """
    data_root = Path(data_root).resolve()
    selected = {}
    for game in games:
        path = Path(game)
        path = (data_root/path).resolve() if not path.is_absolute() else path.resolve()
        try:
            relative = path.relative_to(data_root)
        except ValueError as exc:
            raise ValueError('Preflight game must be inside data_root') from exc
        parts = relative.parts
        if len(parts) < 4 or parts[0:2] != ('json_2.1.1', 'train') or not path.is_file():
            raise ValueError('Preflight accepts official json_2.1.1/train games only')
        family = parts[2].split('-', 1)[0]
        selected.setdefault(family, path)
    if not selected:
        raise ValueError('Preflight requires at least one training game')
    if workers < 1 or repeats < 1:
        raise ValueError('Preflight workers and repeats must be positive')
    rounds = max(repeats, (len(selected) + workers - 1)//workers)
    report = stress_test(list(selected.values()), workers=workers, repeats=rounds, steps=steps)
    report['families'] = sorted(selected)
    report['training_games'] = [str(path.relative_to(data_root)) for path in selected.values()]
    report['requested_rounds'] = repeats
    return report
