"""Real-environment plumbing check using TRAIN/prefix tasks only."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from ttcl.experience_evolution.core import save, workspace_root
from .worker import Client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = workspace_root()
    output = args.output.resolve()
    client = Client(args.url)
    from ttcl.experience_feedback.evaluate import configure_tasks
    from ttcl.structured_memory import run_benchmark as base
    from ttcl.structured_memory.online_bank import run_episode
    configure_tasks()
    os.chdir(base.BENCH)
    client.call('begin', arm='deltamem_online')
    settings = SimpleNamespace(task='exploitable_poker', seed=42, num_instances=120,
        memory_chars=20000, max_turns_per_instance=64, action_retries=2,
        normalize_action=True, allow_initial_experience=True)
    row, episode = run_episode(settings, client, 0, output/'poker_train_prefix', '')
    if row['status'] != 'complete':
        raise RuntimeError(row)
    save(output/'poker_train_prefix/row.json', row)
    cl_state = client.call('finish', destination=str(output/'cl_state.pt'),
                          public_feedback=episode['steps'][-1]['public_feedback'])
    from ttcl.alfworld_comparison.environment import Actor

    class LocalActor(Actor):
        def generate(self, messages, random_seed):
            value = client.call('generate', messages=messages, random_seed=random_seed,
                                tokens=64, temperature=.7, top_p=1.0)
            return dict(value, text=value['raw_response'], usage={})

    data = root/'ttcl/data/alfworld_delta'
    game = next(iter(sorted((data/'json_2.1.1/train').glob('*/*/game.tw-pddl'))))
    actor = LocalActor({'data_root': str(data), 'max_steps': 3})
    try:
        client.call('begin', arm='deltamem_reset')
        ep = actor.run_many([dict(game=str(game.relative_to(data)), memory='',
                                 seed=123, output=str(output/'alf_train'))])[0]
        alf_state = client.call('finish', destination=str(output/'alf_state.pt'),
                               public_feedback=ep['trajectory'][-1]['observation'])
    finally:
        actor.pool.shutdown(wait=True)
    save(output/'environment_audit.json', {'passed': True, 'test_data_used': False,
        'cl_prefix_index': 0, 'alf_train_game': str(game.relative_to(data)),
        'cl_official_scoring_completed': True, 'alf_steps': ep['steps'],
        'cl_state': cl_state, 'alf_state': alf_state})
    print('TRAIN-only environment preflight passed', flush=True)


if __name__ == '__main__':
    main()
