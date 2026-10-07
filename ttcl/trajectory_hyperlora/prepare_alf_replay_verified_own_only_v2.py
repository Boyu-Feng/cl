"""Replay-review old-LoRA-own successes where the paired base actor failed.

This creates fresh content-bound annotations. It never treats an old history ID
as a label, and never uses official validation/test trajectories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.experience_evolution.environment import make_env
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v2 import save


def review(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = json.loads(args.rollouts.read_text())
    if (raw['failures'] or len(raw['games']) != 240 or
            raw['summary']['n'] != 240 or
            raw['summary']['base'] != 57 or
            raw['summary']['own'] != 73 or
            'train_large transfer' not in raw['protocol']):
        raise ValueError('Changed historical official train rollout source')
    selected = [row for row in raw['games']
                if (row['arms']['own']['reward'] == 1 and
                    row['arms']['base']['reward'] == 0)]
    if len(selected) != 42 or len({row['game'] for row in selected}) != 42:
        raise ValueError('Changed self-success bank or duplicated game')
    report = {'protocol': 'Fresh official ALFWorld train replay-reviewed actor-own success with expert-sibling-LoRA source where paired no-memory base failed; each successful action bound to full content; no new actor generation or official validation/test labels',
        'rollouts_sha256': file_hash(args.rollouts),
        'source_checkpoint_sha256': raw['checkpoint_sha256'],
        'source_arm': 'own', 'paired_base_reward': 0,
        'source_game_count': len(selected),
        'episode_budget': 30, 'rows': [], 'failures': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(selected):
        episode = row['arms']['own']
        game = row['game']
        path = args.data_root / game
        if (not path.is_file() or '/train/' not in game or
                episode['status'] != 'complete' or
                episode['termination'] != 'success' or
                episode['invalid_commands'] != 0 or
                not 1 <= episode['steps'] <= 30 or
                episode['steps'] != len(episode['trajectory']) or
                episode['reward'] != 1.):
            report['failures'].append({'index': index, 'game': game,
                                       'reason': 'changed successful source'})
            save(args.output, report)
            raise ValueError(f'Invalid self-success source {game}')
        env = make_env(path)
        targets = []
        try:
            state = env.reset()
            if state['feedback'] != episode['initial_observation']:
                raise ValueError('Changed initial observation')
            for turn, recorded in enumerate(episode['trajectory']):
                before = str(state['feedback'])
                available = list(state['admissible_commands'])
                command = recorded['command']
                if (recorded['turn'] != turn or not recorded['valid'] or
                        command not in available):
                    raise ValueError(f'Unreviewed or invalid action {turn}')
                state, _, done = env.step(command)
                if (str(state['feedback']) != recorded['observation'] or
                        bool(state['won']) != bool(recorded['won']) or
                        (turn + 1 < episode['steps'] and done)):
                    raise ValueError(f'Changed environment transition {turn}')
                content = {'source_game_sha256': file_hash(path),
                    'source_episode_sha256': digest(episode),
                    'turn': turn, 'observation': before,
                    'admissible_commands': available,
                    'action': command, 'feedback': str(state['feedback']),
                    'won': bool(state['won'])}
                targets.append({'turn': turn, 'input_content_sha256':
                    digest(content), 'observation': before,
                    'admissible_commands': available,
                    'target_action': command,
                    'feedback': str(state['feedback']),
                    'won': bool(state['won'])})
            if not state['won'] or not targets[-1]['won']:
                raise ValueError('Replay did not verify success')
            records = records_from_episode(episode)
            report['rows'].append({'game': game, 'family': row['family'],
                'game_sha256': file_hash(path),
                'episode_sha256': digest(episode),
                'records_sha256': digest(records), 'records': records,
                'steps': episode['steps'], 'reviewed_targets': targets})
            save(args.output, report)
            print(json.dumps({'reviewed_sources': len(report['rows']),
                              'reviewed_actions': sum(x['steps']
                                  for x in report['rows'])}), flush=True)
        except Exception as exc:
            report['failures'].append({'index': index, 'game': game,
                                       'reason': repr(exc)})
            save(args.output, report)
            raise
        finally:
            env.close()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_sibling_train240_taskpair_current1000_20261006.json'))
    parser.add_argument('--data-root', type=Path,
                        default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--output', type=Path, default=Path(
        'data/annotations/alf_own_only42_replay_reviewed_20261007.json'))
    review(parser.parse_args())
