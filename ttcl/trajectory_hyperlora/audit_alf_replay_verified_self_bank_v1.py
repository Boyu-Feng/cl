"""Verify fresh action-label bindings of the replay-reviewed self bank."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)


def audit(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = json.loads(args.rollouts.read_text())
    review = json.loads(args.review.read_text())
    selected = [row for row in raw['games']
                if row['arms']['base']['reward'] == 1]
    if (review['rollouts_sha256'] != file_hash(args.rollouts) or
            review['source_checkpoint_sha256'] != raw['checkpoint_sha256'] or
            review['source_arm'] != 'base' or review['failures'] or
            len(selected) != 57 or len(review['rows']) != 57):
        raise ValueError('Incomplete self-verified bank')
    action_hashes = set()
    for source, row in zip(selected, review['rows'], strict=True):
        episode = source['arms']['base']
        path = args.data_root / row['game']
        if (row['game'] != source['game'] or
                row['family'] != source['family'] or
                row['game_sha256'] != file_hash(path) or
                row['episode_sha256'] != digest(episode) or
                row['records_sha256'] != digest(records_from_episode(episode)) or
                row['steps'] != episode['steps'] or
                len(row['reviewed_targets']) != episode['steps']):
            raise ValueError(f'Changed source binding {row["game"]}')
        before = episode['initial_observation']
        for index, (target, step) in enumerate(zip(
                row['reviewed_targets'], episode['trajectory'], strict=True)):
            expected = {'source_game_sha256': row['game_sha256'],
                'source_episode_sha256': row['episode_sha256'],
                'turn': index, 'observation': before,
                'admissible_commands': target['admissible_commands'],
                'action': step['command'],
                'feedback': step['observation'],
                'won': step['won']}
            binding = digest(expected)
            if (target['turn'] != index or
                    target['input_content_sha256'] != binding or
                    target['input_content_sha256'] in action_hashes or
                    target['observation'] != before or
                    target['target_action'] != step['command'] or
                    target['target_action'] not in
                        target['admissible_commands'] or
                    target['feedback'] != step['observation'] or
                    target['won'] != step['won']):
                raise ValueError(f'Changed reviewed action {row["game"]}:{index}')
            action_hashes.add(binding)
            before = step['observation']
    summary = {'sources': len(review['rows']),
        'action_targets': len(action_hashes),
        'families': dict(sorted(Counter(row['family']
            for row in review['rows']).items())),
        'mean_steps': sum(row['steps'] for row in review['rows']) /
            len(review['rows']),
        'source_arm': 'base', 'source_reward': 57}
    result = {'protocol': 'Read-only independent lineage audit of replay-reviewed self-success training bank and every fresh input-content-bound action annotation',
        'rollouts_sha256': file_hash(args.rollouts),
        'review_sha256': file_hash(args.review), 'summary': summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_sibling_train240_taskpair_current1000_20261006.json'))
    parser.add_argument('--review', type=Path, default=Path(
        'data/annotations/alf_self_base57_replay_reviewed_20261007.json'))
    parser.add_argument('--data-root', type=Path,
                        default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_self_base57_replay_reviewed_audited_20261007.json'))
    audit(parser.parse_args())
