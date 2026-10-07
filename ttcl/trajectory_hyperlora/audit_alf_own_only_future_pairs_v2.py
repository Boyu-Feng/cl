"""Audit all different-game own-only-success future-action bindings."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash


def audit(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads(args.source_bank.read_text())
    target = json.loads(args.target_bank.read_text())
    plan = json.loads(args.pairs.read_text())
    if (plan['source_bank_sha256'] != file_hash(args.source_bank) or
            plan['target_bank_sha256'] != file_hash(args.target_bank) or
            source['failures'] or target['failures'] or
            len(source['rows']) != 57 or len(target['rows']) != 42 or
            len(plan['pairs']) != 234):
        raise ValueError('Changed self-success future-pair inputs')
    sources = {row['game']: row for row in source['rows']}
    targets = {row['game']: row for row in target['rows']}
    seen = set()
    families = Counter()
    for pair in plan['pairs']:
        s = sources[pair['source_game']]
        t = targets[pair['target_game']]
        if (s['game'] == t['game'] or s['family'] != t['family'] or
                not 0 <= pair['target_turn'] < t['steps']):
            raise ValueError('Invalid own-only source-target pairing')
        action = t['reviewed_targets'][pair['target_turn']]
        content = {'source_bank_sha256': file_hash(args.source_bank),
            'target_bank_sha256': file_hash(args.target_bank),
            'source_game_sha256': s['game_sha256'],
            'source_episode_sha256': s['episode_sha256'],
            'source_records_sha256': s['records_sha256'],
            'target_game_sha256': t['game_sha256'],
            'target_episode_sha256': t['episode_sha256'],
            'target_action_input_sha256': action['input_content_sha256']}
        expected = {'source_game_sha256': s['game_sha256'],
            'source_episode_sha256': s['episode_sha256'],
            'source_records_sha256': s['records_sha256'],
            'target_game_sha256': t['game_sha256'],
            'target_episode_sha256': t['episode_sha256'],
            'target_action_input_sha256': action['input_content_sha256'],
            'input_content_sha256': digest(content)}
        if (any(pair[key] != value for key, value in expected.items()) or
                pair['input_content_sha256'] in seen):
            raise ValueError('Changed or duplicate own-only action binding')
        seen.add(pair['input_content_sha256'])
        families[t['family']] += 1
    summary = {'sources': 57, 'targets': 42,
        'different_game_action_pairs': len(seen),
        'target_families': dict(sorted(families.items())),
        'target_arm': 'own', 'paired_base_reward': 0,
        'checkpoint_ancestor_exposed': True}
    result = {'protocol': 'Content-bound audit of official train own-LoRA-success versus base-failure future-action pairs; not independent of ancestor checkpoint',
        'source_bank_sha256': file_hash(args.source_bank),
        'target_bank_sha256': file_hash(args.target_bank),
        'pairs_review_sha256': file_hash(args.pairs),
        'summary': summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-bank', type=Path, default=Path(
        'data/annotations/alf_self_base57_replay_reviewed_20261007.json'))
    parser.add_argument('--target-bank', type=Path, default=Path(
        'data/annotations/alf_own_only42_replay_reviewed_20261007.json'))
    parser.add_argument('--pairs', type=Path, default=Path(
        'data/annotations/alf_own_only_future_pairs_reviewed_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_own_only_future_pairs_audited_20261007.json'))
    audit(parser.parse_args())
