"""Audit disjoint different-game self-history to future-action annotations."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash


def audit(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    bank = json.loads(args.bank.read_text())
    plan = json.loads(args.pairs.read_text())
    parent = json.loads(args.bank_audit.read_text())
    if (plan['bank_review_sha256'] != file_hash(args.bank) or
            plan['bank_audit_sha256'] != file_hash(args.bank_audit) or
            parent['review_sha256'] != file_hash(args.bank) or
            len(bank['rows']) != 57 or bank['failures']):
        raise ValueError('Changed replay-reviewed source bank')
    rows = {row['game']: row for row in bank['rows']}
    train, dev = set(plan['train_games']), set(plan['dev_games'])
    if (train & dev or train | dev != set(rows) or
            len(train) != 46 or len(dev) != 11):
        raise ValueError('Invalid disjoint target split')
    seen = set()
    counts = Counter()
    for pair in plan['pairs']:
        source = rows[pair['source_game']]
        target = rows[pair['target_game']]
        split = pair['split']
        if (source['game'] not in train or
                target['game'] not in (train if split == 'train' else dev) or
                source['game'] == target['game'] or
                source['family'] != target['family'] or
                not 0 <= pair['target_turn'] < target['steps']):
            raise ValueError('Unrelated or leaking source-target pair')
        action = target['reviewed_targets'][pair['target_turn']]
        content = {'bank_review_sha256': file_hash(args.bank),
            'split': split,
            'source_game_sha256': source['game_sha256'],
            'source_episode_sha256': source['episode_sha256'],
            'source_records_sha256': source['records_sha256'],
            'target_game_sha256': target['game_sha256'],
            'target_episode_sha256': target['episode_sha256'],
            'target_action_input_sha256': action['input_content_sha256']}
        expected = {'source_game_sha256': source['game_sha256'],
            'source_episode_sha256': source['episode_sha256'],
            'source_records_sha256': source['records_sha256'],
            'target_game_sha256': target['game_sha256'],
            'target_episode_sha256': target['episode_sha256'],
            'target_action_input_sha256': action['input_content_sha256'],
            'input_content_sha256': digest(content)}
        if any(pair[key] != value for key, value in expected.items()):
            raise ValueError('Changed new input-content binding')
        if pair['input_content_sha256'] in seen:
            raise ValueError('Duplicate pair binding')
        seen.add(pair['input_content_sha256'])
        counts[split] += 1
    summary = {'train_games': len(train), 'dev_games': len(dev),
        'train_pairs': counts['train'], 'dev_pairs': counts['dev'],
        'unpaired_train_games': len(plan['unpaired_train_games']),
        'train_families': dict(sorted(Counter(rows[g]['family']
            for g in train).items())),
        'dev_families': dict(sorted(Counter(rows[g]['family']
            for g in dev).items()))}
    result = {'protocol': 'Read-only audit of fresh future-action pair content bindings and disjoint official train game split',
        'bank_review_sha256': file_hash(args.bank),
        'pairs_review_sha256': file_hash(args.pairs),
        'summary': summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank', type=Path, default=Path(
        'data/annotations/alf_self_base57_replay_reviewed_20261007.json'))
    parser.add_argument('--bank-audit', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_self_base57_replay_reviewed_audited_20261007.json'))
    parser.add_argument('--pairs', type=Path, default=Path(
        'data/annotations/alf_self_future_pairs_reviewed_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_self_future_pairs_reviewed_audited_20261007.json'))
    audit(parser.parse_args())
