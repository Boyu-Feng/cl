"""Bind self-success histories to different-game future action targets.

The official-train game split is frozen before model training. Family metadata
is used only to sample related pairs and to report coverage; no family slot is
passed to a generator.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash


def prepare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    bank = json.loads(args.bank.read_text())
    audit = json.loads(args.bank_audit.read_text())
    if (audit['review_sha256'] != file_hash(args.bank) or
            bank['failures'] or len(bank['rows']) != 57 or
            audit['summary']['sources'] != 57 or
            audit['summary']['action_targets'] != 633):
        raise ValueError('Changed self-verified trajectory bank')
    families = defaultdict(list)
    for row in bank['rows']:
        families[row['family']].append(row)
    train, dev = [], []
    for family, rows in sorted(families.items()):
        ordered = sorted(rows, key=lambda row: digest([
            'future_pairs_split_v1', row['game_sha256'],
            row['episode_sha256']]))
        n_dev = max(1, round(len(rows) * .2)) if len(rows) >= 4 else 0
        dev.extend(ordered[:n_dev])
        train.extend(ordered[n_dev:])
    if (set(x['game'] for x in train) & set(x['game'] for x in dev) or
            len(train) + len(dev) != 57):
        raise ValueError('Game split leaked')
    by_family = defaultdict(list)
    for source in train:
        by_family[source['family']].append(source)
    pairs = []
    for split, targets in (('train', train), ('dev', dev)):
        for target in targets:
            eligible = [row for row in by_family[target['family']]
                        if row['game'] != target['game']]
            if not eligible:
                continue
            sources = sorted(eligible, key=lambda row: digest([
                'future_pairs_source_v1', split,
                target['game_sha256'], row['game_sha256']]))[:2]
            steps = target['reviewed_targets']
            turns = sorted({0, len(steps) // 2, len(steps) - 1})
            for source in sources:
                for turn in turns:
                    action = steps[turn]
                    content = {'bank_review_sha256': file_hash(args.bank),
                        'split': split,
                        'source_game_sha256': source['game_sha256'],
                        'source_episode_sha256': source['episode_sha256'],
                        'source_records_sha256': source['records_sha256'],
                        'target_game_sha256': target['game_sha256'],
                        'target_episode_sha256': target['episode_sha256'],
                        'target_action_input_sha256':
                            action['input_content_sha256']}
                    pairs.append({'split': split,
                        'source_game': source['game'],
                        'source_game_sha256': source['game_sha256'],
                        'source_episode_sha256': source['episode_sha256'],
                        'source_records_sha256': source['records_sha256'],
                        'target_game': target['game'],
                        'target_game_sha256': target['game_sha256'],
                        'target_episode_sha256': target['episode_sha256'],
                        'target_turn': turn,
                        'target_action_input_sha256':
                            action['input_content_sha256'],
                        'input_content_sha256': digest(content)})
    report = {'protocol': 'Content-bound same-family but different-game official train self-success history to future successful action; disjoint train/dev target games; family used only for sampling; no manual action slots',
        'bank_review_sha256': file_hash(args.bank),
        'bank_audit_sha256': file_hash(args.bank_audit),
        'train_games': sorted(x['game'] for x in train),
        'dev_games': sorted(x['game'] for x in dev),
        'pairs': pairs,
        'unpaired_train_games': sorted(x['game'] for x in train
            if len(by_family[x['family']]) < 2)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'train_games': len(train), 'dev_games': len(dev),
        'train_pairs': sum(x['split']=='train' for x in pairs),
        'dev_pairs': sum(x['split']=='dev' for x in pairs),
        'unpaired_train_games': len(report['unpaired_train_games'])}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank', type=Path, default=Path(
        'data/annotations/alf_self_base57_replay_reviewed_20261007.json'))
    parser.add_argument('--bank-audit', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_self_base57_replay_reviewed_audited_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path(
        'data/annotations/alf_self_future_pairs_reviewed_20261007.json'))
    prepare(parser.parse_args())
