"""Pair base-success histories with different own-only-success train actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash


def prepare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    source_bank = json.loads(args.source_bank.read_text())
    source_audit = json.loads(args.source_audit.read_text())
    target_bank = json.loads(args.target_bank.read_text())
    target_audit = json.loads(args.target_audit.read_text())
    if (source_audit['review_sha256'] != file_hash(args.source_bank) or
            target_audit['review_sha256'] != file_hash(args.target_bank) or
            source_bank['failures'] or target_bank['failures'] or
            len(source_bank['rows']) != 57 or
            len(target_bank['rows']) != 42 or
            target_bank['source_arm'] != 'own' or
            target_bank['paired_base_reward'] != 0):
        raise ValueError('Changed reviewed source or own-only target bank')
    pairs = []
    for target in target_bank['rows']:
        sources = sorted((row for row in source_bank['rows']
            if (row['family'] == target['family'] and
                row['game'] != target['game'])),
            key=lambda row: digest(['own_only_source_v2',
                target['game_sha256'], row['game_sha256']]))[:2]
        if not sources:
            raise ValueError('Own-only successful target lacks any related source')
        actions = target['reviewed_targets']
        turns = sorted({0, len(actions)//2, len(actions)-1})
        for source in sources:
            for turn in turns:
                action = actions[turn]
                content = {'source_bank_sha256': file_hash(args.source_bank),
                    'target_bank_sha256': file_hash(args.target_bank),
                    'source_game_sha256': source['game_sha256'],
                    'source_episode_sha256': source['episode_sha256'],
                    'source_records_sha256': source['records_sha256'],
                    'target_game_sha256': target['game_sha256'],
                    'target_episode_sha256': target['episode_sha256'],
                    'target_action_input_sha256':
                        action['input_content_sha256']}
                pairs.append({'source_game': source['game'],
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
    result = {'protocol': 'Train-only content-bound different-game base-success source to old-LoRA-own-success and paired-base-failure future action; six categories represented; target actor had expert-sibling LoRA; no independent development targets or environment reward claim',
        'source_bank_sha256': file_hash(args.source_bank),
        'source_audit_sha256': file_hash(args.source_audit),
        'target_bank_sha256': file_hash(args.target_bank),
        'target_audit_sha256': file_hash(args.target_audit),
        'source_count': 57, 'target_count': 42,
        'pairs': pairs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'sources': 57, 'targets': 42,
                      'pairs': len(pairs)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-bank', type=Path, default=Path(
        'data/annotations/alf_self_base57_replay_reviewed_20261007.json'))
    parser.add_argument('--source-audit', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_self_base57_replay_reviewed_audited_20261007.json'))
    parser.add_argument('--target-bank', type=Path, default=Path(
        'data/annotations/alf_own_only42_replay_reviewed_20261007.json'))
    parser.add_argument('--target-audit', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_own_only42_replay_reviewed_audited_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path(
        'data/annotations/alf_own_only_future_pairs_reviewed_20261007.json'))
    prepare(parser.parse_args())
