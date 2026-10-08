"""Replay and summarize first-action interventions in the original ALFWorld env."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.audit_alf_action_sensitive_reward_v5 import replay


def audit(args):
    if args.audit_output.exists():
        raise FileExistsError(args.audit_output)
    review = json.loads(args.review.read_text())
    parent = json.loads(args.parent.read_text())
    report = json.loads(args.report.read_text())
    if (review['parent_report_sha256'] != file_hash(args.parent) or
            review['parent_audit_sha256'] != file_hash(args.parent_audit) or
            report['review_sha256'] != file_hash(args.review) or
            report['checkpoint_sha256'] != parent['checkpoint_sha256'] or
            report['residual_sha256'] != parent['residual_sha256'] or
            len(review['targets']) != 16 or len(report['rows']) != 16 or
            report['failures']):
        raise ValueError('Changed or incomplete intervention lineage')
    result = {'pairs': 16, 'swapped_episodes': 32,
        'freeze_swapped_reward_changed': 0,
        'update_swapped_reward_changed': 0,
        'both_swaps_reverse_own_outcome': 0,
        'neither_swap_changes_own_outcome': 0,
        'benefit_pairs': 0, 'harm_pairs': 0,
        'failed_replays': 0}
    pairs = []
    for target, row in zip(review['targets'], report['rows'], strict=True):
        parent_row = parent['rows'][target['pair_index']]
        if (row['input_content_sha256'] != target['input_content_sha256'] or
                row['pair_index'] != target['pair_index'] or
                row['split'] != target['split'] or
                parent_row['input_content_sha256'] !=
                    target['parent_input_content_sha256'] or
                file_hash(args.data_root / parent_row['game']) !=
                    target['target_game_sha256'] or
                digest(parent_row['freeze']['trajectory']) !=
                    target['freeze_trajectory_sha256'] or
                digest(parent_row['update']['trajectory']) !=
                    target['update_trajectory_sha256'] or
                parent_row['freeze']['trajectory'][target['split']]['command'] !=
                    target['freeze_action'] or
                parent_row['update']['trajectory'][target['split']]['command'] !=
                    target['update_action']):
            raise ValueError('Changed reviewed parent pair')
        changed = {}
        for arm, opposite in (('freeze', 'update'), ('update', 'freeze')):
            value = row['arms'][arm]
            episode = value['swapped_episode']
            reference = parent_row[arm]
            split = target['split']
            alternate = target[f'{opposite}_action']
            if (value['original_reward'] != reference['reward'] or
                    value['swapped_reward'] != episode['reward'] or
                    value['alternate_action'] != alternate or
                    episode['trajectory'][:split] !=
                        reference['trajectory'][:split] or
                    episode['trajectory'][split]['command'] != alternate or
                    episode['initial_observation'] !=
                        reference['initial_observation']):
                raise ValueError('Changed fixed-policy action intervention')
            replay(args.data_root / parent_row['game'], episode,
                   reference['initial_observation'])
            changed[arm] = episode['reward'] != reference['reward']
            result[f'{arm}_swapped_reward_changed'] += changed[arm]
        result['both_swaps_reverse_own_outcome'] += all(changed.values())
        result['neither_swap_changes_own_outcome'] += not any(changed.values())
        result['benefit_pairs'] += parent_row['update']['reward'] > parent_row['freeze']['reward']
        result['harm_pairs'] += parent_row['update']['reward'] < parent_row['freeze']['reward']
        pairs.append({'pair_index': target['pair_index'], 'split': target['split'],
            'freeze_original': parent_row['freeze']['reward'],
            'update_original': parent_row['update']['reward'],
            'freeze_swapped': row['arms']['freeze']['swapped_reward'],
            'update_swapped': row['arms']['update']['swapped_reward']})
    value = {'protocol': 'Independent original-environment replay of all 32 first-action interventions; first split local effect under each frozen branch policy',
        'review_sha256': file_hash(args.review),
        'parent_report_sha256': file_hash(args.parent),
        'raw_report_sha256': file_hash(args.report),
        'summary': result, 'pairs': pairs}
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(value, ensure_ascii=False,
                                           indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--review', type=Path, default=Path('data/annotations/alf_incremental_action16_v12_reviewed_20261007.json'))
    parser.add_argument('--parent', type=Path, default=Path('results/trajectory_hyperlora/alf_incremental_update108_v9_20261007.json'))
    parser.add_argument('--parent-audit', type=Path, default=Path('results/trajectory_hyperlora/alf_incremental_update108_v9_audited_20261007.json'))
    parser.add_argument('--report', type=Path, default=Path('results/trajectory_hyperlora/alf_incremental_action16_v12_20261007.json'))
    parser.add_argument('--audit-output', type=Path, default=Path('results/trajectory_hyperlora/alf_incremental_action16_v12_audited_20261007.json'))
    audit(parser.parse_args())
