"""Independently replay both arms of every frozen cumulative-prefix pair."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.audit_alf_action_sensitive_reward_v5 import replay
from ttcl.trajectory_hyperlora.freeze_alf_incremental_prefix_v18 import expected


def audit(args):
    if args.audit_output.exists():
        raise FileExistsError(args.audit_output)
    review = json.loads(args.prefix_review.read_text())
    report = json.loads(args.report.read_text())
    if (review != expected(args) or
            report['prefix_review_sha256'] != file_hash(args.prefix_review) or
            report['pool_review_sha256'] != file_hash(args.pool_review) or
            report['checkpoint_sha256'] != review['checkpoint_sha256'] or
            report['residual_sha256'] != review['residual_sha256'] or
            report['shard'] != args.shard or report['num_shards'] != 3 or
            report['max_steps'] != 50 or report['max_new_tokens'] != 64 or
            report['actor_history_turns'] != 2 or
            report['loop_guard_max'] != 2 or report['failures'] or
            len(report['rows']) != 104):
        raise ValueError('Changed or incomplete cumulative-prefix shard')
    schedule = [row for row in review['pairs']
                if row['target_id'] % 3 == args.shard]
    if len(schedule) != 104:
        raise ValueError('Changed frozen shard schedule')
    totals = defaultdict(lambda: {'pairs': 0, 'freeze': 0,
        'update': 0, 'update_only': 0, 'freeze_only': 0,
        'different_trajectories': 0})
    vectors = {}
    for position, (binding, row) in enumerate(zip(schedule, report['rows'], strict=True)):
        target = review['targets'][binding['target_id']]
        old_ids = binding['old_source_ids']
        new_id = binding['new_source_id']
        if (row['position'] != position or
                row['target_id'] != target['target_id'] or
                row['split'] != binding['split'] or
                row['family'] != target['family'] or
                row['transition_index'] != binding['transition_index'] or
                row['sequence'] != binding['sequence'] or
                row['old_source_ids'] != old_ids or
                row['new_source_id'] != new_id or
                row['old_prefix_count'] != len(old_ids) or
                row['game'] != target['game'] or
                row['input_content_sha256'] != binding['input_content_sha256'] or
                not all(math.isfinite(row[f'{arm}_code_norm'])
                    for arm in ('freeze', 'update'))):
            raise ValueError('Changed source prefix, target, or correction')
        vector_pair = (row['old_vector_sha256'],
                       row['updated_vector_sha256'])
        key = binding['transition_index']
        if key in vectors and vectors[key] != vector_pair:
            raise ValueError('Same prefix transition yielded different memory')
        vectors[key] = vector_pair
        initial = row['freeze']['initial_observation']
        game = args.data_root / target['game']
        for arm in ('freeze', 'update'):
            if (row[arm]['initial_observation'] != initial or
                    row[arm]['invalid_commands'] != 0):
                raise ValueError('Unmatched or invalid paired episode')
            replay(game, row[arm], initial)
        item = totals[(binding['split'], len(old_ids))]
        item['pairs'] += 1
        item['freeze'] += row['freeze']['reward']
        item['update'] += row['update']['reward']
        item['update_only'] += (row['update']['reward'] >
                                row['freeze']['reward'])
        item['freeze_only'] += (row['update']['reward'] <
                                row['freeze']['reward'])
        item['different_trajectories'] += (
            row['update']['trajectory'] != row['freeze']['trajectory'])
    for split, count, expected_pairs in (
            ('train', 2, 40), ('train', 3, 20), ('train', 4, 20),
            ('dev', 2, 12), ('dev', 3, 6), ('dev', 4, 6)):
        if totals[(split, count)]['pairs'] != expected_pairs:
            raise ValueError('Changed split or prefix-length composition')
    value = {'protocol': 'Original-environment replay of every v18 full-prefix write/freeze arm',
        'prefix_review_sha256': file_hash(args.prefix_review),
        'raw_report_sha256': file_hash(args.report),
        'shard': args.shard, 'failed_replays': 0,
        'totals': {f'{split}_{count}to{count+1}': item
            for (split, count), item in sorted(totals.items())}}
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(value, ensure_ascii=False,
                                           indent=2) + '\n')
    print(json.dumps(value['totals']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--data-root', type=Path, default=Path(
        'ttcl/data/alfworld_delta'))
    parser.add_argument('--checkpoint', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--residual', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_20261007.pt'))
    parser.add_argument('--pool-review', type=Path, default=Path(
        'data/annotations/alf_incremental_pool78_v14_reviewed_20261008.json'))
    parser.add_argument('--prefix-review', type=Path, default=Path(
        'data/annotations/alf_incremental_prefix78_v18_reviewed_20261008.json'))
    parser.add_argument('--report', type=Path)
    parser.add_argument('--audit-output', type=Path)
    args = parser.parse_args()
    if args.report is None:
        args.report = Path(f'results/trajectory_hyperlora/alf_incremental_prefix78_v18_shard{args.shard}_20261008.json')
    if args.audit_output is None:
        args.audit_output = Path(f'results/trajectory_hyperlora/alf_incremental_prefix78_v18_shard{args.shard}_audited_20261008.json')
    audit(args)
