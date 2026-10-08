"""Independent original-environment replay of one v14 target shard."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.audit_alf_action_sensitive_reward_v5 import replay


def audit(args):
    if args.audit_output.exists():
        raise FileExistsError(args.audit_output)
    review = json.loads(args.pool_review.read_text())
    report = json.loads(args.report.read_text())
    if (report['pool_review_sha256'] != file_hash(args.pool_review) or
            report['checkpoint_sha256'] != review['checkpoint_sha256'] or
            report['residual_sha256'] != review['residual_sha256'] or
            report['shard'] != args.shard or report['num_shards'] != 3 or
            report['max_steps'] != 50 or report['max_new_tokens'] != 64 or
            report['actor_history_turns'] != 2 or
            report['loop_guard_max'] != 2 or report['failures'] or
            len(report['rows']) != 156):
        raise ValueError('Changed or incomplete frozen target shard')
    schedule = [row for row in review['pairs']
                if row['target_id'] % 3 == args.shard]
    if len(schedule) != 156:
        raise ValueError('Changed pair schedule')
    totals = defaultdict(lambda: {'pairs': 0, 'freeze': 0,
        'update': 0, 'update_only': 0, 'freeze_only': 0,
        'different_trajectories': 0})
    vectors = {}
    for position, (binding, row) in enumerate(zip(schedule, report['rows'], strict=True)):
        target = review['targets'][binding['target_id']]
        old_id, new_id = review['transitions'][binding['transition_index']]
        content = {'method': 'incremental-v14',
            'target_id': target['target_id'], 'split': target['split'],
            'target_game_sha256': target['game_sha256'],
            'transition_index': binding['transition_index'],
            'old_records_sha256': review['sources'][old_id]['records_sha256'],
            'new_records_sha256': review['sources'][new_id]['records_sha256'],
            'checkpoint_sha256': review['checkpoint_sha256'],
            'residual_sha256': review['residual_sha256'],
            'alpha': [0., .5], 'max_steps': 50}
        if (binding != {**content, 'input_content_sha256': digest(content),
                        'reviewed_target': True} or
                row['position'] != position or
                row['target_id'] != target['target_id'] or
                row['split'] != target['split'] or
                row['family'] != target['family'] or
                row['transition_index'] != binding['transition_index'] or
                row['old_source_id'] != old_id or
                row['new_source_id'] != new_id or
                row['game'] != target['game'] or
                row['input_content_sha256'] != binding['input_content_sha256'] or
                file_hash(args.data_root / target['game']) !=
                    target['game_sha256'] or
                any(not math.isfinite(row[f'{arm}_code_norm'])
                    for arm in ('freeze', 'update'))):
            raise ValueError('Changed bound target, source, or correction')
        vector_key = binding['transition_index']
        vector_pair = (row['old_vector_sha256'],
                       row['updated_vector_sha256'])
        if vector_key in vectors and vectors[vector_key] != vector_pair:
            raise ValueError('Same source transition yielded different memory')
        vectors[vector_key] = vector_pair
        initial = row['freeze']['initial_observation']
        for arm in ('freeze', 'update'):
            if (row[arm]['initial_observation'] != initial or
                    row[arm]['invalid_commands'] != 0):
                raise ValueError('Unmatched or invalid paired episode')
            replay(args.data_root / target['game'], row[arm], initial)
        item = totals[target['split']]
        item['pairs'] += 1
        item['freeze'] += row['freeze']['reward']
        item['update'] += row['update']['reward']
        item['update_only'] += row['update']['reward'] > row['freeze']['reward']
        item['freeze_only'] += row['update']['reward'] < row['freeze']['reward']
        item['different_trajectories'] += (
            row['update']['trajectory'] != row['freeze']['trajectory'])
    if (totals['train']['pairs'] != 120 or
            totals['dev']['pairs'] != 36):
        raise ValueError('Changed train/dev composition')
    value = {'protocol': 'Original-environment replay of every frozen v14 shard episode, content-bound target/source/LoRA audit',
        'pool_review_sha256': file_hash(args.pool_review),
        'raw_report_sha256': file_hash(args.report),
        'shard': args.shard, 'failed_replays': 0,
        'totals': dict(totals)}
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(value, ensure_ascii=False,
                                           indent=2) + '\n')
    print(json.dumps(value['totals']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--data-root', type=Path,
                        default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--pool-review', type=Path, default=Path(
        'data/annotations/alf_incremental_pool78_v14_reviewed_20261008.json'))
    parser.add_argument('--report', type=Path)
    parser.add_argument('--audit-output', type=Path)
    args = parser.parse_args()
    if args.report is None:
        args.report = Path(f'results/trajectory_hyperlora/alf_incremental_pool78_v14_shard{args.shard}_20261008.json')
    if args.audit_output is None:
        args.audit_output = Path(f'results/trajectory_hyperlora/alf_incremental_pool78_v14_shard{args.shard}_audited_20261008.json')
    audit(args)
