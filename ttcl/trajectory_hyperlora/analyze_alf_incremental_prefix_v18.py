"""Audited full-prefix write effects with target-cluster uncertainty."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash


BOOTSTRAP_SEED = 20261008
BOOTSTRAP_REPLICATES = 10000


def summarize(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    review = json.loads(args.prefix_review.read_text())
    if (len(review['targets']) != 78 or len(review['pairs']) != 312 or
            review['v14_pool_review_sha256'] != file_hash(args.pool_review)):
        raise ValueError('Changed frozen full-prefix target pool')
    rows = {}
    report_hashes = []
    audit_hashes = []
    for shard in range(3):
        raw = args.results_root / f'alf_incremental_prefix78_v18_shard{shard}_20261008.json'
        audited = args.results_root / f'alf_incremental_prefix78_v18_shard{shard}_audited_20261008.json'
        report = json.loads(raw.read_text())
        audit = json.loads(audited.read_text())
        if (report['prefix_review_sha256'] != file_hash(args.prefix_review) or
                audit['prefix_review_sha256'] != file_hash(args.prefix_review) or
                audit['raw_report_sha256'] != file_hash(raw) or
                audit['failed_replays'] != 0 or
                report['failures'] or len(report['rows']) != 104):
            raise ValueError('Missing complete audited full-prefix shard')
        report_hashes.append(file_hash(raw))
        audit_hashes.append(file_hash(audited))
        for row in report['rows']:
            key = row['input_content_sha256']
            if key in rows:
                raise ValueError('Duplicate prefix pair')
            rows[key] = row
    if len(rows) != 312:
        raise ValueError('Incomplete full-prefix label matrix')
    per_target = defaultdict(dict)
    per_transition = defaultdict(list)
    for binding in review['pairs']:
        row = rows.get(binding['input_content_sha256'])
        target = review['targets'][binding['target_id']]
        if (row is None or row['target_id'] != target['target_id'] or
                row['split'] != binding['split'] or
                row['family'] != target['family'] or
                row['transition_index'] != binding['transition_index'] or
                row['old_source_ids'] != binding['old_source_ids'] or
                row['new_source_id'] != binding['new_source_id']):
            raise ValueError('Changed reviewed prefix/source/target identity')
        reward_freeze = row['freeze']['reward']
        reward_update = row['update']['reward']
        item = {'freeze': reward_freeze, 'update': reward_update,
            'difference': reward_update-reward_freeze,
            'trajectory_changed': (row['freeze']['trajectory'] !=
                                   row['update']['trajectory'])}
        target_id = binding['target_id']
        transition = binding['transition_index']
        if transition in per_target[target_id]:
            raise ValueError('Duplicate target/transition')
        per_target[target_id][transition] = item
        per_transition[(binding['split'], transition)].append(item)
    if set(per_target) != set(range(78)) or any(set(group) != set(range(4))
                                               for group in per_target.values()):
        raise ValueError('Incomplete four-transition target clusters')
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summaries = {}
    for split in ('train', 'dev'):
        targets = [row for row in review['targets'] if row['split'] == split]
        families = sorted(set(row['family'] for row in targets))
        if len(families) != 6:
            raise ValueError('Changed official family coverage')
        family_differences = {}
        family_stats = {}
        for family in families:
            family_targets = [row for row in targets
                              if row['family'] == family]
            expected_n = 10 if split == 'train' else 3
            if len(family_targets) != expected_n:
                raise ValueError('Changed per-family train/dev size')
            differences = np.asarray([
                [per_target[target['target_id']][transition]['difference']
                 for transition in range(4)]
                for target in family_targets], dtype='float64')
            family_differences[family] = differences
            records = [per_target[target['target_id']][transition]
                       for target in family_targets for transition in range(4)]
            family_stats[family] = {'targets': len(family_targets),
                'pairs': len(records),
                'freeze': sum(item['freeze'] for item in records),
                'update': sum(item['update'] for item in records),
                'update_only': sum(item['difference'] > 0 for item in records),
                'freeze_only': sum(item['difference'] < 0 for item in records)}
        # Index 0 and 3 are independent source sequences at 2->3; the
        # 3->4 and 4->5 columns both come from sequence 0. Report this
        # confounding instead of treating prefix length as randomized.
        by_prefix = {'2to3': (0, 3), '3to4': (1,), '4to5': (2,)}
        draws = {key: np.empty(BOOTSTRAP_REPLICATES, dtype='float64')
                 for key in ('overall', *by_prefix)}
        for i in range(BOOTSTRAP_REPLICATES):
            sampled = []
            for family in families:
                values = family_differences[family]
                chosen = rng.integers(0, len(values), size=len(values))
                sampled.append(values[chosen].mean(axis=0))
            mean_by_transition = np.mean(sampled, axis=0)
            draws['overall'][i] = mean_by_transition.mean()
            for prefix, transitions in by_prefix.items():
                draws[prefix][i] = mean_by_transition[list(transitions)].mean()
        all_records = [item for target in targets
                       for item in per_target[target['target_id']].values()]
        prefix_stats = {}
        for prefix, transitions in by_prefix.items():
            records = [per_target[target['target_id']][transition]
                       for target in targets for transition in transitions]
            prefix_stats[prefix] = {
                'pairs': len(records),
                'freeze': sum(item['freeze'] for item in records),
                'update': sum(item['update'] for item in records),
                'update_only': sum(item['difference'] > 0 for item in records),
                'freeze_only': sum(item['difference'] < 0 for item in records),
                'trajectory_changed': sum(item['trajectory_changed']
                                          for item in records),
                'mean_update_minus_freeze': float(np.mean(
                    [item['difference'] for item in records])),
                'target_cluster_family_stratified_95pct_interval':
                    [float(x) for x in np.quantile(draws[prefix], [.025, .975])]}
        summaries[split] = {'targets': len(targets),
            'pairs': len(all_records),
            'freeze': sum(item['freeze'] for item in all_records),
            'update': sum(item['update'] for item in all_records),
            'update_only': sum(item['difference'] > 0 for item in all_records),
            'freeze_only': sum(item['difference'] < 0 for item in all_records),
            'trajectory_changed': sum(item['trajectory_changed']
                                      for item in all_records),
            'mean_update_minus_freeze': float(np.mean(
                [item['difference'] for item in all_records])),
            'target_cluster_family_stratified_95pct_interval':
                [float(x) for x in np.quantile(draws['overall'], [.025, .975])],
            'by_prefix': prefix_stats,
            'by_transition': {str(transition): {
                'freeze': sum(item['freeze'] for item in
                              per_transition[(split, transition)]),
                'update': sum(item['update'] for item in
                              per_transition[(split, transition)])}
                for transition in range(4)},
            'by_family': family_stats}
    value = {'protocol': 'Predeclared audited v18 full-prefix source write/freeze summary; 10000 target-cluster family-stratified resamples, fixed source bank; prefix length confounded with source sequence; train/dev descriptive only',
        'prefix_review_sha256': file_hash(args.prefix_review),
        'report_sha256': report_hashes,
        'audit_sha256': audit_hashes,
        'bootstrap_seed': BOOTSTRAP_SEED,
        'bootstrap_replicates': BOOTSTRAP_REPLICATES,
        'summary': summaries}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, ensure_ascii=False,
                                      indent=2) + '\n')
    print(json.dumps({split: {key: group[key] for key in (
        'targets', 'pairs', 'freeze', 'update', 'update_only', 'freeze_only',
        'mean_update_minus_freeze')}
        for split, group in summaries.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool-review', type=Path, default=Path(
        'data/annotations/alf_incremental_pool78_v14_reviewed_20261008.json'))
    parser.add_argument('--prefix-review', type=Path, default=Path(
        'data/annotations/alf_incremental_prefix78_v18_reviewed_20261008.json'))
    parser.add_argument('--results-root', type=Path, default=Path(
        'results/trajectory_hyperlora'))
    parser.add_argument('--output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_incremental_prefix78_v18_summary_20261008.json'))
    summarize(parser.parse_args())
