"""Audited paired-reward summary with target-cluster uncertainty.

Each official game contributes six correlated source transitions. The
stratified bootstrap resamples games within family, not individual rollouts.
It describes this fixed source bank and local train/dev target pool only.
"""
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
    review = json.loads(args.pool_review.read_text())
    rows = {}
    reports = []
    audits = []
    for shard in range(3):
        report_path = args.results_root / f'alf_incremental_pool78_v14_shard{shard}_20261008.json'
        audit_path = args.results_root / f'alf_incremental_pool78_v14_shard{shard}_audited_20261008.json'
        report = json.loads(report_path.read_text())
        audit = json.loads(audit_path.read_text())
        if (report['pool_review_sha256'] != file_hash(args.pool_review) or
                audit['pool_review_sha256'] != file_hash(args.pool_review) or
                audit['raw_report_sha256'] != file_hash(report_path) or
                audit['failed_replays'] != 0 or
                report['failures'] or len(report['rows']) != 156):
            raise ValueError('Missing original-environment audited shard')
        reports.append(file_hash(report_path))
        audits.append(file_hash(audit_path))
        for row in report['rows']:
            key = row['input_content_sha256']
            if key in rows:
                raise ValueError('Duplicate pair in completed shards')
            rows[key] = row
    if len(rows) != 468:
        raise ValueError('Incomplete 78-game by 6-transition matrix')
    per_target = defaultdict(list)
    per_transition = defaultdict(list)
    for binding in review['pairs']:
        row = rows[binding['input_content_sha256']]
        target = review['targets'][binding['target_id']]
        if (row['target_id'] != target['target_id'] or
                row['transition_index'] != binding['transition_index'] or
                row['split'] != target['split'] or
                row['family'] != target['family']):
            raise ValueError('Changed reviewed pair identity')
        f, u = row['freeze']['reward'], row['update']['reward']
        item = {'freeze': f, 'update': u, 'difference': u-f,
                'transition_index': binding['transition_index']}
        per_target[target['target_id']].append(item)
        per_transition[(target['split'], binding['transition_index'])].append(item)
    if set(per_target) != set(range(78)) or any(len(v) != 6 for v in per_target.values()):
        raise ValueError('Target matrix has missing source transitions')
    summaries = {}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for split in ('train', 'dev'):
        targets = [row for row in review['targets'] if row['split'] == split]
        records = [item for target in targets
                   for item in per_target[target['target_id']]]
        family_values = {}
        family_summaries = {}
        for family in sorted({row['family'] for row in targets}):
            family_targets = [row for row in targets if row['family'] == family]
            values = np.asarray([np.mean([item['difference'] for item in
                per_target[target['target_id']]]) for target in family_targets],
                dtype='float64')
            family_values[family] = values
            family_records = [item for target in family_targets
                              for item in per_target[target['target_id']]]
            family_summaries[family] = {'targets': len(family_targets),
                'freeze': sum(item['freeze'] for item in family_records),
                'update': sum(item['update'] for item in family_records),
                'update_only': sum(item['difference'] > 0
                                   for item in family_records),
                'freeze_only': sum(item['difference'] < 0
                                   for item in family_records)}
        family_names = sorted(family_values)
        draws = np.empty(BOOTSTRAP_REPLICATES, dtype='float64')
        for index in range(BOOTSTRAP_REPLICATES):
            draws[index] = np.mean([rng.choice(family_values[family],
                size=len(family_values[family]), replace=True).mean()
                for family in family_names])
        summaries[split] = {'targets': len(targets), 'pairs': len(records),
            'freeze': sum(item['freeze'] for item in records),
            'update': sum(item['update'] for item in records),
            'update_only': sum(item['difference'] > 0 for item in records),
            'freeze_only': sum(item['difference'] < 0 for item in records),
            'mean_update_minus_freeze_per_pair':
                float(np.mean([item['difference'] for item in records])),
            'target_cluster_family_stratified_95pct_interval':
                [float(x) for x in np.quantile(draws, [.025, .975])],
            'per_family': family_summaries,
            'per_transition': {str(index): {
                'freeze': sum(item['freeze'] for item in
                              per_transition[(split, index)]),
                'update': sum(item['update'] for item in
                              per_transition[(split, index)])}
                for index in range(6)}}
    value = {'protocol': 'Audited fixed source-transition train/dev outcome summary; 10000 target-game cluster resamples within six families, fixed source bank and declared splits; no independent official test claim',
        'pool_review_sha256': file_hash(args.pool_review),
        'report_sha256': reports, 'audit_sha256': audits,
        'bootstrap_seed': BOOTSTRAP_SEED,
        'bootstrap_replicates': BOOTSTRAP_REPLICATES,
        'summary': summaries}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({name: {key: val for key, val in group.items()
        if key not in ('per_family', 'per_transition')}
        for name, group in summaries.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool-review', type=Path, default=Path(
        'data/annotations/alf_incremental_pool78_v14_reviewed_20261008.json'))
    parser.add_argument('--results-root', type=Path,
                        default=Path('results/trajectory_hyperlora'))
    parser.add_argument('--output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_incremental_pool78_v14_summary_20261008.json'))
    summarize(parser.parse_args())
