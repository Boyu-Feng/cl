"""Descriptive read-opportunity audit; does not replace environment replay."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def analyze(report):
    counts = {}
    reads = []
    scores = {arm: 0 for arm in ('base', 'frozen', 'persistent', 'direct')}
    args = report['run_binding']['arguments']
    if not args.get('hindsight_basis'):
        raise ValueError('audit currently covers the family-keyed hindsight controller only')
    for index, row in enumerate(report['rows']):
        family = Path(row['game']).parts[-3].split('-')[0]
        for arm in scores:
            scores[arm] += row[arm]['reward']
        if counts.get(family, 0):
            actions = lambda arm: [s['command'] for s in row[arm]['trajectory']]
            reads.append({'index': index + 1, 'family': family,
                          'prior_writes': counts[family],
                          'actions_differ_from_frozen': actions('persistent') != actions('frozen'),
                          'reward_difference': row['persistent']['reward'] - row['frozen']['reward']})
        update = row['update']
        if update.get('persistent'):
            counts[family] = counts.get(family, 0) + 1
    return {'completed': len(report['rows']), 'scores': scores,
            'writes_by_family': counts, 'post_write_reads': reads,
            'post_write_read_count': len(reads),
            'post_write_action_changes': sum(r['actions_differ_from_frozen'] for r in reads),
            'post_write_net_success': sum(r['reward_difference'] for r in reads),
            'limitation': 'Log-derived diagnostic; no independent environment replay or saved-state verification.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    result = analyze(json.loads(raw))
    result['report_sha256'] = hashlib.sha256(raw).hexdigest()
    result['report_path'] = str(args.report)
    with args.output.open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'post_write_reads'}))
