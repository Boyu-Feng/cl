"""Audited same-game comparison of own-history MemRL and v19 online arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.freeze_alf_onpolicy_v19 import METHODS


def summarize(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    review = json.loads(args.review.read_text())
    parent = json.loads(args.parent_review.read_text())
    prior_summary = json.loads(args.v19_summary.read_text())
    if (review['parent_review_sha256'] != file_hash(args.parent_review) or
            review['plan_sha256'] != file_hash(args.plan) or
            parent['plan_sha256'] != file_hash(args.plan) or
            prior_summary['review_sha256'] != file_hash(args.parent_review) or
            len(review['targets']) != 90 or
            len(prior_summary['raw_report_sha256']) != 12 or
            len(prior_summary['audit_sha256']) != 12):
        raise ValueError('Changed frozen v19/v20 official comparison')
    # Verify every v19 result and independent replay on which the comparison
    # is conditioned, rather than trusting a stale summary JSON alone.
    for name, expected_hash in prior_summary['raw_report_sha256'].items():
        if file_hash(args.results_root / f'{name}_20261008.json') != expected_hash:
            raise ValueError('Changed v19 paired-arm result')
    for name, expected_hash in prior_summary['audit_sha256'].items():
        if file_hash(args.results_root / f'{name}_audited_20261008.json') != expected_hash:
            raise ValueError('Changed v19 original-environment replay')
    per_order = {}
    report_hashes = []
    audit_hashes = []
    for order in range(3):
        stem = f'alf_onpolicy_memrl_v20_order{order}'
        raw = args.results_root / f'{stem}_20261008.json'
        audited = args.results_root / f'{stem}_audited_20261008.json'
        report = json.loads(raw.read_text())
        audit = json.loads(audited.read_text())
        if (report['review_sha256'] != file_hash(args.review) or
                report['order'] != order or report['failures'] or
                len(report['rows']) != 30 or
                audit['review_sha256'] != file_hash(args.review) or
                audit['raw_report_sha256'] != file_hash(raw) or
                audit['order'] != order or audit['episodes'] != 30 or
                audit['failed_replays'] != 0):
            raise ValueError('Missing complete audited native MemRL order')
        targets = [row for row in review['targets'] if row['order_id'] == order]
        if [row['target_binding_sha256'] for row in report['rows']] != [
                target['input_content_sha256'] for target in targets]:
            raise ValueError('Changed MemRL target chronology')
        report_hashes.append(file_hash(raw))
        audit_hashes.append(file_hash(audited))
        v19_rows = {}
        for method in METHODS:
            path = args.results_root / (
                f'alf_onpolicy_v19_order{order}_{method}_20261008.json')
            v19_rows[method] = json.loads(path.read_text())['rows']
            if ([row['game'] for row in v19_rows[method]] !=
                    [row['game'] for row in report['rows']]):
                raise ValueError('Mismatched official target game order')
        memrl_rewards = [int(row['episode']['reward'])
                         for row in report['rows']]
        by_family = {}
        for family in sorted({row['family'] for row in report['rows']}):
            rows = [row for row in report['rows'] if row['family'] == family]
            if len(rows) != 5:
                raise ValueError('Changed official family coverage')
            by_family[family] = sum(int(row['episode']['reward']) for row in rows)
        comparison = {}
        for method in METHODS:
            rewards = [int(row['episode']['reward'])
                       for row in v19_rows[method]]
            comparison[method] = {
                'memrl_only_success': sum(m > c for m, c in
                                          zip(memrl_rewards, rewards, strict=True)),
                'comparator_only_success': sum(m < c for m, c in
                                               zip(memrl_rewards, rewards, strict=True)),
                'both_success': sum(m and c for m, c in
                                    zip(memrl_rewards, rewards, strict=True))}
        per_order[str(order)] = {'memrl_successes': sum(memrl_rewards),
            'memrl_writes': sum(row['memory_write'] for row in report['rows']),
            'retrieval_hits': sum(bool(row['retrieved']['ids'])
                                  for row in report['rows']),
            'writer_calls': audit['writer_calls'],
            'writer_input_tokens': audit['writer_input_tokens'],
            'writer_output_tokens': audit['writer_output_tokens'],
            'retrieved_source_tokens_per_decision_upper_proxy': sum(
                row['retrieved']['tokens'] * row['episode']['steps']
                for row in report['rows']),
            'by_family': by_family,
            'against_v19': comparison}
    totals = {'memrl_successes': sum(
        per_order[str(order)]['memrl_successes'] for order in range(3)),
        'memrl_writes': sum(per_order[str(order)]['memrl_writes']
                            for order in range(3)),
        'per_order_memrl_successes': [per_order[str(order)]['memrl_successes']
                                      for order in range(3)],
        'v19_successes': {method: prior_summary['totals'][method]['successes']
                         for method in METHODS}}
    value = {'protocol': 'All three independently collected native MemRL orders and all twelve v19 arms original-environment audited; paired official games but method-specific histories; three orders are repetitions, no 90-independent-chain significance claim',
        'review_sha256': file_hash(args.review),
        'parent_review_sha256': file_hash(args.parent_review),
        'v19_summary_sha256': file_hash(args.v19_summary),
        'memrl_report_sha256': report_hashes,
        'memrl_audit_sha256': audit_hashes,
        'per_order': per_order, 'totals': totals}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, ensure_ascii=False,
                                      indent=2) + '\n')
    print(json.dumps(totals), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, default=Path(
        'ttcl/trajectory_hyperlora/alf_valid_unseen_multiorder_v10_plan.json'))
    parser.add_argument('--parent-review', type=Path, default=Path(
        'data/annotations/alf_onpolicy_multiorder_v19_reviewed_20261008.json'))
    parser.add_argument('--review', type=Path, default=Path(
        'data/annotations/alf_onpolicy_memrl_v20_reviewed_20261008.json'))
    parser.add_argument('--results-root', type=Path, default=Path(
        'results/trajectory_hyperlora'))
    parser.add_argument('--v19-summary', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_onpolicy_v19_summary_20261008.json'))
    parser.add_argument('--output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_onpolicy_memrl_v20_summary_20261008.json'))
    summarize(parser.parse_args())
