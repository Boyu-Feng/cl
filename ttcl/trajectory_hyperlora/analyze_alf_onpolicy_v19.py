"""Audited paired-game summary for independently evolving online arms."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.freeze_alf_onpolicy_v19 import METHODS


def summarize(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    review = json.loads(args.review.read_text())
    if (review['plan_sha256'] != file_hash(args.plan) or
            review['methods'] != list(METHODS) or
            len(review['targets']) != 360):
        raise ValueError('Changed frozen online comparison')
    reports = {}
    report_hashes = {}
    audit_hashes = {}
    for order in range(3):
        for method in METHODS:
            name = f'alf_onpolicy_v19_order{order}_{method}'
            raw = args.results_root / f'{name}_20261008.json'
            audited = args.results_root / f'{name}_audited_20261008.json'
            report = json.loads(raw.read_text())
            audit = json.loads(audited.read_text())
            if (report['review_sha256'] != file_hash(args.review) or
                    audit['review_sha256'] != file_hash(args.review) or
                    audit['raw_report_sha256'] != file_hash(raw) or
                    audit['failed_replays'] != 0 or
                    report['order'] != order or report['method'] != method or
                    report['failures'] or len(report['rows']) != 30 or
                    audit['episodes'] != 30):
                raise ValueError('Missing complete audited online arm')
            expected_targets = [target for target in review['targets']
                if target['order_id'] == order and
                   target['method'] == method]
            if ([row['target_binding_sha256'] for row in report['rows']] !=
                    [target['input_content_sha256'] for target in expected_targets]):
                raise ValueError('Changed online target order')
            reports[(order, method)] = report['rows']
            report_hashes[name] = file_hash(raw)
            audit_hashes[name] = file_hash(audited)
    per_order = {}
    discordance = defaultdict(dict)
    for order in range(3):
        by_method = {}
        for method in METHODS:
            rows = reports[(order, method)]
            successes = [int(row['episode']['reward']) for row in rows]
            writes = sum(row['memory_write'] for row in rows)
            first_success = next((i for i, value in enumerate(successes)
                                  if value), None)
            by_family = {}
            for family in sorted({row['family'] for row in rows}):
                family_rows = [row for row in rows if row['family'] == family]
                if len(family_rows) != 5:
                    raise ValueError('Changed five-per-family official order')
                by_family[family] = sum(int(row['episode']['reward'])
                                        for row in family_rows)
            by_method[method] = {'successes': sum(successes),
                'writes': writes, 'first_success_position': first_success,
                'by_family': by_family,
                'source_encoding_tokens_retained_on_writes': sum(
                    row.get('source_tokens_retained', 0) for row in rows),
                'raw_text_source_tokens_per_decision_upper_proxy': sum(
                    row['raw_text_retained_tokens'] * row['episode']['steps']
                    for row in rows)}
        for comparator in ('base', 'fixed_first_success',
                           'own_success_raw_text'):
            a = reports[(order, 'continuous_success_mean')]
            b = reports[(order, comparator)]
            if [x['game'] for x in a] != [x['game'] for x in b]:
                raise ValueError('Mismatched paired official games')
            discordance[str(order)][comparator] = {
                'continuous_only_success': sum(x['episode']['reward'] >
                    y['episode']['reward'] for x, y in zip(a, b, strict=True)),
                'comparator_only_success': sum(x['episode']['reward'] <
                    y['episode']['reward'] for x, y in zip(a, b, strict=True))}
        per_order[str(order)] = by_method
    totals = {method: {'successes': sum(
        per_order[str(order)][method]['successes'] for order in range(3)),
        'writes': sum(per_order[str(order)][method]['writes']
                      for order in range(3)),
        'per_order_successes': [per_order[str(order)][method]['successes']
                                for order in range(3)]}
        for method in METHODS}
    value = {'protocol': 'All 12 original-environment audited method-on-policy arms on 3 precommitted, disjoint official valid_unseen orders; games previously named in local reports; each arm acquires own trajectories; descriptive three-order outcome, no 360-independent-trials p-value',
        'review_sha256': file_hash(args.review),
        'raw_report_sha256': report_hashes,
        'audit_sha256': audit_hashes,
        'per_order': per_order,
        'continuous_discordance': dict(discordance),
        'totals': totals}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, ensure_ascii=False,
                                      indent=2) + '\n')
    print(json.dumps(totals), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, default=Path(
        'ttcl/trajectory_hyperlora/alf_valid_unseen_multiorder_v10_plan.json'))
    parser.add_argument('--review', type=Path, default=Path(
        'data/annotations/alf_onpolicy_multiorder_v19_reviewed_20261008.json'))
    parser.add_argument('--results-root', type=Path, default=Path(
        'results/trajectory_hyperlora'))
    parser.add_argument('--output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_onpolicy_v19_summary_20261008.json'))
    summarize(parser.parse_args())
