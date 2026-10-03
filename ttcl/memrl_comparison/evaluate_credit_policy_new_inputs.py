"""Test the frozen paired-credit predictor on new Poker prefix inputs.

The model and reward scale come only from the earlier audited training data.
New inputs are never used to refit or choose its deletion threshold.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from ttcl.icl_mem0_comparison.protocol import read

from .analyze_credit_prefix_extension import analyze as analyze_extension
from .paired_credit_rl import features, predict


def evaluate(dataset: dict, policy: dict, outputs: list[Path]) -> dict:
    expected = dataset['dataset_sha256']
    body = {key: value for key, value in dataset.items()
            if key != 'dataset_sha256'}
    if (hashlib.sha256(json.dumps(body, sort_keys=True,
                             allow_nan=False).encode()).hexdigest() != expected or
            policy['dataset_sha256'] != expected):
        raise ValueError('Frozen training data or fitted policy changed')
    extension = analyze_extension(outputs)
    old_inputs = {row['binding'][4] for row in dataset['examples']}
    scale = dataset['scales']['exploitable_poker']
    grouped = defaultdict(list)
    for output in outputs:
        design = read(output / 'design.json')
        from .analyze_cl_credit_probe import analyze as analyze_cl
        report = analyze_cl(output)
        for row in report['rows']:
            if row['source_input_sha256'] in old_inputs:
                raise ValueError('New-input evaluation overlaps policy training')
            if row['task'] != 'exploitable_poker':
                raise ValueError('Unexpected task in Poker evaluation')
            for position, memory_id in enumerate(row['memory_ids']):
                key = (design['selection_sha256'], row['case'], memory_id,
                       row['memory_text_sha256'][memory_id],
                       row['source_input_sha256'], row['retrieval_sha256'],
                       row['snapshot_sha256'])
                delta = row['delta_by_memory'][position]
                bounded = max(-1.0, min(1.0, delta / scale))
                x = features(row['memory_features'][memory_id],
                             len(row['memory_ids']), position)
                grouped[key].append((row['actor_repeat'], delta, bounded, x))
    examples = []
    for key, rows in sorted(grouped.items()):
        if (len(rows) != 3 or len({row[0] for row in rows}) != 3 or
                any(row[3] != rows[0][3] for row in rows[1:])):
            raise ValueError('Incomplete or inconsistent new-input seeds')
        estimate = predict(policy, rows[0][3])
        label = statistics.fmean(row[2] for row in rows)
        examples.append(dict(case=key[1], memory_id=key[2],
                             memory_text_sha256=key[3],
                             public_input_sha256=key[4],
                             seed_deltas=[row[1] for row in rows],
                             bounded_label=label, **estimate))
    if len({row['public_input_sha256'] for row in examples}) != \
            extension['distinct_public_inputs']:
        raise ValueError('Public input count differs from audited extension')
    model_mse = statistics.fmean((row['mean'] - row['bounded_label']) ** 2
                                 for row in examples)
    zero_mse = statistics.fmean(row['bounded_label'] ** 2 for row in examples)
    removed = [row for row in examples if row['remove']]
    by_input = defaultdict(list)
    for row in examples:
        by_input[row['public_input_sha256']].append(row)
    input_gains = [dict(public_input_sha256=input_hash,
                        memory_examples=len(rows),
                        zero_minus_model_mse=statistics.fmean(
                            row['bounded_label'] ** 2 -
                            (row['mean'] - row['bounded_label']) ** 2
                            for row in rows))
                   for input_hash, rows in sorted(by_input.items())]
    return dict(schema='paired_credit_new_input_test_v1',
                frozen_training_sha256=expected,
                frozen_policy_schema=policy['schema'],
                training_reward_scale=scale,
                new_public_inputs=extension['distinct_public_inputs'],
                new_memory_examples=len(examples),
                new_seed_differences=sum(len(row['seed_deltas'])
                                         for row in examples),
                model_mse=model_mse, zero_mse=zero_mse,
                input_error_gains=input_gains,
                removed=len(removed),
                isolated_bounded_gain=-sum(row['bounded_label'] for row in removed),
                examples=examples,
                caveat='Fixed-snapshot calibration-prefix inputs disjoint by content from training; shared source lineage and no online-chain validation')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--outputs', nargs=2, type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(read(args.dataset), read(args.policy),
                      [path.resolve() for path in args.outputs])
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key != 'examples'}, sort_keys=True))


if __name__ == '__main__':
    main()
