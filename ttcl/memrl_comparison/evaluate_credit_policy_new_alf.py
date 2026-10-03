"""Evaluate the frozen paired-credit rule on disjoint ALFWorld train games.

The reward scale, coefficients, and deletion threshold come exclusively from
the earlier audited training dataset.  These new game rewards never refit it.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from ttcl.icl_mem0_comparison.protocol import read, save, sha

from .analyze_credit_train_extension_probes import analyze
from .paired_credit_rl import features, predict


def evaluate(dataset: dict, policy: dict, outputs: list[Path],
             prediction_manifest: dict) -> dict:
    body = {key: value for key, value in dataset.items()
            if key != 'dataset_sha256'}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True,
                                       allow_nan=False).encode()).hexdigest()
    if digest != dataset['dataset_sha256'] or policy['dataset_sha256'] != digest:
        raise ValueError('Frozen training dataset or policy changed')
    old_inputs = {row['binding'][4] for row in dataset['examples']}
    grouped = defaultdict(list)
    full_rewards = {}
    selection_hashes = set()
    for output in outputs:
        report = analyze(output)
        if report['completed'] != report['expected'] or report['missing']:
            raise ValueError(f'Incomplete new ALFWorld probe: {output}')
        selection_hashes.add(report['selection_sha256'])
        design = read(output / 'design.json')
        position = design['paired_drop_index']
        for row in report['rows']:
            full_key = (row['case'], row['actor_repeat'])
            if (full_key in full_rewards and
                    full_rewards[full_key] != row['full']):
                raise ValueError('Full-context reward changed across deletion probes')
            full_rewards[full_key] = row['full']
            if row['source_input_sha256'] in old_inputs:
                raise ValueError('New ALFWorld input overlaps training')
            if row['task'] not in dataset['scales']:
                raise ValueError('Untrained ALFWorld family')
            source = read(output / 'alfworld' / row['task'] /
                          row['case'].split('/')[2] /
                          row['case'].split('/')[-1] /
                          'source.json')
            ids = source['ids']
            if ids[position] != row['memory_id']:
                raise ValueError('Probe position differs from source retrieval')
            bounded = max(-1.0, min(1.0, row['delta'] /
                                    dataset['scales'][row['task']]))
            key = (row['case'], row['memory_id'], row['memory_text_sha256'],
                   row['source_input_sha256'], row['retrieval_sha256'],
                   row['snapshot_sha256'])
            grouped[key].append((row['actor_repeat'], row['delta'], bounded,
                                 features(row['memory_features'], len(ids), position),
                                 row['task']))
    if len(selection_hashes) != 1 or not grouped:
        raise ValueError('Missing or inconsistent new ALFWorld selection')
    selection_hash = next(iter(selection_hashes))
    if prediction_manifest['selection_sha256'] != selection_hash:
        raise ValueError('Frozen prediction selection changed')
    frozen = {(row['case'], row['memory_id']): row
              for row in prediction_manifest['rows']}
    if len(frozen) != len(grouped):
        raise ValueError('Frozen prediction set differs from probes')
    examples = []
    for key, rows in sorted(grouped.items()):
        if (len(rows) != 3 or len({row[0] for row in rows}) != 3 or
                any(row[3:] != rows[0][3:] for row in rows[1:])):
            raise ValueError(f'Incomplete or inconsistent actor seeds: {key}')
        estimate = predict(policy, rows[0][3])
        prior = frozen.get((key[0], key[1]))
        if (prior is None or prior['memory_text_sha256'] != key[2] or
                prior['source_input_sha256'] != key[3] or
                prior['prediction'] != estimate):
            raise ValueError('Frozen pre-outcome prediction changed')
        label = statistics.fmean(row[2] for row in rows)
        examples.append(dict(case=key[0], memory_id=key[1],
                             memory_text_sha256=key[2], input_sha256=key[3],
                             task=rows[0][4], seed_deltas=[row[1] for row in rows],
                             bounded_label=label, **estimate))
    by_input = defaultdict(list)
    for row in examples:
        by_input[row['input_sha256']].append(row)
    removed = [row for row in examples if row['remove']]
    return dict(schema='paired_credit_new_alf_input_test_v1',
                frozen_training_sha256=digest,
                frozen_policy_schema=policy['schema'],
                selection_sha256=selection_hash,
                new_train_games=len(by_input),
                new_memory_examples=len(examples),
                new_seed_differences=3 * len(examples),
                model_mse=statistics.fmean(
                    (row['mean'] - row['bounded_label']) ** 2 for row in examples),
                zero_mse=statistics.fmean(
                    row['bounded_label'] ** 2 for row in examples),
                removed=len(removed),
                isolated_bounded_gain=-sum(row['bounded_label'] for row in removed),
                per_input=[dict(input_sha256=input_hash,
                                family=rows[0]['task'],
                                memory_examples=len(rows),
                                zero_minus_model_mse=statistics.fmean(
                                    row['bounded_label'] ** 2 -
                                    (row['mean'] - row['bounded_label']) ** 2
                                    for row in rows))
                           for input_hash, rows in sorted(by_input.items())],
                examples=examples,
                caveat='Official train games disjoint by input hash from frozen policy training; fixed-snapshot first-attempt marginal, shared source memory lineage, no online-chain gain')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--prediction-manifest', type=Path, required=True)
    parser.add_argument('--outputs', nargs='+', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    manifest = read(args.prediction_manifest)
    if manifest['policy_sha256'] != sha(args.policy):
        raise ValueError('Frozen prediction policy file changed')
    result = evaluate(read(args.dataset), read(args.policy),
                      [path.resolve() for path in args.outputs], manifest)
    result['prediction_manifest_sha256'] = sha(args.prediction_manifest)
    save(args.report, result)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {'examples', 'per_input'}}, sort_keys=True))


if __name__ == '__main__':
    main()
