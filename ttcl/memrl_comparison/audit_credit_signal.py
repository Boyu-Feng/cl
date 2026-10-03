"""Audit identifiability of source-bound paired memory credit labels."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


def audit(dataset: dict, crossfit: dict) -> dict:
    expected = dataset.get('dataset_sha256')
    body = {key: value for key, value in dataset.items()
            if key != 'dataset_sha256'}
    actual = hashlib.sha256(json.dumps(
        body, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if expected != actual or crossfit.get('dataset_sha256') != expected:
        raise ValueError('Dataset or crossfit content binding changed')
    examples = dataset['examples']
    if len(examples) != dataset['example_count']:
        raise ValueError('Example count changed')
    counts = Counter()
    for row in examples:
        values = row['raw_deltas']
        shapley = row['shapley_deltas']
        if (len(values) != len(row['actor_repeats']) or
                len(values) != len(shapley) or
                not all(isinstance(v, (int, float)) and math.isfinite(v)
                        for v in values)):
            raise ValueError('Incomplete paired seed labels')
        positive = sum(v > 0 for v in values)
        negative = sum(v < 0 for v in values)
        counts['examples'] += 1
        counts['seed_differences'] += len(values)
        counts['zero_seed_differences'] += sum(v == 0 for v in values)
        counts['all_zero_examples'] += positive + negative == 0
        counts['replicated_positive_examples'] += positive >= 2 and negative == 0
        counts['replicated_negative_examples'] += negative >= 2 and positive == 0
        counts['conflicting_sign_examples'] += positive > 0 and negative > 0
        counts['single_nonzero_examples'] += positive + negative == 1
        for loo, share in zip(values, shapley):
            if share is None:
                continue
            if not isinstance(share, (int, float)) or not math.isfinite(share):
                raise ValueError('Invalid Shapley credit')
            counts['shapley_seed_estimates'] += 1
            counts['shapley_interaction_nonzero'] += abs(loo - share) > 1e-9
            counts['shapley_sign_reversals'] += loo * share < 0
    if counts['seed_differences'] != dataset['observation_count']:
        raise ValueError('Observation count changed')
    rows = crossfit['rows']
    if len(rows) != len(examples):
        raise ValueError('Crossfit row count changed')
    model_mse = statistics.fmean((row['mean'] - row['label']) ** 2
                                  for row in rows)
    zero_mse = statistics.fmean(row['label'] ** 2 for row in rows)
    return dict(dataset_sha256=expected, **counts,
                crossfit_removed=sum(bool(row['remove']) for row in rows),
                crossfit_model_mse=model_mse,
                crossfit_zero_mse=zero_mse)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--crossfit', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(json.loads(args.dataset.read_text()),
                           json.loads(args.crossfit.read_text())),
                     sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
