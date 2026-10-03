"""Exploratory set-conditioned marginal-value model for MemRL credit.

The linear coalition value has individual and pairwise terms. Training uses
only audited old leave-one-out labels; evaluation uses content-disjoint Poker
prefix inputs. This script does not alter retrieval or online Q values.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from ttcl.icl_mem0_comparison.protocol import read

from .analyze_credit_prefix_extension import analyze as audit_extension
from .analyze_cl_credit_probe import analyze as audit_cl
from .credit_probe import memory_arms
from .paired_credit_rl import FEATURES, RIDGE, features


def _group(row: dict) -> str:
    return row['source'] + ':' + row['source_input_sha256']


def _marginal_features(records: list[dict], position: int) -> list[float]:
    """Features of V(S)-V(S without the indexed memory)."""
    size = len(records)
    if not 0 <= position < size:
        raise ValueError('Invalid memory position')
    vectors = [features(record, size, index)
               for index, record in enumerate(records)]
    own = vectors[position]
    pairwise = [sum(own[column] * other[column]
                    for index, other in enumerate(vectors)
                    if index != position)
                for column in range(len(FEATURES))]
    return own + pairwise


def _fit(rows: list[dict]) -> np.ndarray:
    x = np.asarray([row['x'] for row in rows], dtype=np.float64)
    y = np.asarray([row['label'] for row in rows], dtype=np.float64)
    if not len(rows) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Missing or nonfinite coalition labels')
    counts = Counter(_group(row) for row in rows)
    weights = np.asarray([1 / counts[_group(row)] for row in rows])
    regularizer = np.eye(x.shape[1]) * RIDGE
    regularizer[0, 0] = RIDGE / 4
    return np.linalg.solve(x.T @ (weights[:, None] * x) + regularizer,
                           x.T @ (weights * y))


def _old_examples(dataset: dict, alf_origin: Path,
                  cl_origin: Path) -> list[dict]:
    expected = dataset['dataset_sha256']
    body = {key: value for key, value in dataset.items()
            if key != 'dataset_sha256'}
    if hashlib.sha256(json.dumps(body, sort_keys=True,
                                 allow_nan=False).encode()).hexdigest() != expected:
        raise ValueError('Old training dataset changed')
    origins = {'alfworld_train': alf_origin,
               'clbench_calibration': cl_origin}
    cache = {}
    examples = []
    for old in dataset['examples']:
        source, case, memory_id, text_hash, input_hash, retrieval_hash, snapshot_hash = (
            old['binding'])
        if (source, case) not in cache:
            cache[source, case] = memory_arms(origins[source], case)[0]
        spec = cache[source, case]
        if (spec['source_input_sha256'] != input_hash or
                spec['retrieval_sha256'] != retrieval_hash or
                spec['snapshot_sha256'] != snapshot_hash or
                spec['memory_text_sha256'][memory_id] != text_hash or
                features(spec['memory_features'][memory_id], len(spec['ids']),
                         spec['ids'].index(memory_id)) != old['x']):
            raise ValueError('Old per-memory source binding changed')
        records = [spec['memory_features'][mid] for mid in spec['ids']]
        x = _marginal_features(records, spec['ids'].index(memory_id))
        examples.append(dict(source=source, case=case,
                             source_input_sha256=input_hash,
                             memory_id=memory_id, x=x,
                             label=old['label']))
    return examples


def _new_examples(dataset: dict, outputs: list[Path]) -> tuple[list[dict], int]:
    extension = audit_extension(outputs)
    old_inputs = {row['binding'][4] for row in dataset['examples']}
    scale = dataset['scales']['exploitable_poker']
    grouped = defaultdict(list)
    for output in outputs:
        report = audit_cl(output)
        for row in report['rows']:
            if row['source_input_sha256'] in old_inputs:
                raise ValueError('New input overlaps model training')
            records = [row['memory_features'][mid] for mid in row['memory_ids']]
            for position, memory_id in enumerate(row['memory_ids']):
                key = (row['case'], row['source_input_sha256'],
                       memory_id, row['memory_text_sha256'][memory_id])
                delta = row['delta_by_memory'][position]
                bounded = max(-1., min(1., delta / scale))
                grouped[key].append((row['actor_repeat'], bounded,
                                     _marginal_features(records, position)))
    examples = []
    for (case, input_hash, memory_id, _), rows in sorted(grouped.items()):
        if len(rows) != 3 or len({row[0] for row in rows}) != 3 or \
                any(row[2] != rows[0][2] for row in rows[1:]):
            raise ValueError('New coalition comparison is incomplete')
        examples.append(dict(source='clbench_calibration', case=case,
                             source_input_sha256=input_hash,
                             memory_id=memory_id, x=rows[0][2],
                             label=statistics.fmean(row[1] for row in rows)))
    if len({row['source_input_sha256'] for row in examples}) != \
            extension['distinct_public_inputs']:
        raise ValueError('Audited public input count changed')
    return examples, extension['distinct_public_inputs']


def _score(rows: list[dict], predictions: list[float]) -> dict:
    if len(rows) != len(predictions):
        raise ValueError('Prediction count changed')
    return dict(examples=len(rows),
                mse=statistics.fmean((prediction - row['label']) ** 2
                                     for row, prediction in zip(rows, predictions)),
                zero_mse=statistics.fmean(row['label'] ** 2 for row in rows),
                mae=statistics.fmean(abs(prediction - row['label'])
                                     for row, prediction in zip(rows, predictions)))


def _evaluate_variant(old: list[dict], new: list[dict]) -> dict:
    by_task = defaultdict(set)
    for row in old:
        by_task[row['case'].split('/')[1]].add(_group(row))
    folds = {group: index % 5 for groups in by_task.values()
             for index, group in enumerate(sorted(groups))}
    old_predictions = [None] * len(old)
    for fold in range(5):
        training = [row for row in old if folds[_group(row)] != fold]
        coefficients = _fit(training)
        for index, row in enumerate(old):
            if folds[_group(row)] == fold:
                old_predictions[index] = float(np.dot(row['x'], coefficients))
    if any(value is None for value in old_predictions):
        raise ValueError('Incomplete input-held-out crossfit')
    coefficients = _fit(old)
    new_predictions = [float(np.dot(row['x'], coefficients)) for row in new]
    return dict(old_crossfit=_score(old, old_predictions),
                new_holdout=_score(new, new_predictions),
                new_predictions=new_predictions)


def probe(dataset: dict, alf_origin: Path, cl_origin: Path,
          outputs: list[Path]) -> dict:
    old = _old_examples(dataset, alf_origin, cl_origin)
    new, distinct = _new_examples(dataset, outputs)
    pairwise = _evaluate_variant(old, new)
    individual_old = [dict(row, x=row['x'][:len(FEATURES)]) for row in old]
    individual_new = [dict(row, x=row['x'][:len(FEATURES)]) for row in new]
    individual = _evaluate_variant(individual_old, individual_new)
    return dict(schema='coalition_credit_probe_v1',
                old_dataset_sha256=dataset['dataset_sha256'],
                feature_schema='sum of native pre-action features plus pairwise elementwise products',
                ridge=RIDGE, old_public_inputs=len({_group(row) for row in old}),
                new_public_inputs=distinct,
                old_crossfit=pairwise['old_crossfit'],
                new_holdout=pairwise['new_holdout'],
                individual_only_old_crossfit=individual['old_crossfit'],
                individual_only_new_holdout=individual['new_holdout'],
                new_predictions=[dict(case=row['case'],
                                      public_input_sha256=row['source_input_sha256'],
                                      memory_id=row['memory_id'],
                                      label=row['label'], prediction=prediction,
                                      individual_only_prediction=individual_prediction)
                                 for row, prediction, individual_prediction in zip(
                                     new, pairwise['new_predictions'],
                                     individual['new_predictions'])],
                caveat='Model structure chosen after inspecting the six new-input outcomes; exploratory evaluation only, no independent model-selection holdout or online Q result')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--alf-origin', type=Path, required=True)
    parser.add_argument('--cl-origin', type=Path, required=True)
    parser.add_argument('--new-outputs', nargs=2, type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = probe(read(args.dataset), args.alf_origin,
                   args.cl_origin, [path.resolve() for path in args.new_outputs])
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key != 'new_predictions'}, sort_keys=True))


if __name__ == '__main__':
    main()
