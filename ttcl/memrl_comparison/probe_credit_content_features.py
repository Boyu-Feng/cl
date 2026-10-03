"""Exploratory content-feature crossfit on audited train/calibration credit data.

This is a fixed-snapshot diagnostic, not a fitted online policy. It reads
ignored source runs and verifies every example's full content binding before
extracting any pre-action query/experience text.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics

import numpy as np

from .credit_probe import memory_arms


def _tokens(value: str) -> set[str]:
    return set(re.findall(r'(?u)\b\w+\b', value.lower()))


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(1, len(left | right))


def _overlap(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(1, min(len(left), len(right)))


def _group(example: dict) -> str:
    binding = example['binding']
    return binding[0] + ':' + binding[4]


def _content_features(query: str, entry: str) -> list[float]:
    task, abstract = entry.split('\nExperience: ', 1)
    query_terms, task_terms, abstract_terms = (
        _tokens(query), _tokens(task), _tokens(abstract))
    query_numbers = {term for term in query_terms
                     if any(char.isdigit() for char in term)}
    task_numbers = {term for term in task_terms
                    if any(char.isdigit() for char in term)}
    return [_jaccard(query_terms, task_terms),
            _overlap(query_terms, task_terms),
            _jaccard(query_terms, abstract_terms),
            _overlap(query_terms, abstract_terms),
            _jaccard(query_numbers, task_numbers)]


def _folds(examples: list[dict]) -> dict[str, int]:
    by_task = defaultdict(set)
    for example in examples:
        by_task[example['task']].add(_group(example))
    return {group: position % 5 for groups in by_task.values()
            for position, group in enumerate(sorted(groups))}


def _fit(examples: list[dict], key: str) -> np.ndarray:
    x = np.asarray([example[key] for example in examples], dtype=np.float64)
    y = np.asarray([example['label'] for example in examples], dtype=np.float64)
    counts = Counter(_group(example) for example in examples)
    weights = np.asarray([1.0 / counts[_group(example)]
                          for example in examples], dtype=np.float64)
    regularizer = np.eye(x.shape[1]) * 8.0
    regularizer[0, 0] = 2.0
    return np.linalg.solve(x.T @ (weights[:, None] * x) + regularizer,
                           x.T @ (weights * y))


def probe(dataset: dict, alf_origin: Path, cl_origin: Path) -> dict:
    expected = dataset['dataset_sha256']
    body = {key: value for key, value in dataset.items()
            if key != 'dataset_sha256'}
    actual = hashlib.sha256(json.dumps(
        body, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if expected != actual:
        raise ValueError('Training dataset content changed')
    examples = dataset['examples']
    origins = {'alfworld_train': alf_origin, 'clbench_calibration': cl_origin}
    cases = {}
    for example in examples:
        source, case, memory_id, text_hash, input_hash, retrieval_hash, snapshot_hash = (
            example['binding'])
        origin = origins[source]
        if (source, case) not in cases:
            spec, arms = memory_arms(origin, case)
            retrieval_name = ('retrieval_1.json' if source == 'alfworld_train'
                              else 'retrieval.json')
            retrieval = json.loads((origin / 'runs' / case /
                                    retrieval_name).read_text())
            cases[source, case] = (spec, arms, retrieval['query'])
        spec, arms, query = cases[source, case]
        if (spec['source_input_sha256'] != input_hash or
                spec['retrieval_sha256'] != retrieval_hash or
                spec['snapshot_sha256'] != snapshot_hash or
                spec['memory_text_sha256'][memory_id] != text_hash):
            raise ValueError(f'Source content binding changed: {case}')
        entry = arms[f"only_{spec['ids'].index(memory_id)}"]
        z = _content_features(query, entry)
        example['lexical_features'] = [1.0, *z]
        example['combined_features'] = [*example['x'], *z]
    folds = _folds(examples)
    result = {}
    for key in ('x', 'lexical_features', 'combined_features'):
        predictions = []
        for fold in range(5):
            training = [row for row in examples if folds[_group(row)] != fold]
            held = [row for row in examples if folds[_group(row)] == fold]
            coefficients = _fit(training, key)
            predictions.extend((float(np.dot(row[key], coefficients)), row['label'])
                               for row in held)
        if len(predictions) != len(examples):
            raise ValueError('Incomplete input-held-out crossfit')
        result[key] = dict(mse=statistics.fmean((pred - label) ** 2
                                                for pred, label in predictions),
                           mae=statistics.fmean(abs(pred - label)
                                                for pred, label in predictions))
    result['zero'] = dict(mse=statistics.fmean(row['label'] ** 2
                                               for row in examples),
                          mae=statistics.fmean(abs(row['label'])
                                               for row in examples))
    return dict(dataset_sha256=expected, source_cases=len(cases),
                examples=len(examples), five_fold_group='source+public_input_sha256',
                ridge=8.0, lexical_features=(
                    'query_task_jaccard', 'query_task_overlap',
                    'query_abstract_jaccard', 'query_abstract_overlap',
                    'query_task_number_jaccard'), scores=result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--alf-origin', type=Path, required=True)
    parser.add_argument('--cl-origin', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(probe(json.loads(args.dataset.read_text()),
                           args.alf_origin, args.cl_origin),
                     sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
