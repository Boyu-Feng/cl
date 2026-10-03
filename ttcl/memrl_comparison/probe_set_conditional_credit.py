"""CPU-only cross-input probe of a set-conditioned ALFWorld credit predictor.

Group5 complete coalitions supply training labels. Content-disjoint group4
and group3 fixed-snapshot interventions are held out from fitting and ridge
selection. This is a retrospective mechanism test, never an online policy.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics

import numpy as np

from ttcl.icl_mem0_comparison.protocol import read, sha
from .audit_alf_coalition_credit import audit as audit_coalitions
from .audit_alf_single_memory_rule import audit as audit_only_third
from .audit_alf_source_success_filter import audit as audit_success_filter
from .credit_probe import memory_arms


RIDGES = (0.1, 1.0, 8.0, 32.0)
FEATURES = ('count', 'source_success', 'source_failure', 'q_value',
            'retrieval_similarity', 'source_task_overlap', 'text_kchars',
            'pair_count', 'pair_mixed_source_result', 'pair_task_overlap')
STOP = frozenset(('the', 'a', 'an', 'some', 'with', 'and', 'in', 'on', 'at',
                  'to', 'of', 'from', 'under', 'into', 'it', 'then', 'look',
                  'put', 'pick', 'up', 'place', 'task', 'is', 'examine'))


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r'[a-z]+', value.lower())
            if token not in STOP and len(token) > 1}


def _task(origin: Path, case: str) -> str:
    episode = read(origin / 'runs' / case / 'attempt_1' / 'episode.json')
    match = re.search(r'Your task is to:\s*([^\n]+)', episode['initial_observation'])
    if not match:
        raise ValueError(f'Missing public task instruction: {case}')
    return match.group(1).strip()


def _set_features(spec: dict, arms: dict, task: str,
                  selected: tuple[int, ...]) -> np.ndarray:
    current = _tokens(task)
    values, tasks = [], []
    for i in selected:
        mid = spec['ids'][i]
        feature = spec['memory_features'][mid]
        metadata, retrieval = feature['metadata'], feature['retrieval']
        source_task = arms[f'only_{i}'].split('\nExperience:', 1)[0].removeprefix('Task: ')
        tokens = _tokens(source_task)
        tasks.append(tokens)
        values.append(dict(success=float(metadata['success'] is True),
                           failure=float(metadata['success'] is False),
                           q=float(metadata['q_value'] or 0.),
                           sim=float(retrieval['similarity'] or 0.),
                           overlap=len(current & tokens) / max(1, len(current)),
                           length=feature['text_characters'] / 1000.))
    pair_count = len(selected) * (len(selected) - 1) / 2
    mixed = sum(values[i]['success'] != values[j]['success']
                for i in range(len(values)) for j in range(i + 1, len(values)))
    pair_overlap = sum(len(tasks[i] & tasks[j]) / max(1, len(tasks[i] | tasks[j]))
                       for i in range(len(tasks)) for j in range(i + 1, len(tasks)))
    return np.asarray((len(selected), sum(v['success'] for v in values),
                       sum(v['failure'] for v in values),
                       sum(v['q'] for v in values),
                       sum(v['sim'] for v in values),
                       sum(v['overlap'] for v in values),
                       sum(v['length'] for v in values),
                       pair_count, mixed, pair_overlap), dtype=np.float64)


def _subset(name: str) -> tuple[int, ...]:
    if name == 'none':
        return ()
    if name == 'full':
        return (0, 1, 2)
    if name.startswith('only_'):
        return (int(name[-1]),)
    if name.startswith('drop_'):
        return tuple(i for i in range(3) if i != int(name[-1]))
    raise ValueError(f'Unknown three-memory coalition: {name}')


def _example(origin: Path, case: str, subset: tuple[int, ...],
             label: float, repeats: list[int], origin_kind: str) -> dict:
    spec, arms = memory_arms(origin, case)
    if len(spec['ids']) != 3 or not all(0 <= i < 3 for i in subset):
        raise ValueError('Invalid three-memory source binding')
    task = _task(origin, case)
    x = _set_features(spec, arms, task, subset) - _set_features(
        spec, arms, task, (0, 1, 2))
    if not math.isfinite(label) or not np.isfinite(x).all():
        raise ValueError('Nonfinite set credit example')
    return dict(origin=origin_kind, case=case,
                input_sha256=spec['source_input_sha256'],
                source_snapshot_sha256=spec['snapshot_sha256'],
                retrieval_sha256=spec['retrieval_sha256'],
                ids=spec['ids'], memory_text_sha256=spec['memory_text_sha256'],
                subset=list(subset), repeats=repeats,
                x=x.tolist(), label=float(label))


def build(train_output: Path, only_output: Path,
          success_output: Path) -> tuple[list[dict], list[dict], dict]:
    train_report = audit_coalitions(train_output)
    only_report = audit_only_third(only_output)
    success_report = audit_success_filter(success_output)
    for report in (train_report, only_report, success_report):
        if report['missing'] or report['audited'] != report['expected']:
            raise ValueError('Incomplete audited credit probe')
    train_origin = Path(read(train_output / 'design.json')['origin'])
    only_origin = Path(read(only_output / 'design.json')['origin'])
    success_origin = Path(read(success_output / 'design.json')['origin'])
    train_groups = defaultdict(list)
    for unit in train_report['units']:
        if unit['full_repeat_reward_equal']:
            train_groups[unit['case']].append(unit)
    training = []
    for case, units in sorted(train_groups.items()):
        if len(units) < 2:
            raise ValueError('Insufficient repeated baseline agreement')
        for arm in ('none', *(f'only_{i}' for i in range(3)),
                    *(f'drop_{i}' for i in range(3))):
            label = statistics.fmean(unit['rewards'][arm] -
                                     unit['rewards']['full'] for unit in units)
            training.append(_example(train_origin, case, _subset(arm), label,
                                     [unit['repeat'] for unit in units], 'group5'))
    held = []
    for output, report, origin, kind, selected in (
            (only_output, only_report, only_origin, 'group4', lambda unit:(2,)),
            (success_output, success_report, success_origin, 'group3',
             lambda unit:tuple(unit['kept_indices']))):
        groups = defaultdict(list)
        for unit in report['units']:
            groups[unit['case']].append(unit)
        for case, units in sorted(groups.items()):
            if len(units) != 3 or len({u['repeat'] for u in units}) != 3:
                raise ValueError('Missing held-out actor repeats')
            subsets = {selected(unit) for unit in units}
            if len(subsets) != 1:
                raise ValueError('Held-out candidate subset changed')
            held.append(_example(origin, case, next(iter(subsets)),
                                 statistics.fmean(u['corrected_effect'] for u in units),
                                 [u['repeat'] for u in units], kind))
    train_hashes = {row['input_sha256'] for row in training}
    held_hashes = [row['input_sha256'] for row in held]
    if (len(train_hashes) != 6 or len(held_hashes) != 12 or
            train_hashes.intersection(held_hashes)):
        raise ValueError('Train and held-out public inputs overlap')
    lineage = dict(train_design_sha256=sha(train_output / 'design.json'),
                   train_analysis_sha256=sha(train_output / 'analysis.json'),
                   only_design_sha256=sha(only_output / 'design.json'),
                   only_analysis_sha256=sha(only_output / 'analysis.json'),
                   success_design_sha256=sha(success_output / 'design.json'),
                   success_analysis_sha256=sha(success_output / 'analysis.json'))
    return training, held, lineage


def _fit(rows: list[dict], columns: tuple[int, ...], ridge: float) -> np.ndarray:
    x = np.asarray([[row['x'][i] for i in columns] for row in rows], dtype=np.float64)
    y = np.asarray([row['label'] for row in rows], dtype=np.float64)
    counts = defaultdict(int)
    for row in rows:
        counts[row['input_sha256']] += 1
    weight = np.asarray([1 / counts[row['input_sha256']] for row in rows])
    return np.linalg.solve(x.T @ (weight[:, None] * x) + ridge * np.eye(len(columns)),
                           x.T @ (weight * y))


def _predict(rows: list[dict], columns: tuple[int, ...], coef: np.ndarray) -> list[float]:
    return [float(np.dot([row['x'][i] for i in columns], coef)) for row in rows]


def _mse(rows: list[dict], prediction: list[float]) -> float:
    groups = defaultdict(list)
    for row, value in zip(rows, prediction):
        groups[row['input_sha256']].append((row['label'] - value) ** 2)
    return statistics.fmean(statistics.fmean(values) for values in groups.values())


def _crossfit(rows: list[dict], columns: tuple[int, ...], ridge: float) -> float:
    cases = sorted({row['input_sha256'] for row in rows})
    predictions = []
    ordered = []
    for case in cases:
        train = [row for row in rows if row['input_sha256'] != case]
        test = [row for row in rows if row['input_sha256'] == case]
        predictions.extend(_predict(test, columns, _fit(train, columns, ridge)))
        ordered.extend(test)
    return _mse(ordered, predictions)


def _validate_reviews(rows: list[dict], reviews: dict,
                      origins: dict[str, Path]) -> None:
    if reviews.get('schema') != 'alf_set_credit_reviewed_targets_v1':
        raise ValueError('Missing newly reviewed task targets')
    entries = reviews.get('targets')
    if not isinstance(entries, list):
        raise ValueError('Missing reviewed input-content bindings')
    by_key = {}
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError('Invalid reviewed target')
        key = (item.get('origin'), item.get('input_sha256'))
        if key in by_key or item.get('reviewed') is not True:
            raise ValueError('Duplicate or unreviewed target')
        by_key[key] = item
    for row in rows:
        key = (row['origin'], row['input_sha256'])
        item = by_key.get(key)
        if item is None or item.get('case') != row['case']:
            raise ValueError('Target review missing for content-bound input')
        task = _task(origins[row['origin']], row['case'])
        if (item.get('public_task') != task or
                item.get('public_task_sha256') !=
                hashlib.sha256(task.encode()).hexdigest()):
            raise ValueError('Reviewed public task differs from trajectory')
    if len(by_key) != len({(row['origin'], row['input_sha256']) for row in rows}):
        raise ValueError('Reviewed target set contains unrelated inputs')


def probe(train_output: Path, only_output: Path,
          success_output: Path, reviewed_targets: Path) -> dict:
    training, held, lineage = build(train_output, only_output, success_output)
    reviews = read(reviewed_targets)
    _validate_reviews(training + held, reviews, dict(
        group5=Path(read(train_output / 'design.json')['origin']),
        group4=Path(read(only_output / 'design.json')['origin']),
        group3=Path(read(success_output / 'design.json')['origin'])))
    lineage['reviewed_targets_sha256'] = sha(reviewed_targets)
    variants = {'source_result_only':(1, 2),
                'set_conditioned':tuple(range(len(FEATURES)))}
    candidates = []
    for name, columns in variants.items():
        for ridge in RIDGES:
            candidates.append((name, columns, ridge,
                               _crossfit(training, columns, ridge)))
    name, columns, ridge, train_cv = min(candidates,
                                         key=lambda item:(item[3], item[0], item[2]))
    coefficient = _fit(training, columns, ridge)
    prediction = _predict(held, columns, coefficient)
    held_zero = _mse(held, [0.] * len(held))
    per_case = [dict(origin=row['origin'], case=row['case'],
                     input_sha256=row['input_sha256'], subset=row['subset'],
                     label=row['label'], prediction=value)
                for row, value in zip(held, prediction)]
    rng = random.Random(93491)
    improvements = [row['label'] ** 2 - (row['label'] - row['prediction']) ** 2
                    for row in per_case]
    bootstrap = [statistics.fmean(rng.choice(improvements)
                                  for _ in improvements) for _ in range(2000)]
    bootstrap.sort()
    return dict(schema='set_conditional_credit_probe_v1', lineage=lineage,
                feature_names=FEATURES, train_inputs=6, train_examples=len(training),
                held_inputs=12, held_examples=len(held),
                train_zero_mse=_mse(training, [0.] * len(training)),
                candidates=[dict(name=n, ridge=r, train_input_loo_mse=m)
                            for n,_,r,m in candidates],
                selected_variant=name, selected_ridge=ridge,
                train_input_loo_mse=train_cv,
                held_mse=_mse(held, prediction), held_zero_mse=held_zero,
                held_mse_improvement_interval=[bootstrap[50], bootstrap[1950]],
                eligible_for_online_ablation=(
                    _mse(held, prediction) < held_zero and bootstrap[50] > 0),
                held_predictions=per_case,
                caveat='Retrospective exploratory fit; group3/4 results were visible during design, no online action or independent model-selection holdout')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--only-third', type=Path, required=True)
    parser.add_argument('--source-success', type=Path, required=True)
    parser.add_argument('--reviewed-targets', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.train.resolve(), args.only_third.resolve(),
                   args.source_success.resolve(), args.reviewed_targets.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'held_predictions'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
