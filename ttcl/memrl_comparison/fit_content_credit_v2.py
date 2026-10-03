"""Fit a CPU-only set-credit model on reviewed group5/group6 train probes.

All features are available before acting. Group7 is reserved for a later
content-disjoint test and is never read here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import statistics

from ttcl.icl_mem0_comparison.protocol import read, sha
from .audit_alf_coalition_credit import audit as audit_coalitions
from .credit_probe import memory_arms
from .probe_set_conditional_credit import (
    FEATURES as BASE_FEATURES, _crossfit, _example, _fit, _mse, _predict,
    _subset, _task, _tokens, _validate_reviews,
)


ARMS = ('none', 'only_0', 'only_1', 'only_2', 'drop_0', 'drop_1', 'drop_2')
RIDGES = (1.0, 8.0, 32.0)
CONTENT_FEATURES = ('successful_task_overlap', 'failed_task_overlap',
                    'successful_same_task', 'failed_same_task',
                    'failed_same_task_noeffect')
NO_EFFECT = re.compile(
    r'(?i)nothing happens|does not respond|does not (?:provide|yield|work)|'
    r'not supported|ineffective|impossible|misleading|cannot (?:be|use|complete)'
)


def _content_set_features(spec: dict, arms: dict, task: str,
                          selected: tuple[int, ...]) -> list[float]:
    current = _tokens(task)
    values = [0.] * len(CONTENT_FEATURES)
    for i in selected:
        mid = spec['ids'][i]
        content = arms[f'only_{i}']
        source_task, experience = content.split('\nExperience:', 1)
        source = _tokens(source_task.removeprefix('Task: '))
        metadata = spec['memory_features'][mid]['metadata']
        success = metadata['success'] is True
        failure = metadata['success'] is False
        overlap = len(current & source) / max(1, len(current | source))
        same = current == source
        no_effect = bool(NO_EFFECT.search(experience))
        values[0] += float(success) * overlap
        values[1] += float(failure) * overlap
        values[2] += float(success and same)
        values[3] += float(failure and same)
        values[4] += float(failure and same and no_effect)
    return values


def _row(origin: Path, case: str, subset: tuple[int, ...],
         label: float, repeats: list[int], kind: str) -> dict:
    row = _example(origin, case, subset, label, repeats, kind)
    spec, arms = memory_arms(origin, case)
    task = _task(origin, case)
    full = _content_set_features(spec, arms, task, (0, 1, 2))
    partial = _content_set_features(spec, arms, task, subset)
    row['x'].extend(a - b for a, b in zip(partial, full))
    if len(row['x']) != len(BASE_FEATURES) + len(CONTENT_FEATURES) or \
            not all(math.isfinite(value) for value in row['x']):
        raise ValueError('Invalid content-credit features')
    row['family'] = case.split('/')[1]
    return row


def build(group5: Path, group6: Path, reviews5: Path,
          reviews6: Path) -> tuple[list[dict], dict]:
    rows, lineage = [], {}
    for kind, output, reviews in (('group5', group5, reviews5),
                                   ('group6', group6, reviews6)):
        report = audit_coalitions(output)
        if report['missing'] or report['audited'] != report['expected'] or \
                report['audited'] != 18:
            raise ValueError(f'Incomplete {kind} coalition audit')
        origin = Path(read(output / 'design.json')['origin'])
        grouped = defaultdict(list)
        for unit in report['units']:
            if unit['full_repeat_reward_equal']:
                grouped[unit['case']].append(unit)
        if len(grouped) != 6:
            raise ValueError(f'Expected six {kind} inputs')
        group_rows = []
        for case, units in sorted(grouped.items()):
            if len(units) < 2 or len({u['repeat'] for u in units}) != len(units):
                raise ValueError(f'Unstable or duplicate baseline: {case}')
            for arm in ARMS:
                label = statistics.fmean(u['rewards'][arm] - u['rewards']['full']
                                         for u in units)
                group_rows.append(_row(origin, case, _subset(arm), label,
                                       [u['repeat'] for u in units], kind))
        review_doc = read(reviews)
        relevant = dict(review_doc, targets=[item for item in review_doc['targets']
                                             if item['origin'] == kind])
        _validate_reviews(group_rows, relevant, {kind: origin})
        rows.extend(group_rows)
        lineage[kind] = dict(design_sha256=sha(output / 'design.json'),
                             analysis_sha256=sha(output / 'analysis.json'),
                             reviewed_targets_sha256=sha(reviews))
    if len(rows) != 84 or len({r['input_sha256'] for r in rows}) != 12:
        raise ValueError('Expected 12 unique reviewed input-content bindings')
    return rows, lineage


def _family_crossfit(rows: list[dict], columns: tuple[int, ...],
                     ridge: float) -> float:
    families = sorted({row['family'] for row in rows})
    if len(families) != 6:
        raise ValueError('Expected six ALFWorld families')
    ordered, predicted = [], []
    for family in families:
        train = [row for row in rows if row['family'] != family]
        held = [row for row in rows if row['family'] == family]
        ordered.extend(held)
        predicted.extend(_predict(held, columns, _fit(train, columns, ridge)))
    return _mse(ordered, predicted)


def fit(group5: Path, group6: Path, reviews5: Path,
        reviews6: Path) -> dict:
    rows, lineage = build(group5, group6, reviews5, reviews6)
    variants = dict(base=tuple(range(len(BASE_FEATURES))),
                    content=tuple(range(len(BASE_FEATURES) + len(CONTENT_FEATURES))))
    candidates = []
    for name, columns in variants.items():
        for ridge in RIDGES:
            candidates.append(dict(name=name, ridge=ridge,
                                   input_loo_mse=_crossfit(rows, columns, ridge),
                                   family_loo_mse=_family_crossfit(rows, columns, ridge)))
    selected = min(candidates, key=lambda c:(c['input_loo_mse'],
                                               c['family_loo_mse'],
                                               c['name'], c['ridge']))
    columns = variants[selected['name']]
    coefficient = _fit(rows, columns, selected['ridge'])
    return dict(schema='content_set_credit_v2_fit_v1', lineage=lineage,
                fit_script_sha256=sha(Path(__file__)),
                base_feature_script_sha256=sha(Path(__file__).with_name('probe_set_conditional_credit.py')),
                credit_probe_sha256=sha(Path(__file__).with_name('credit_probe.py')),
                features=list(BASE_FEATURES) + list(CONTENT_FEATURES),
                rows=84, train_inputs=12,
                train_input_sha256=sorted({r['input_sha256'] for r in rows}),
                train_zero_mse=_mse(rows, [0.] * len(rows)),
                candidates=candidates, selected=selected,
                selected_columns=list(columns), coefficients=coefficient.tolist(),
                selected_input_loo_mse=selected['input_loo_mse'],
                selected_family_loo_mse=selected['family_loo_mse'],
                caveat='Official-train fixed-snapshot model selection only; no group7 labels, online Q update, or benchmark gain')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('group5', 'group6', 'reviews5', 'reviews6', 'report'):
        parser.add_argument('--' + option, type=Path, required=True)
    args = parser.parse_args()
    result = fit(args.group5.resolve(), args.group6.resolve(),
                 args.reviews5.resolve(), args.reviews6.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('train_input_sha256', 'coefficients')},
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
