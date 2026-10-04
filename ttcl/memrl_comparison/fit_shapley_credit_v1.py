"""Fit per-memory Shapley credit from reviewed three-memory train coalitions.

Only action-before information enters features. Group8 is reserved as an
untouched prospective test; this script reads group5/group6/group7 only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from ttcl.icl_mem0_comparison.protocol import read, sha
from .audit_alf_coalition_credit import audit as audit_coalitions
from .credit_probe import memory_arms
from .fit_content_credit_v2 import NO_EFFECT, _family_crossfit
from .probe_set_conditional_credit import (
    _crossfit, _fit, _mse, _task, _tokens, _validate_reviews,
)


BASE_FEATURES = ('bias', 'source_success', 'source_failure', 'q_value',
                 'retrieval_similarity', 'source_task_overlap', 'text_kchars')
CONTENT_FEATURES = ('same_task', 'noeffect_claim',
                    'failed_same_task_noeffect', 'other_successes',
                    'other_noeffect_claims', 'source_task_overlap_other')
FAMILY_FEATURES = ('source_success', 'source_failure', 'noeffect_claim')
RIDGES = (1.0, 8.0, 32.0)


def _features(spec: dict, arms: dict, task: str,
              index: int, families: list[str]) -> list[float]:
    current = _tokens(task)
    source_tasks, claims = [], []
    for i in range(3):
        source_task, experience = arms[f'only_{i}'].split('\nExperience:', 1)
        source_tasks.append(_tokens(source_task.removeprefix('Task: ')))
        claims.append(bool(NO_EFFECT.search(experience)))
    mid = spec['ids'][index]
    feature = spec['memory_features'][mid]
    metadata, retrieval = feature['metadata'], feature['retrieval']
    success = metadata['success'] is True
    failure = metadata['success'] is False
    source = source_tasks[index]
    overlap = len(current & source) / max(1, len(current | source))
    other = [i for i in range(3) if i != index]
    values = [1., float(success), float(failure),
              float(metadata['q_value'] or 0.),
              float(retrieval['similarity'] or 0.), overlap,
              feature['text_characters'] / 1000.,
              float(current == source), float(claims[index]),
              float(failure and current == source and claims[index]),
              float(sum(spec['memory_features'][spec['ids'][i]]['metadata']['success'] is True
                        for i in other)),
              float(sum(claims[i] for i in other)),
              statistics.fmean(len(source & source_tasks[i]) /
                               max(1, len(source | source_tasks[i])) for i in other)]
    family = spec['task']
    values.extend(value if family == name else 0.
                  for name in families
                  for value in (float(success), float(failure), float(claims[index])))
    if len(values) != len(BASE_FEATURES) + len(CONTENT_FEATURES) + \
            len(families) * len(FAMILY_FEATURES) or \
            not all(math.isfinite(value) for value in values):
        raise ValueError('Invalid per-memory Shapley features')
    return values


def feature_row(origin: Path, case: str, index: int, label: float,
                repeats: list[int], kind: str, families: list[str]) -> dict:
    spec, arms = memory_arms(origin, case)
    if len(spec['ids']) != 3:
        raise ValueError('Expected three retrieved memories')
    task = _task(origin, case)
    return dict(origin=kind, case=case, family=spec['task'],
                input_sha256=spec['source_input_sha256'],
                snapshot_sha256=spec['snapshot_sha256'],
                retrieval_sha256=spec['retrieval_sha256'],
                memory_text_sha256=spec['memory_text_sha256'][spec['ids'][index]],
                memory_index=index, memory_id=spec['ids'][index],
                repeats=repeats, x=_features(spec, arms, task, index, families),
                label=float(label))


def build(group5: Path, group6: Path, group7: Path,
          reviews5: Path, reviews6: Path, reviews7: Path) -> tuple[list[dict], dict, list[str]]:
    outputs = [('group5', group5, reviews5), ('group6', group6, reviews6),
               ('group7', group7, reviews7)]
    designs = [(kind, read(output / 'design.json')) for kind, output, _ in outputs]
    families = sorted({item['case'].split('/')[1]
                       for _, design in designs for item in design['cases']})
    if len(families) != 6:
        raise ValueError('Expected six official task families')
    rows, lineage = [], {}
    for kind, output, reviews in outputs:
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
            raise ValueError(f'Expected six stable {kind} inputs')
        group_rows = []
        for case, units in sorted(grouped.items()):
            if len(units) < 2 or len({u['repeat'] for u in units}) != len(units):
                raise ValueError(f'Unstable or duplicate baseline: {case}')
            for i in range(3):
                value = statistics.fmean(u['shapley'][i] for u in units)
                group_rows.append(feature_row(origin, case, i, value,
                                              [u['repeat'] for u in units],
                                              kind, families))
        review_doc = read(reviews)
        relevant = dict(review_doc, targets=[item for item in review_doc['targets']
                                             if item['origin'] == kind])
        _validate_reviews(group_rows, relevant, {kind: origin})
        rows.extend(group_rows)
        lineage[kind] = dict(design_sha256=sha(output / 'design.json'),
                             analysis_sha256=sha(output / 'analysis.json'),
                             reviewed_targets_sha256=sha(reviews))
    if len(rows) != 54 or len({row['input_sha256'] for row in rows}) != 18:
        raise ValueError('Expected 18 content-distinct input bindings and 54 labels')
    return rows, lineage, families


def fit(group5: Path, group6: Path, group7: Path,
        reviews5: Path, reviews6: Path, reviews7: Path) -> dict:
    rows, lineage, families = build(group5, group6, group7,
                                    reviews5, reviews6, reviews7)
    base = tuple(range(len(BASE_FEATURES)))
    content = tuple(range(len(BASE_FEATURES) + len(CONTENT_FEATURES)))
    variants = dict(base=base, content=content,
                    family_content=tuple(range(len(rows[0]['x']))))
    candidates = []
    for name, columns in variants.items():
        for ridge in RIDGES:
            candidates.append(dict(name=name, ridge=ridge,
                                   input_loo_mse=_crossfit(rows, columns, ridge),
                                   family_loo_mse=_family_crossfit(rows, columns, ridge)))
    selected = min(candidates, key=lambda item:(item['input_loo_mse'],
                                                 item['family_loo_mse'],
                                                 item['name'], item['ridge']))
    columns = variants[selected['name']]
    coefficient = _fit(rows, columns, selected['ridge'])
    zero_mse = _mse(rows, [0.] * len(rows))
    return dict(schema='shapley_credit_v1_fit_v1', lineage=lineage,
                fit_script_sha256=sha(Path(__file__)),
                content_feature_script_sha256=sha(Path(__file__).with_name('fit_content_credit_v2.py')),
                base_feature_script_sha256=sha(Path(__file__).with_name('probe_set_conditional_credit.py')),
                credit_probe_sha256=sha(Path(__file__).with_name('credit_probe.py')),
                families=families,
                feature_names=list(BASE_FEATURES) + list(CONTENT_FEATURES) +
                [f'{family}_{name}' for family in families
                 for name in FAMILY_FEATURES],
                train_inputs=18, train_examples=len(rows),
                train_input_sha256=sorted({row['input_sha256'] for row in rows}),
                train_zero_mse=zero_mse,
                candidates=candidates, selected=selected,
                selected_columns=list(columns), coefficients=coefficient.tolist(),
                eligible_for_group8=(selected['input_loo_mse'] < zero_mse and
                                     selected['family_loo_mse'] < zero_mse),
                caveat='Per-memory Shapley target from official-train coalition replay; no group8 labels, online Q update, or benchmark gain')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('group5', 'group6', 'group7', 'reviews5', 'reviews6',
                   'reviews7', 'report'):
        parser.add_argument('--' + option, type=Path, required=True)
    args = parser.parse_args()
    result = fit(args.group5.resolve(), args.group6.resolve(),
                 args.group7.resolve(), args.reviews5.resolve(),
                 args.reviews6.resolve(), args.reviews7.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('train_input_sha256', 'coefficients',
                                     'feature_names')}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
