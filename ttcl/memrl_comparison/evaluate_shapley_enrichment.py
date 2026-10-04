"""Evaluate a frozen per-memory critic on blind and enriched ALFWorld inputs.

The old group5–7 fit was frozen before either new coalition result. The
enriched inputs were selected using source reward and are development data,
not an unbiased benchmark sample. A pooled cross-fit is exploratory only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from ttcl.icl_mem0_comparison.protocol import read, sha
from .audit_alf_coalition_credit import audit as audit_coalitions
from .fit_content_credit_v2 import _family_crossfit
from .fit_shapley_credit_v1 import (
    BASE_FEATURES, CONTENT_FEATURES, build as build_train, feature_row,
)
from .probe_set_conditional_credit import _crossfit, _mse, _validate_reviews


def _held_rows(output: Path, reviews: Path, kind: str,
               families: list[str]) -> list[dict]:
    report = audit_coalitions(output)
    if report['missing'] or report['audited'] != report['expected']:
        raise ValueError('Incomplete held-out coalition audit')
    origin = Path(read(output / 'design.json')['origin'])
    grouped = defaultdict(list)
    for unit in report['units']:
        if unit['full_repeat_reward_equal']:
            grouped[unit['case']].append(unit)
    expected = 6 if kind == 'group8' else 4
    if len(grouped) != expected:
        raise ValueError('Held-out case count changed')
    rows = []
    for case, units in sorted(grouped.items()):
        if len(units) < 2:
            raise ValueError('Insufficient baseline-stable actor seeds')
        for i in range(3):
            rows.append(feature_row(origin, case, i,
                                    statistics.fmean(u['shapley'][i] for u in units),
                                    [u['repeat'] for u in units], kind, families))
    _validate_reviews(rows, read(reviews), {kind: origin})
    return rows


def _score(rows: list[dict], fit: dict) -> dict:
    predictions = [sum(row['x'][i] * value
                       for i, value in zip(fit['selected_columns'],
                                           fit['coefficients'])) for row in rows]
    return dict(inputs=len({row['input_sha256'] for row in rows}),
                examples=len(rows),
                nonzero_labels=sum(abs(row['label']) > 1e-9 for row in rows),
                frozen_model_mse=_mse(rows, predictions),
                zero_mse=_mse(rows, [0.] * len(rows)),
                predictions=[dict(case=row['case'],
                                  input_sha256=row['input_sha256'],
                                  memory_index=row['memory_index'],
                                  memory_text_sha256=row['memory_text_sha256'],
                                  label=row['label'], prediction=value)
                             for row, value in zip(rows, predictions)])


def evaluate(fit_path: Path, groups: list[Path], reviews: list[Path],
             group8: Path, reviews8: Path, enriched: Path,
             reviews_enriched: Path) -> dict:
    fit = read(fit_path)
    if fit['schema'] != 'shapley_credit_v1_fit_v1' or \
            fit['eligible_for_group8']:
        raise ValueError('Expected frozen, offline-only per-memory critic')
    train, lineage, families = build_train(*groups, *reviews)
    if fit['lineage'] != lineage or fit['families'] != families or \
            fit['train_input_sha256'] != sorted({r['input_sha256'] for r in train}):
        raise ValueError('Frozen fit does not bind to training labels')
    blind = _held_rows(group8, reviews8, 'group8', families)
    selected = _held_rows(enriched, reviews_enriched, 'enriched', families)
    hashes = [row['input_sha256'] for rows in (train, blind, selected)
              for row in rows[::3]]
    if len(hashes) != len(set(hashes)):
        raise ValueError('Training and evaluation public inputs overlap')
    pooled = train + blind + selected
    base = tuple(range(len(BASE_FEATURES)))
    content = tuple(range(len(BASE_FEATURES) + len(CONTENT_FEATURES)))
    variants = {'base':base, 'content':content,
                'family_content':tuple(range(len(train[0]['x'])))}
    exploratory = [dict(name=name, ridge=ridge,
                        input_loo_mse=_crossfit(pooled, columns, ridge),
                        family_loo_mse=_family_crossfit(pooled, columns, ridge))
                   for name, columns in variants.items() for ridge in (1., 8., 32.)]
    return dict(schema='shapley_credit_enrichment_evaluation_v1',
                frozen_fit_sha256=sha(fit_path),
                group8_design_sha256=sha(group8 / 'design.json'),
                enriched_design_sha256=sha(enriched / 'design.json'),
                enriched_reviews_sha256=sha(reviews_enriched),
                train_inputs=18, group8=_score(blind, fit),
                enriched=_score(selected, fit),
                exploratory_pooled=dict(inputs=len(hashes),
                                        examples=len(pooled),
                                        zero_mse=_mse(pooled, [0.] * len(pooled)),
                                        candidates=exploratory),
                eligible_for_online_ablation=False,
                caveat='Frozen old critic remains worse than zero on new inputs; pooled 28-input cross-fit is post-outcome exploration with source-selected examples, not prospective policy evidence')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-report', type=Path, required=True)
    for name in ('group5', 'group6', 'group7', 'reviews5', 'reviews6',
                 'reviews7', 'group8', 'reviews8', 'enriched',
                 'reviews-enriched', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.fit_report.resolve(),
                      [getattr(args, name).resolve() for name in
                       ('group5', 'group6', 'group7')],
                      [getattr(args, name).resolve() for name in
                       ('reviews5', 'reviews6', 'reviews7')],
                      args.group8.resolve(), args.reviews8.resolve(),
                      args.enriched.resolve(), args.reviews_enriched.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({name: {key:value for key,value in result[name].items()
                             if key != 'predictions'}
                      for name in ('group8','enriched')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
