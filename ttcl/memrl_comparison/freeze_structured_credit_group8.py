"""Seal group8 set-credit predictions before running group8 coalition arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import read, sha
from .audit_credit_train_extension import audit as audit_source
from .fit_content_credit_v2 import ARMS, _row
from .fit_structured_credit_v3 import augment_shape, fit
from .probe_set_conditional_credit import _predict, _subset, _validate_reviews


def freeze(fit_report: Path, group5: Path, group6: Path, group7: Path,
           reviews5: Path, reviews6: Path, reviews7: Path,
           group8: Path, reviews8: Path) -> dict:
    frozen = read(fit_report)
    recreated = fit(group5, group6, group7, reviews5, reviews6, reviews7)
    if frozen != json.loads(json.dumps(recreated)):
        raise ValueError('Frozen structured-credit model changed')
    design = read(group8 / 'design.json')
    if any(group8.rglob('episode.json')) or any(group8.rglob('summary.json')):
        raise ValueError('Group8 coalition outcomes already exist')
    origin = Path(design['origin'])
    source = audit_source(origin)
    if not source['complete'] or source['missing'] or source['audited_pairs'] != 18:
        raise ValueError('Group8 source chain incomplete')
    rows = [augment_shape(_row(origin, item['case'], _subset(arm), 0., [],
                               'group8'), frozen['families'])
            for item in design['cases'] for arm in ARMS]
    if len(rows) != 42 or len({row['input_sha256'] for row in rows}) != 6:
        raise ValueError('Expected six content-distinct group8 inputs')
    _validate_reviews(rows, read(reviews8), {'group8': origin})
    predictions = _predict(rows, tuple(frozen['selected_columns']),
                           frozen['coefficients'])
    return dict(schema='structured_set_credit_group8_prediction_freeze_v1',
                fit_report_sha256=sha(fit_report),
                group8_design_sha256=sha(group8 / 'design.json'),
                group8_source_design_sha256=sha(origin / 'design.json'),
                reviews8_sha256=sha(reviews8),
                rows=[dict(case=row['case'], input_sha256=row['input_sha256'],
                           snapshot_sha256=row['source_snapshot_sha256'],
                           retrieval_sha256=row['retrieval_sha256'],
                           subset=row['subset'], prediction=prediction)
                      for row, prediction in zip(rows, predictions)],
                note='Predictions use reviewed public group8 inputs and frozen group5/group6/group7 coefficients, before group8 coalition outcomes')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('fit-report', 'group5', 'group6', 'group7',
                   'reviews5', 'reviews6', 'reviews7', 'group8',
                   'reviews8', 'report'):
        parser.add_argument('--' + option, type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.fit_report.resolve(), args.group5.resolve(),
                    args.group6.resolve(), args.group7.resolve(),
                    args.reviews5.resolve(), args.reviews6.resolve(),
                    args.reviews7.resolve(), args.group8.resolve(),
                    args.reviews8.resolve())
    path = args.report.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and read(path) != result:
        raise ValueError('Existing group8 prediction freeze changed')
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(report=str(path), sha256=sha(path),
                          rows=len(result['rows'])), ensure_ascii=False))


if __name__ == '__main__':
    main()
