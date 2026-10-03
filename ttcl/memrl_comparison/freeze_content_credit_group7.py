"""Seal group7 credit predictions before running any group7 coalition arm."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import read, sha
from .audit_credit_train_extension import audit as audit_source
from .fit_content_credit_v2 import ARMS, _row, fit
from .probe_set_conditional_credit import _predict, _subset, _validate_reviews


def freeze(fit_report: Path, group5: Path, group6: Path, reviews5: Path,
           reviews6: Path, group7: Path, reviews7: Path) -> dict:
    frozen = read(fit_report)
    if frozen != json.loads(json.dumps(fit(group5, group6, reviews5, reviews6))):
        raise ValueError('Frozen training model changed')
    design = read(group7 / 'design.json')
    if any(group7.rglob('episode.json')) or any(group7.rglob('summary.json')):
        raise ValueError('Group7 coalition outcomes already exist; cannot seal predictions')
    origin = Path(design['origin'])
    source = audit_source(origin)
    if not source['complete'] or source['missing'] or source['audited_pairs'] != 18:
        raise ValueError('Group7 source chain incomplete')
    rows = [_row(origin, item['case'], _subset(arm), 0., [], 'group7')
            for item in design['cases'] for arm in ARMS]
    if len(rows) != 42 or len({row['input_sha256'] for row in rows}) != 6:
        raise ValueError('Expected six content-distinct group7 inputs')
    _validate_reviews(rows, read(reviews7), {'group7': origin})
    predictions = _predict(rows, tuple(frozen['selected_columns']),
                           frozen['coefficients'])
    return dict(schema='content_set_credit_group7_prediction_freeze_v1',
                fit_report_sha256=sha(fit_report),
                group7_design_sha256=sha(group7 / 'design.json'),
                group7_source_design_sha256=sha(origin / 'design.json'),
                reviews7_sha256=sha(reviews7),
                rows=[dict(case=row['case'], input_sha256=row['input_sha256'],
                           snapshot_sha256=row['source_snapshot_sha256'],
                           retrieval_sha256=row['retrieval_sha256'],
                           subset=row['subset'], prediction=prediction)
                      for row, prediction in zip(rows, predictions)],
                note='Predictions use only reviewed public inputs, frozen retrieved memories and group5/group6 coefficients; created before group7 coalition outcomes')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('fit-report', 'group5', 'group6', 'reviews5', 'reviews6',
                   'group7', 'reviews7', 'report'):
        parser.add_argument('--' + option, type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.fit_report.resolve(), args.group5.resolve(),
                    args.group6.resolve(), args.reviews5.resolve(),
                    args.reviews6.resolve(), args.group7.resolve(),
                    args.reviews7.resolve())
    path = args.report.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and read(path) != result:
        raise ValueError('Existing group7 prediction freeze changed')
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(report=str(path), sha256=sha(path),
                          rows=len(result['rows'])), ensure_ascii=False))


if __name__ == '__main__':
    main()
