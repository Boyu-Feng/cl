"""Fit a set-credit critic with task-type coverage and interference features.

Group5/group6/group7 are reviewed ALFWorld train calibration inputs. Group8 is
reserved for a later untouched prospective test and is never read here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from ttcl.icl_mem0_comparison.protocol import read, sha
from .audit_alf_coalition_credit import audit as audit_coalitions
from .fit_content_credit_v2 import ARMS, _family_crossfit, _row, build as build_old
from .probe_set_conditional_credit import (
    FEATURES as BASE_FEATURES, _crossfit, _fit, _mse, _subset,
    _validate_reviews,
)


RIDGES = (1.0, 8.0, 32.0)


def augment_shape(row: dict, families: list[str]) -> dict:
    count = len(row['subset'])
    if count not in (0, 1, 2) or row['family'] not in families:
        raise ValueError('Expected proper subset and known task family')
    shape = [float(count == i) for i in (0, 1, 2)]
    per_family = [value if row['family'] == family else 0.
                  for family in families for value in shape]
    row = dict(row)
    row['x'] = row['x'][:len(BASE_FEATURES)] + shape + per_family
    return row


def build(group5: Path, group6: Path, group7: Path,
          reviews5: Path, reviews6: Path, reviews7: Path) -> tuple[list[dict], dict]:
    old_rows, lineage = build_old(group5, group6, reviews5, reviews6)
    audit = audit_coalitions(group7)
    if audit['missing'] or audit['audited'] != audit['expected'] or audit['audited'] != 18:
        raise ValueError('Incomplete group7 coalition audit')
    origin = Path(read(group7 / 'design.json')['origin'])
    grouped = defaultdict(list)
    for unit in audit['units']:
        if unit['full_repeat_reward_equal']:
            grouped[unit['case']].append(unit)
    if len(grouped) != 6:
        raise ValueError('Expected six group7 inputs')
    new_rows = []
    for case, units in sorted(grouped.items()):
        if len(units) < 2 or len({u['repeat'] for u in units}) != len(units):
            raise ValueError(f'Unstable or duplicate group7 baseline: {case}')
        for arm in ARMS:
            label = statistics.fmean(u['rewards'][arm] - u['rewards']['full']
                                     for u in units)
            new_rows.append(_row(origin, case, _subset(arm), label,
                                 [u['repeat'] for u in units], 'group7'))
    _validate_reviews(new_rows, read(reviews7), {'group7': origin})
    lineage['group7'] = dict(design_sha256=sha(group7 / 'design.json'),
                             analysis_sha256=sha(group7 / 'analysis.json'),
                             reviewed_targets_sha256=sha(reviews7))
    rows = old_rows + new_rows
    families = sorted({row['family'] for row in rows})
    if len(families) != 6 or len(rows) != 126 or \
            len({row['input_sha256'] for row in rows}) != 18:
        raise ValueError('Expected 18 content-distinct reviewed inputs')
    rows = [augment_shape(row, families) for row in rows]
    return rows, lineage


def fit(group5: Path, group6: Path, group7: Path,
        reviews5: Path, reviews6: Path, reviews7: Path) -> dict:
    rows, lineage = build(group5, group6, group7,
                          reviews5, reviews6, reviews7)
    base = tuple(range(len(BASE_FEATURES)))
    global_shape = tuple(range(len(BASE_FEATURES), len(BASE_FEATURES) + 3))
    family_shape = tuple(range(len(BASE_FEATURES), len(rows[0]['x'])))
    variants = dict(base=base, global_shape=global_shape,
                    family_shape=family_shape,
                    base_family_shape=base + family_shape)
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
    families = sorted({row['family'] for row in rows})
    return dict(schema='structured_set_credit_v3_fit_v1', lineage=lineage,
                fit_script_sha256=sha(Path(__file__)),
                base_feature_script_sha256=sha(Path(__file__).with_name('probe_set_conditional_credit.py')),
                content_feature_script_sha256=sha(Path(__file__).with_name('fit_content_credit_v2.py')),
                credit_probe_sha256=sha(Path(__file__).with_name('credit_probe.py')),
                families=families,
                feature_names=list(BASE_FEATURES) +
                ['empty', 'singleton', 'pair'] +
                [f'{family}_{shape}' for family in families
                 for shape in ('empty', 'singleton', 'pair')],
                train_inputs=18, train_examples=len(rows),
                train_input_sha256=sorted({row['input_sha256'] for row in rows}),
                train_zero_mse=_mse(rows, [0.] * len(rows)),
                candidates=candidates, selected=selected,
                selected_columns=list(columns), coefficients=coefficient.tolist(),
                caveat='Official-train fixed-snapshot model selection only; coverage/interference features reflect set cardinality and public task family, no group8 labels or online Q update')


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
