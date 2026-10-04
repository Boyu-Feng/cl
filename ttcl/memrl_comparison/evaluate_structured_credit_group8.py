"""Evaluate pre-sealed structured set-credit predictions on new group8 inputs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics

from ttcl.icl_mem0_comparison.protocol import read, sha
from .audit_alf_coalition_credit import audit as audit_coalitions
from .fit_content_credit_v2 import ARMS, _row
from .fit_structured_credit_v3 import augment_shape, build, fit
from .probe_set_conditional_credit import _mse, _predict, _subset, _validate_reviews


def evaluate(fit_report: Path, group5: Path, group6: Path, group7: Path,
             reviews5: Path, reviews6: Path, reviews7: Path,
             group8: Path, reviews8: Path, prediction_freeze: Path) -> dict:
    frozen = read(fit_report)
    recreated = fit(group5, group6, group7, reviews5, reviews6, reviews7)
    if frozen != json.loads(json.dumps(recreated)):
        raise ValueError('Frozen structured-credit model changed')
    training, lineage = build(group5, group6, group7,
                              reviews5, reviews6, reviews7)
    if lineage != frozen['lineage']:
        raise ValueError('Reviewed training bindings changed')
    audited = audit_coalitions(group8)
    if audited['missing'] or audited['audited'] != audited['expected'] or \
            audited['audited'] != 18:
        raise ValueError('Group8 coalition audit incomplete')
    origin = Path(read(group8 / 'design.json')['origin'])
    grouped = defaultdict(list)
    for unit in audited['units']:
        grouped[unit['case']].append(unit)
    testing, stability = [], []
    for case, units in sorted(grouped.items()):
        if len(units) != 3 or len({u['repeat'] for u in units}) != 3:
            raise ValueError('Missing distinct group8 actor repeats')
        stable = [unit for unit in units if unit['full_repeat_reward_equal']]
        if len(stable) < 2:
            raise ValueError(f'Fewer than two stable baselines: {case}')
        stability.append(dict(case=case,
                              stable_repeats=[u['repeat'] for u in stable],
                              unstable_repeats=[u['repeat'] for u in units if u not in stable]))
        for arm in ARMS:
            label = statistics.fmean(u['rewards'][arm] - u['rewards']['full']
                                     for u in stable)
            testing.append(augment_shape(
                _row(origin, case, _subset(arm), label,
                     [u['repeat'] for u in stable], 'group8'),
                frozen['families']))
    train_hashes = {row['input_sha256'] for row in training}
    test_hashes = {row['input_sha256'] for row in testing}
    if len(train_hashes) != 18 or len(test_hashes) != 6 or train_hashes & test_hashes:
        raise ValueError('Training and group8 input content overlap')
    _validate_reviews(testing, read(reviews8), {'group8': origin})
    prediction = _predict(testing, tuple(frozen['selected_columns']),
                          frozen['coefficients'])
    sealed = read(prediction_freeze)
    expected_rows = [dict(case=row['case'], input_sha256=row['input_sha256'],
                          snapshot_sha256=row['source_snapshot_sha256'],
                          retrieval_sha256=row['retrieval_sha256'],
                          subset=row['subset'], prediction=value)
                     for row, value in zip(testing, prediction)]
    if (sealed['schema'] != 'structured_set_credit_group8_prediction_freeze_v1' or
            sealed['fit_report_sha256'] != sha(fit_report) or
            sealed['group8_design_sha256'] != sha(group8 / 'design.json') or
            sealed['group8_source_design_sha256'] != sha(origin / 'design.json') or
            sealed['reviews8_sha256'] != sha(reviews8) or
            sealed['rows'] != expected_rows):
        raise ValueError('Sealed pre-outcome group8 predictions changed')
    model_mse = _mse(testing, prediction)
    zero_mse = _mse(testing, [0.] * len(testing))
    by_input = defaultdict(list)
    examples = []
    for row, value in zip(testing, prediction):
        item = dict(case=row['case'], input_sha256=row['input_sha256'],
                    subset=row['subset'], label=row['label'], prediction=value)
        examples.append(item)
        by_input[row['input_sha256']].append(row['label'] ** 2 -
                                              (row['label'] - value) ** 2)
    improvements = [statistics.fmean(values) for values in by_input.values()]
    rng = random.Random(93791)
    bootstrap = sorted(statistics.fmean(rng.choice(improvements)
                                        for _ in improvements) for _ in range(2000))
    nonzero = [row for row in examples if row['label']]
    return dict(schema='structured_set_credit_group8_prospective_v1',
                frozen_fit_sha256=sha(fit_report),
                prediction_freeze_sha256=sha(prediction_freeze),
                group8_design_sha256=sha(group8 / 'design.json'),
                group8_source_design_sha256=sha(origin / 'design.json'),
                group8_reviews_sha256=sha(reviews8),
                selected=frozen['selected'],
                train_inputs=18, train_examples=len(training),
                test_inputs=6, test_examples=len(testing), stability=stability,
                nonzero_test_labels=len(nonzero),
                correct_nonzero_signs=sum((row['label'] > 0) ==
                                          (row['prediction'] > 0) for row in nonzero),
                model_mse=model_mse, zero_mse=zero_mse,
                mse_improvement=zero_mse - model_mse,
                mse_improvement_interval=[bootstrap[50], bootstrap[1950]],
                eligible_for_online_ablation=(model_mse < zero_mse and
                                              bootstrap[50] > 0),
                predictions=examples,
                caveat='Selected task-type coverage/interference CPU critic on content-disjoint official-train group8 fixed snapshots; no online Q or writer update')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('fit-report', 'group5', 'group6', 'group7',
                   'reviews5', 'reviews6', 'reviews7', 'group8',
                   'reviews8', 'prediction-freeze', 'report'):
        parser.add_argument('--' + option, type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.fit_report.resolve(), args.group5.resolve(),
                      args.group6.resolve(), args.group7.resolve(),
                      args.reviews5.resolve(), args.reviews6.resolve(),
                      args.reviews7.resolve(), args.group8.resolve(),
                      args.reviews8.resolve(), args.prediction_freeze.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key != 'predictions'}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
