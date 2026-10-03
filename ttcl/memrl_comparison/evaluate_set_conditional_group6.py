"""Prospectively test the frozen group5 set-credit model on group6 coalitions.

The old model, hyperparameters and group5 training labels are frozen by the
saved prior report. Group6 supplies only independent evaluation labels.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics

from ttcl.icl_mem0_comparison.protocol import read, sha
from .audit_alf_coalition_credit import audit as audit_coalitions
from .probe_set_conditional_credit import (
    FEATURES, _example, _fit, _mse, _predict, _subset, _validate_reviews,
    build, probe,
)


ARMS = ('none', 'only_0', 'only_1', 'only_2', 'drop_0', 'drop_1', 'drop_2')


def evaluate(prior_report: Path, train: Path, only_third: Path,
             source_success: Path, old_reviews: Path, group6: Path,
             new_reviews: Path) -> dict:
    frozen = read(prior_report)
    recreated = probe(train, only_third, source_success, old_reviews)
    if frozen != json.loads(json.dumps(recreated)) or \
            frozen['selected_variant'] != 'set_conditioned' or \
            frozen['selected_ridge'] != 32.0:
        raise ValueError('Previously selected model or training lineage changed')
    training, _, lineage = build(train, only_third, source_success)
    lineage['reviewed_targets_sha256'] = sha(old_reviews)
    if lineage != frozen['lineage']:
        raise ValueError('Frozen group5 training lineage changed')
    audited = audit_coalitions(group6)
    if audited['missing'] or audited['audited'] != audited['expected'] or \
            audited['audited'] != 18:
        raise ValueError('Group6 coalition audit incomplete')
    group6_origin = Path(read(group6 / 'design.json')['origin'])
    units_by_case = defaultdict(list)
    for unit in audited['units']:
        units_by_case[unit['case']].append(unit)
    testing, stability = [], []
    for case, units in sorted(units_by_case.items()):
        if len(units) != 3 or len({unit['repeat'] for unit in units}) != 3:
            raise ValueError('Missing distinct group6 actor repeats')
        stable = [unit for unit in units if unit['full_repeat_reward_equal']]
        if len(stable) < 2:
            raise ValueError(f'Fewer than two stable baselines: {case}')
        stability.append(dict(case=case, stable_repeats=[u['repeat'] for u in stable],
                              unstable_repeats=[u['repeat'] for u in units if u not in stable]))
        for arm in ARMS:
            label = statistics.fmean(u['rewards'][arm] - u['rewards']['full']
                                     for u in stable)
            testing.append(_example(group6_origin, case, _subset(arm), label,
                                    [u['repeat'] for u in stable], 'group6'))
    train_hashes = {row['input_sha256'] for row in training}
    test_hashes = {row['input_sha256'] for row in testing}
    if len(train_hashes) != 6 or len(test_hashes) != 6 or train_hashes & test_hashes:
        raise ValueError('Group5 training and group6 evaluation inputs overlap')
    _validate_reviews(testing, read(new_reviews), {'group6': group6_origin})
    columns = tuple(range(len(FEATURES)))
    coefficient = _fit(training, columns, frozen['selected_ridge'])
    predicted = _predict(testing, columns, coefficient)
    model_mse = _mse(testing, predicted)
    zero_mse = _mse(testing, [0.] * len(testing))
    by_input = defaultdict(list)
    examples = []
    for row, value in zip(testing, predicted):
        item = dict(case=row['case'], input_sha256=row['input_sha256'],
                    subset=row['subset'], label=row['label'], prediction=value)
        examples.append(item)
        by_input[row['input_sha256']].append(row['label'] ** 2 -
                                              (row['label'] - value) ** 2)
    improvements = [statistics.fmean(values) for values in by_input.values()]
    rng = random.Random(93591)
    bootstrap = sorted(statistics.fmean(rng.choice(improvements)
                                        for _ in improvements) for _ in range(2000))
    return dict(schema='set_conditional_group6_prospective_v1',
                prior_report_sha256=sha(prior_report),
                group6_design_sha256=sha(group6 / 'design.json'),
                group6_source_design_sha256=sha(group6_origin / 'design.json'),
                group6_reviews_sha256=sha(new_reviews),
                frozen_model=dict(variant=frozen['selected_variant'],
                                  ridge=frozen['selected_ridge'],
                                  features=list(FEATURES)),
                train_inputs=6, train_examples=len(training),
                test_inputs=6, test_examples=len(testing), stability=stability,
                model_mse=model_mse, zero_mse=zero_mse,
                mse_improvement=zero_mse - model_mse,
                mse_improvement_interval=[bootstrap[50], bootstrap[1950]],
                eligible_for_online_ablation=(model_mse < zero_mse and
                                              bootstrap[50] > 0),
                predictions=examples,
                caveat='Frozen group5 CPU model tested on content-disjoint official-train group6 coalitions; fixed snapshot only, no online Q or writer update')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('prior-report', 'train', 'only-third', 'source-success',
                   'old-reviews', 'group6', 'new-reviews', 'report'):
        parser.add_argument('--' + option, type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.prior_report.resolve(), args.train.resolve(),
                      args.only_third.resolve(), args.source_success.resolve(),
                      args.old_reviews.resolve(), args.group6.resolve(),
                      args.new_reviews.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key != 'predictions'}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
