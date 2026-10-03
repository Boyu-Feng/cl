"""Aggregate audited ALFWorld three-memory coalition diagnostics by input."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .audit_alf_coalition_credit import audit


def summarize(report: dict) -> dict:
    if report['missing'] or report['audited'] != report['expected']:
        raise ValueError('Cannot summarize incomplete coalition probe')
    grouped = defaultdict(list)
    for unit in report['units']:
        grouped[unit['case']].append(unit)
    cases = []
    for case, units in grouped.items():
        units.sort(key=lambda unit: unit['repeat'])
        if len(units) != 3 or len({unit['repeat'] for unit in units}) != 3:
            raise ValueError('Expected three independent actor repeats per case')
        rewards = [unit['rewards'] for unit in units]
        stable = [unit for unit in units if unit['full_repeat_actions_equal']]
        cases.append(dict(case=case, repeats=[unit['repeat'] for unit in units],
                          full=[r['full'] for r in rewards],
                          none=[r['none'] for r in rewards],
                          singleton=[[r[f'only_{i}'] for r in rewards] for i in range(3)],
                          deletion=[[r[f'drop_{i}'] for r in rewards] for i in range(3)],
                          loo=[[unit['loo'][i] for unit in units] for i in range(3)],
                          shapley=[[unit['shapley'][i] for unit in units] for i in range(3)],
                          any_singleton_better=sum(any(r[f'only_{i}'] > r['full']
                                                       for i in range(3)) for r in rewards),
                          any_deletion_better=sum(any(r[f'drop_{i}'] > r['full']
                                                      for i in range(3)) for r in rewards),
                          stable_repeats=[unit['repeat'] for unit in stable],
                          stable_any_singleton_better=sum(any(unit['rewards'][f'only_{i}'] >
                                                              unit['rewards']['full']
                                                              for i in range(3)) for unit in stable),
                          stable_any_deletion_better=sum(any(unit['rewards'][f'drop_{i}'] >
                                                             unit['rewards']['full']
                                                             for i in range(3)) for unit in stable),
                          full_repeat_reward_agree=sum(unit['full_repeat_reward_equal']
                                                       for unit in units),
                          full_repeat_actions_agree=sum(unit['full_repeat_actions_equal']
                                                        for unit in units),
                          actor_calls=sum(unit['actor_calls'] for unit in units)))
    total_units = sum(len(case['repeats']) for case in cases)
    return dict(schema='alf_three_memory_coalition_summary_v1',
                design_sha256=report['design_sha256'],
                cases=sorted(cases, key=lambda item: item['case']),
                totals=dict(cases=len(cases), units=total_units,
                            full_repeat_reward_agree=sum(c['full_repeat_reward_agree'] for c in cases),
                            full_repeat_actions_agree=sum(c['full_repeat_actions_agree'] for c in cases),
                            any_singleton_better=sum(c['any_singleton_better'] for c in cases),
                            any_deletion_better=sum(c['any_deletion_better'] for c in cases),
                            stable_units=sum(len(c['stable_repeats']) for c in cases),
                            stable_any_singleton_better=sum(c['stable_any_singleton_better']
                                                            for c in cases),
                            stable_any_deletion_better=sum(c['stable_any_deletion_better']
                                                           for c in cases),
                            actor_calls=sum(c['actor_calls'] for c in cases)),
                caveat='Oracle subset comparisons are diagnostic upper bounds, not an inference-time selection policy')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    summary = summarize(audit(args.output.resolve()))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
