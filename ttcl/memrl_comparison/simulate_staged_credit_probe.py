"""Audit a predeclared early-stop rule for expensive coalition credit probes.

The rule would stop after six arms on each of two actor seeds when all five
unique coalitions agree and both repeated full arms agree. Full completed
probes are retained to measure any robust Shapley signal that stopping would
have missed. This does not update online Q or replace official scoring.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import read, sha
from .audit_alf_coalition_credit import audit as audit_coalitions


EARLY_ARMS = ('full', 'full_repeat', 'none', 'only_0', 'only_1', 'only_2')
PAIR_ARMS = ('drop_0', 'drop_1', 'drop_2')


def _robust_shapley_signs(units: list[dict]) -> list[int]:
    stable = [unit for unit in units if unit['full_repeat_reward_equal']]
    signs = []
    for i in range(3):
        positive = sum(unit['shapley'][i] > 0 for unit in stable)
        negative = sum(unit['shapley'][i] < 0 for unit in stable)
        signs.append(1 if positive >= 2 and negative == 0 else
                     -1 if negative >= 2 and positive == 0 else 0)
    return signs


def _saved_calls(output: Path, case: str, units: list[dict]) -> tuple[int, int]:
    skipped = [(unit['repeat'], name)
               for unit in units[:2] for name in PAIR_ARMS]
    skipped.extend((units[2]['repeat'], name)
                   for name in (*EARLY_ARMS, *PAIR_ARMS))
    calls = 0
    for repeat, name in skipped:
        episode = read(output / case / f'actor_repeat_{repeat}' / name / 'episode.json')
        calls += len(episode['generations'])
    return len(skipped), calls


def simulate(outputs: list[Path]) -> dict:
    cases, lineage = [], {}
    for output in outputs:
        report = audit_coalitions(output)
        if report['missing'] or report['audited'] != report['expected']:
            raise ValueError(f'Incomplete coalition probe: {output}')
        design = read(output / 'design.json')
        lineage[str(output)] = dict(design_sha256=sha(output / 'design.json'),
                                    analysis_sha256=sha(output / 'analysis.json'),
                                    input_content=[item['input_sha256']
                                                   for item in design['cases']])
        groups = defaultdict(list)
        for unit in report['units']:
            groups[unit['case']].append(unit)
        for case, units in sorted(groups.items()):
            units.sort(key=lambda unit: unit['repeat'])
            if len(units) != 3:
                raise ValueError('Expected three actor seeds')
            early_stop = all(
                unit['full_repeat_reward_equal'] and
                len({unit['rewards'][name] for name in EARLY_ARMS}) == 1
                for unit in units[:2]
            )
            signs = _robust_shapley_signs(units)
            episodes_saved, calls_saved = (_saved_calls(output, case, units)
                                           if early_stop else (0, 0))
            cases.append(dict(output=str(output), case=case,
                              early_stop=early_stop,
                              robust_shapley_signs=signs,
                              missed_robust_sign=early_stop and any(signs),
                              episodes_saved=episodes_saved,
                              actor_calls_saved=calls_saved,
                              actor_calls_full=sum(unit['actor_calls'] for unit in units)))
    hashes = [hash_ for values in lineage.values() for hash_ in values['input_content']]
    if len(hashes) != len(set(hashes)):
        raise ValueError('Coalition inputs overlap in content')
    return dict(schema='staged_coalition_credit_probe_simulation_v1',
                lineage=lineage,
                rule='After full, full_repeat, none, only_0/1/2 on each of two seeds: stop if each seed has one common official outcome and full matches its repeat; otherwise run all arms and a third seed',
                robust_sign='Among baseline-stable units, at least two Shapley signs agree and none oppose',
                totals=dict(cases=len(cases), early_stops=sum(c['early_stop'] for c in cases),
                            missed_robust_signs=sum(c['missed_robust_sign'] for c in cases),
                            episodes_saved=sum(c['episodes_saved'] for c in cases),
                            actor_calls_saved=sum(c['actor_calls_saved'] for c in cases),
                            actor_calls_full=sum(c['actor_calls_full'] for c in cases)),
                cases=cases,
                caveat='Full probes are retained for independent audit; saved calls are a counterfactual budget estimate, not actual skipped computation. An adaptive run would move full_repeat earlier than the frozen full-probe arm order, which may alter model sampling and needs direct validation')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, nargs='+', required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = simulate([path.resolve() for path in args.outputs])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result['totals'], ensure_ascii=False))


if __name__ == '__main__':
    main()
