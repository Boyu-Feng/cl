"""Retrospective object-match gating of reviewed three-memory coalitions.

The rule was proposed after observing source-object leakage, so these scores
are development diagnostics. Exact eight-subset official rewards allow a
read-only comparison without composing untested deletion effects.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics

from ttcl.icl_mem0_comparison.protocol import read, sha
from .annotate_claim_applicability_pilot import REVIEWS
from .audit_alf_coalition_credit import audit as audit_coalitions
from .credit_probe import memory_arms
from .probe_goal_slot_credit import _slots
from .probe_set_conditional_credit import _task


def _arm(kept: list[int]) -> str:
    if len(kept) == 0:
        return 'none'
    if len(kept) == 3:
        return 'full'
    if len(kept) == 1:
        return f'only_{kept[0]}'
    if len(kept) == 2:
        return f'drop_{next(i for i in range(3) if i not in kept)}'
    raise ValueError('Expected at most three memories')


def _interval(values: list[float]) -> list[float]:
    rng = random.Random(92721)
    draws = sorted(statistics.fmean(rng.choice(values) for _ in values)
                   for _ in range(2000))
    return [draws[49], draws[1949]]


def audit(opportunity_path: Path) -> dict:
    source = read(opportunity_path)
    if source['schema'] != 'alf_credit_q_downstream_opportunity_v1':
        raise ValueError('Wrong audited coalition source')
    review_root = Path(__file__).parents[2] / 'data' / 'annotations'
    rows = []
    for output in source['outputs']:
        coalition = Path(output['output'])
        origin = Path(read(coalition / 'design.json')['origin'])
        report = audit_coalitions(coalition)
        if report['missing'] or report['audited'] != report['expected']:
            raise ValueError('Incomplete official coalition replay')
        kind, review_name = REVIEWS[coalition.name]
        review_path = review_root / review_name
        reviews = read(review_path)['targets']
        units = defaultdict(list)
        for unit in report['units']:
            if unit['full_repeat_reward_equal']:
                units[unit['case']].append(unit)
        for case in output['cases']:
            case_name = case['case']
            task = _task(origin, case_name)
            matches = [review for review in reviews
                       if review['origin'] == kind and
                       review['case'] == case_name and
                       review['input_sha256'] == case['input_sha256'] and
                       review['public_task'] == task and review['reviewed']]
            if len(matches) != 1:
                raise ValueError('Missing reviewed content-bound game')
            spec, arms = memory_arms(origin, case_name)
            if spec['source_input_sha256'] != case['input_sha256'] or \
                    len(units[case_name]) < 2:
                raise ValueError('Input or stable-seed binding changed')
            current_object, _, _ = _slots(task)
            source_objects = [_slots(arms[f'only_{i}'].split(
                '\nExperience:', 1)[0].removeprefix('Task: '))[0]
                for i in range(3)]
            kept = [i for i, obj in enumerate(source_objects)
                    if obj == current_object]
            chosen = _arm(kept)
            candidate = statistics.fmean(unit['rewards'][chosen]
                                         for unit in units[case_name])
            full = statistics.fmean(unit['rewards']['full']
                                    for unit in units[case_name])
            empty = statistics.fmean(unit['rewards']['none']
                                     for unit in units[case_name])
            rows.append(dict(output=str(coalition), case=case_name,
                             input_sha256=case['input_sha256'],
                             reviewed_targets_sha256=sha(review_path),
                             current_object=current_object,
                             source_objects=source_objects, kept_indices=kept,
                             chosen_arm=chosen,
                             stable_repeats=[unit['repeat']
                                             for unit in units[case_name]],
                             candidate_success=candidate,
                             native_full_success=full,
                             no_memory_success=empty,
                             candidate_minus_full=candidate - full,
                             candidate_minus_none=candidate - empty))
    if len(rows) != 28 or len({row['input_sha256'] for row in rows}) != 28:
        raise ValueError('Expected 28 content-distinct reviewed train inputs')
    full_diffs = [row['candidate_minus_full'] for row in rows]
    none_diffs = [row['candidate_minus_none'] for row in rows]
    return dict(schema='alf_goal_object_gate_diagnostic_v1',
                script_sha256=sha(Path(__file__)),
                source_opportunity_sha256=sha(opportunity_path), rows=rows,
                totals=dict(inputs=len(rows),
                            changed_contexts=sum(row['chosen_arm'] != 'full'
                                                 for row in rows),
                            vs_full=dict(wins=sum(v > 0 for v in full_diffs),
                                         losses=sum(v < 0 for v in full_diffs),
                                         ties=sum(v == 0 for v in full_diffs),
                                         mean=statistics.fmean(full_diffs),
                                         input_bootstrap_interval=_interval(full_diffs)),
                            vs_none=dict(wins=sum(v > 0 for v in none_diffs),
                                         losses=sum(v < 0 for v in none_diffs),
                                         ties=sum(v == 0 for v in none_diffs),
                                         mean=statistics.fmean(none_diffs),
                                         input_bootstrap_interval=_interval(none_diffs))),
                caveat='Rule was chosen after inspecting development outcomes. This is fixed-snapshot first-attempt coalition replay, not an online chain, unseen-game or CLBench result.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opportunities', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.opportunities.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['totals']))


if __name__ == '__main__':
    main()
