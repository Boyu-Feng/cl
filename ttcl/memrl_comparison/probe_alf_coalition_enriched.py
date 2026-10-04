"""Probe four source-discordant, unprobed official-train ALFWorld inputs.

This is outcome-aware *development* sampling based on native-vs-none source
rewards, never a prospective policy evaluation. No coalition outcome is used
for selection. Full coalitions are retained for exact interaction audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import Client, read, save, sha
from .audit_credit_train_extension import audit as audit_source
from .credit_probe import memory_arms, run_alf


REPEATS = (93801, 93802, 93803)


def selected_cases(origin: Path, prior_probe: Path) -> list[str]:
    design = read(origin / 'design.json')
    if design['group_index'] != 7 or \
            read(prior_probe / 'design.json')['origin'] != str(origin):
        raise ValueError('Expected group8 source and prior blind probe')
    prior = {row['case'] for row in read(prior_probe / 'design.json')['cases']}
    cases = []
    for family, games in sorted(design['selected_games'].items()):
        for offset, game in enumerate(games, 1):
            index = design['bootstrap'][family]['index'] + offset
            case = f'alfworld/{family}/{design["repeat"]}/memrl/episode_{index:03d}'
            if case in prior:
                continue
            retrieval = read(origin / 'runs' / case / 'retrieval_1.json')
            if len(retrieval['ids']) != 3:
                continue
            native = read(origin / 'runs' / case / 'row.json')
            none = read(origin / 'runs' / case.replace('/memrl/', '/none/') / 'row.json')
            if (native['input_sha256'] != game['sha256'] or
                    none['input_sha256'] != game['sha256']):
                raise ValueError('Source input changed')
            if native['within_three'] > none['within_three']:
                cases.append(case)
    if len(cases) != 4 or len({case.split('/')[1] for case in cases}) != 4:
        raise ValueError('Expected four source-discordant unprobed task families')
    return cases


def design_for(origin: Path, prior_probe: Path) -> dict:
    report = audit_source(origin)
    if not report['complete'] or report['missing'] or report['audited_pairs'] != 18:
        raise ValueError('Native source chain is incomplete')
    previous = read(prior_probe / 'analysis.json')
    if (previous['audited'] != previous['expected'] or previous['missing'] or
            previous['design_sha256'] != sha(prior_probe / 'design.json')):
        raise ValueError('Prior blind coalition probe is incomplete')
    cases = []
    for case in selected_cases(origin, prior_probe):
        spec, arms = memory_arms(origin, case)
        if len(spec['ids']) != 3 or len(arms) != 8:
            raise ValueError('Expected eight coalitions of three memories')
        cases.append(dict(case=case, input_sha256=spec['source_input_sha256'],
                          ids=spec['ids'], snapshot_sha256=spec['snapshot_sha256'],
                          retrieval_sha256=spec['retrieval_sha256'],
                          original_row_sha256=spec['original_memrl_row_sha256'],
                          context_sha256=spec['arm_context_sha256']))
    return dict(schema='alf_three_memory_coalition_enriched_v1',
                origin=str(origin), plan_sha256=sha(origin / 'plan.json'),
                source_design_sha256=sha(origin / 'design.json'),
                source_complete_sha256=sha(origin / 'complete.json'),
                prior_probe=str(prior_probe),
                prior_probe_design_sha256=sha(prior_probe / 'design.json'),
                prior_probe_analysis_sha256=sha(prior_probe / 'analysis.json'),
                runner_sha256=sha(Path(__file__)),
                credit_probe_sha256=sha(Path(__file__).with_name('credit_probe.py')),
                repeats=list(REPEATS), cases=cases,
                selection='Completed official-train group8 unprobed three-memory cases with native within-three success and none failure; four development inputs, no coalition outcomes consulted',
                budget='Nine arms per case and repeat; at most 50 environment actions per arm; no writer or Q updates')


def run(origin: Path, prior_probe: Path, output: Path, url: str,
        prepare_only: bool) -> None:
    design = design_for(origin, prior_probe)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'design.json'
    if path.exists():
        if read(path) != design:
            raise ValueError('Frozen selection or source changed')
    else:
        save(path, design)
    if prepare_only:
        print(json.dumps(dict(prepared=str(path), design_sha256=sha(path),
                              cases=[item['case'] for item in design['cases']])))
        return
    plan = read(origin / 'plan.json')
    plan['url'] = url
    plan['alf']['actor_url'] = url
    for item in design['cases']:
        case = item['case']
        spec, arms = memory_arms(origin, case)
        for repeat in REPEATS:
            names = [name for name in arms if name != 'full']
            random.Random(int(hashlib.sha256(f'{case}/{repeat}'.encode()).hexdigest(), 16)).shuffle(names)
            contexts = {'full': arms['full']}
            contexts.update((name, arms[name]) for name in names)
            contexts['full_repeat'] = arms['full']
            target = output / case / f'actor_repeat_{repeat}'
            target.mkdir(parents=True, exist_ok=True)
            order_path = target / 'order.json'
            order = dict(names=list(contexts), seed=repeat)
            if order_path.exists() and read(order_path) != order:
                raise ValueError('Frozen arm order changed')
            save(order_path, order)
            summary_path = target / 'summary.json'
            if summary_path.exists():
                continue
            result = run_alf(plan, Client(plan, repeat), dict(spec, repeat=repeat), contexts, target)
            save(summary_path, dict(case=case, repeat=repeat, design_sha256=sha(path),
                                    order=list(contexts), replay=result))
            print(json.dumps(dict(case=case, repeat=repeat,
                                  rewards={name: row['reward'] for name, row in result.items()})),
                  flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin', type=Path, required=True)
    parser.add_argument('--prior-probe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18559')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run(args.origin.resolve(), args.prior_probe.resolve(), args.output.resolve(),
        args.url, args.prepare_only)


if __name__ == '__main__':
    main()
