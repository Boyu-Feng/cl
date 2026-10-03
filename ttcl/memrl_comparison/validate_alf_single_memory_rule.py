"""Frozen cross-input validation of an exploratory ALFWorld memory-sparsity rule.

The rule was chosen on group5: retain only the third retrieved memory when
exactly three are present. Here each family contributes a content-distinct
group4 official-train input selected by minimum input SHA-256. Each arm is
run twice per actor seed in a balanced ABBA or BAAB order. Read-only probe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import Client, read, save, sha
from .audit_credit_train_extension import audit as audit_source
from .credit_probe import memory_arms, run_alf


REPEATS = (93301, 93302, 93303)
ORDERS = (('full', 'only_2', 'only_2_repeat', 'full_repeat'),
          ('only_2', 'full', 'full_repeat', 'only_2_repeat'))


def selected_cases(origin: Path) -> list[str]:
    design = read(origin / 'design.json')
    cases = []
    for family, games in sorted(design['selected_games'].items()):
        eligible = []
        for offset, game in enumerate(games, 1):
            index = design['bootstrap'][family]['index'] + offset
            case = f'alfworld/{family}/{design["repeat"]}/memrl/episode_{index:03d}'
            if len(read(origin / 'runs' / case / 'retrieval_1.json')['ids']) == 3:
                eligible.append((game['sha256'], case))
        if not eligible:
            raise ValueError(f'No three-memory validation input: {family}')
        cases.append(min(eligible)[1])
    if len(cases) != 6:
        raise ValueError('Expected six official ALFWorld train families')
    return cases


def frozen_design(origin: Path, exploration: Path) -> dict:
    source = audit_source(origin)
    if not source['complete'] or source['missing'] or source['audited_pairs'] != 18:
        raise ValueError('Validation source is incomplete')
    prior = read(exploration / 'summary.json')
    if prior['schema'] != 'alf_three_memory_coalition_summary_v1':
        raise ValueError('Missing exploration provenance')
    cases = []
    exploration_origin = Path(read(exploration / 'design.json')['origin'])
    exploration_hashes = {game['sha256'] for games in
                          read(exploration_origin / 'design.json')['selected_games'].values()
                          for game in games}
    for case in selected_cases(origin):
        spec, arms = memory_arms(origin, case)
        if len(spec['ids']) != 3 or spec['source_input_sha256'] in exploration_hashes:
            raise ValueError('Validation input overlaps exploration or changed retrieval')
        cases.append(dict(case=case, input_sha256=spec['source_input_sha256'],
                          ids=spec['ids'], snapshot_sha256=spec['snapshot_sha256'],
                          retrieval_sha256=spec['retrieval_sha256'],
                          original_row_sha256=spec['original_memrl_row_sha256'],
                          full_sha256=spec['arm_context_sha256']['full'],
                          only_2_sha256=spec['arm_context_sha256']['only_2']))
    return dict(schema='alf_single_memory_rule_validation_v1',
                origin=str(origin), exploration=str(exploration),
                exploration_design_sha256=sha(exploration / 'design.json'),
                exploration_summary_sha256=sha(exploration / 'summary.json'),
                source_plan_sha256=sha(origin / 'plan.json'),
                source_design_sha256=sha(origin / 'design.json'),
                source_complete_sha256=sha(origin / 'complete.json'),
                runner_sha256=sha(Path(__file__)),
                credit_probe_sha256=sha(Path(__file__).with_name('credit_probe.py')),
                repeats=list(REPEATS), cases=cases,
                rule='Exactly three retrieved memories: actor context is only index 2. Fixed snapshot first attempt, no writer or Q changes.',
                selection='One minimum input-content SHA-256 eligible group4 case per family; no group4 outcome read',
                budget='Four arms per game/seed, at most 50 ALFWorld actions per arm')


def run(origin: Path, exploration: Path, output: Path, url: str,
        prepare_only: bool) -> None:
    design = frozen_design(origin, exploration)
    output.mkdir(parents=True, exist_ok=True)
    design_path = output / 'design.json'
    if design_path.exists():
        if read(design_path) != design:
            raise ValueError('Frozen validation selection changed')
    else:
        save(design_path, design)
    if prepare_only:
        print(json.dumps(dict(design_sha256=sha(design_path),
                              cases=[item['case'] for item in design['cases']])))
        return
    plan = read(origin / 'plan.json')
    plan['url'] = url
    plan['alf']['actor_url'] = url
    for item in design['cases']:
        case = item['case']
        spec, arms = memory_arms(origin, case)
        for repeat in REPEATS:
            index = int(hashlib.sha256(f'{case}/{repeat}'.encode()).hexdigest(), 16) % 2
            names = ORDERS[index]
            contexts = {name: arms['full' if name.startswith('full') else 'only_2']
                        for name in names}
            target = output / case / f'actor_repeat_{repeat}'
            target.mkdir(parents=True, exist_ok=True)
            order_path = target / 'order.json'
            order = dict(names=list(names), seed=repeat)
            if order_path.exists() and read(order_path) != order:
                raise ValueError('Frozen balanced order changed')
            save(order_path, order)
            summary_path = target / 'summary.json'
            if summary_path.exists():
                continue
            result = run_alf(plan, Client(plan, repeat), dict(spec, repeat=repeat),
                             contexts, target)
            save(summary_path, dict(case=case, repeat=repeat,
                                    design_sha256=sha(design_path),
                                    order=list(names), replay=result))
            print(json.dumps(dict(case=case, repeat=repeat,
                                  rewards={name: row['reward'] for name, row in result.items()})), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin', type=Path, required=True)
    parser.add_argument('--exploration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18559')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run(args.origin.resolve(), args.exploration.resolve(), args.output.resolve(),
        args.url, args.prepare_only)


if __name__ == '__main__':
    main()
