"""Outcome-blind three-memory coalition probe on the new group7 train chain.

One content-distinct task per family is selected by minimum input hash. This
is a fixed-snapshot mechanism probe; it does not train or update MemRL.
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


REPEATS = (93601, 93602, 93603)


def selected_cases(origin: Path) -> list[str]:
    design = read(origin / 'design.json')
    if design['group_index'] != 6:
        raise ValueError('Expected the seventh official-train group')
    repeat = design['repeat']
    selected = []
    for family, games in sorted(design['selected_games'].items()):
        eligible = []
        for offset, game in enumerate(games, 1):
            index = design['bootstrap'][family]['index'] + offset
            case = f'alfworld/{family}/{repeat}/memrl/episode_{index:03d}'
            retrieval = read(origin / 'runs' / case / 'retrieval_1.json')
            if len(retrieval['ids']) == 3:
                eligible.append((game['sha256'], case))
        if not eligible:
            raise ValueError(f'No three-memory input: {family}')
        selected.append(min(eligible)[1])
    if len(selected) != 6:
        raise ValueError('Expected exactly six ALFWorld task families')
    return selected


def design_for(origin: Path) -> dict:
    report = audit_source(origin)
    if not report['complete'] or report['missing'] or report['audited_pairs'] != 18:
        raise ValueError('Native source chain is incomplete')
    current_hashes = {game['sha256'] for games in read(origin / 'design.json')['selected_games'].values()
                      for game in games}
    if len(current_hashes) != 18:
        raise ValueError('Duplicate group7 task content')
    prior_hashes = {}
    for group in ('3', '4', '5', '6'):
        previous = origin.parent / f'20261003_alf_train_credit_extension_group{group}_v1' / 'design.json'
        prior_hashes[group] = sha(previous)
        earlier = {game['sha256'] for games in read(previous)['selected_games'].values()
                   for game in games}
        if current_hashes & earlier:
            raise ValueError(f'Group7 overlaps group{group} input content')
    cases = []
    for case in selected_cases(origin):
        spec, arms = memory_arms(origin, case)
        if len(spec['ids']) != 3 or len(arms) != 8:
            raise ValueError('Expected eight coalitions of three memories')
        cases.append(dict(case=case, input_sha256=spec['source_input_sha256'],
                          ids=spec['ids'], snapshot_sha256=spec['snapshot_sha256'],
                          retrieval_sha256=spec['retrieval_sha256'],
                          original_row_sha256=spec['original_memrl_row_sha256'],
                          context_sha256=spec['arm_context_sha256']))
    return dict(schema='alf_three_memory_coalition_group7_v1', origin=str(origin),
                plan_sha256=sha(origin / 'plan.json'),
                source_design_sha256=sha(origin / 'design.json'),
                source_complete_sha256=sha(origin / 'complete.json'),
                prior_design_sha256=prior_hashes,
                runner_sha256=sha(Path(__file__)),
                credit_probe_sha256=sha(Path(__file__).with_name('credit_probe.py')),
                repeats=list(REPEATS), cases=cases,
                selection='Within each official-train family, minimum input SHA-256 among group7 cases with exactly three retrieved memories; no outcomes consulted',
                budget='Nine arms per case and repeat; at most 50 environment actions per arm; no writer or Q updates')


def run(origin: Path, output: Path, url: str, prepare_only: bool) -> None:
    design = design_for(origin)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'design.json'
    if path.exists():
        if read(path) != design:
            raise ValueError('Frozen selection or source changed')
    else:
        save(path, design)
    if prepare_only:
        print(json.dumps(dict(prepared=str(path), design_sha256=sha(path),
                              cases=[x['case'] for x in design['cases']])))
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
                                  rewards={name: row['reward'] for name, row in result.items()})), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18559')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run(args.origin.resolve(), args.output.resolve(), args.url, args.prepare_only)


if __name__ == '__main__':
    main()
