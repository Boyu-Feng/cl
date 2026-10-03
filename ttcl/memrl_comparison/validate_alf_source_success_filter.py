"""Test a frozen source-success memory filter on distinct ALFWorld train inputs.

When exactly three memories are retrieved, keep those whose source task
succeeded; retain the complete context if none or all succeeded. This is a
read-only first-attempt probe, not an online MemRL policy or training label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import Client, read, save, sha
from .audit_credit_train_extension import audit as audit_source
from .credit_probe import memory_arms, run_alf


REPEATS = (93401, 93402, 93403)
ORDERS = (('full', 'filtered', 'filtered_repeat', 'full_repeat'),
          ('filtered', 'full', 'full_repeat', 'filtered_repeat'))


def filtered_context(spec: dict, arms: dict) -> tuple[list[int], str]:
    if len(spec['ids']) != 3:
        raise ValueError('Source-success probe requires three memories')
    kept = [i for i, mid in enumerate(spec['ids'])
            if spec['memory_features'][mid]['metadata']['success'] is True]
    if not kept or len(kept) == 3:
        return list(range(3)), arms['full']
    return kept, '\n\n'.join(arms[f'only_{i}'] for i in kept)


def selected_cases(origin: Path) -> list[str]:
    design = read(origin / 'design.json')
    cases = []
    for family, games in sorted(design['selected_games'].items()):
        eligible = []
        for offset, game in enumerate(games, 1):
            index = design['bootstrap'][family]['index'] + offset
            case = f'alfworld/{family}/{design["repeat"]}/memrl/episode_{index:03d}'
            retrieval = read(origin / 'runs' / case / 'retrieval_1.json')
            if len(retrieval['ids']) != 3:
                continue
            spec, arms = memory_arms(origin, case)
            kept, _ = filtered_context(spec, arms)
            if 0 < len(kept) < 3:
                eligible.append((game['sha256'], case))
        if not eligible:
            raise ValueError(f'No nontrivial source-success filter input: {family}')
        cases.append(min(eligible)[1])
    if len(cases) != 6:
        raise ValueError('Expected six ALFWorld official train families')
    return cases


def frozen_design(origin: Path, exploration: Path, prior_validation: Path) -> dict:
    source = audit_source(origin)
    if not source['complete'] or source['missing'] or source['audited_pairs'] != 18:
        raise ValueError('Validation source incomplete')
    exploration_origin = Path(read(exploration / 'design.json')['origin'])
    prior_origin = Path(read(prior_validation / 'design.json')['origin'])
    old_hashes = {game['sha256'] for earlier in (exploration_origin, prior_origin)
                  for games in read(earlier / 'design.json')['selected_games'].values()
                  for game in games}
    cases = []
    for case in selected_cases(origin):
        spec, arms = memory_arms(origin, case)
        kept, context = filtered_context(spec, arms)
        if spec['source_input_sha256'] in old_hashes:
            raise ValueError('Validation input overlaps prior source group')
        cases.append(dict(case=case, input_sha256=spec['source_input_sha256'],
                          ids=spec['ids'], kept_indices=kept,
                          snapshot_sha256=spec['snapshot_sha256'],
                          retrieval_sha256=spec['retrieval_sha256'],
                          original_row_sha256=spec['original_memrl_row_sha256'],
                          full_sha256=spec['arm_context_sha256']['full'],
                          filtered_sha256=hashlib.sha256(context.encode()).hexdigest()))
    return dict(schema='alf_source_success_filter_validation_v1',
                origin=str(origin), exploration=str(exploration),
                prior_validation=str(prior_validation),
                exploration_design_sha256=sha(exploration / 'design.json'),
                exploration_summary_sha256=sha(exploration / 'summary.json'),
                prior_design_sha256=sha(prior_validation / 'design.json'),
                prior_analysis_sha256=sha(prior_validation / 'analysis.json'),
                source_plan_sha256=sha(origin / 'plan.json'),
                source_design_sha256=sha(origin / 'design.json'),
                source_complete_sha256=sha(origin / 'complete.json'),
                runner_sha256=sha(Path(__file__)),
                credit_probe_sha256=sha(Path(__file__).with_name('credit_probe.py')),
                repeats=list(REPEATS), cases=cases,
                rule='Keep all retrieved memories with source success=True in original order; if none or all, retain full context',
                selection='Per family: minimum input SHA-256 among group3 native train cases with exactly three retrieved memories and a nontrivial filter; no group3 outcome consulted',
                budget='Four ABBA/BAAB arms per game and seed, at most 50 ALFWorld actions each')


def run(origin: Path, exploration: Path, prior_validation: Path, output: Path,
        url: str, prepare_only: bool) -> None:
    design = frozen_design(origin, exploration, prior_validation)
    output.mkdir(parents=True, exist_ok=True)
    design_path = output / 'design.json'
    if design_path.exists():
        if read(design_path) != design:
            raise ValueError('Frozen source-success filter design changed')
    else:
        save(design_path, design)
    if prepare_only:
        print(json.dumps(dict(design_sha256=sha(design_path),
                              cases=[(item['case'], item['kept_indices'])
                                     for item in design['cases']])))
        return
    plan = read(origin / 'plan.json')
    plan['url'] = url
    plan['alf']['actor_url'] = url
    for item in design['cases']:
        case = item['case']
        spec, arms = memory_arms(origin, case)
        _, filtered = filtered_context(spec, arms)
        for repeat in REPEATS:
            index = int(hashlib.sha256(f'{case}/{repeat}'.encode()).hexdigest(), 16) % 2
            names = ORDERS[index]
            contexts = {name: arms['full'] if name.startswith('full') else filtered
                        for name in names}
            target = output / case / f'actor_repeat_{repeat}'
            target.mkdir(parents=True, exist_ok=True)
            order_path = target / 'order.json'
            order = dict(names=list(names), seed=repeat)
            if order_path.exists() and read(order_path) != order:
                raise ValueError('Balanced arm order changed')
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
    parser.add_argument('--prior-validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18559')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run(args.origin.resolve(), args.exploration.resolve(),
        args.prior_validation.resolve(), args.output.resolve(), args.url,
        args.prepare_only)


if __name__ == '__main__':
    main()
