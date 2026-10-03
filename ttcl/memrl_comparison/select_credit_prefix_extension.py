"""Freeze new Poker calibration-prefix credit probes by source hash ranking.

The rule excludes previously probed cases and reads status/retrieval only.
It never uses source or counterfactual rewards to rank eligible cases.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import read, save, sha

from .credit_probe import memory_arms


ACTOR_REPEATS = (92811, 92812, 92813)
SALT = 'paired-credit-poker-new-inputs-v1'


def select(origin: Path, prior: Path, repeat: int, count: int = 6) -> dict:
    if count < 1:
        raise ValueError('Count must be positive')
    plan = read(origin / 'plan.json')
    previous = read(prior)
    if (previous['origin'] != str(origin) or
            previous['origin_plan_sha256'] != sha(origin / 'plan.json') or
            previous['repeat'] != repeat):
        raise ValueError('Previous selection is from another source')
    task = 'exploitable_poker'
    excluded = {row['case'] for row in previous['cases']}
    eligible = []
    for index in range(1, int(.2 * plan['tasks'][task]) + 1):
        case = f'clbench/{task}/{repeat}/memrl/episode_{index:03d}'
        if case in excluded:
            continue
        folder = origin / 'runs' / case
        if not all((folder / name).exists() for name in
                   ('row.json', 'retrieval.json', 'update.json')):
            continue
        row = read(folder / 'row.json')
        retrieval = read(folder / 'retrieval.json')
        if row['status'] != 'complete' or not retrieval['ids']:
            continue
        spec, _ = memory_arms(origin, case)
        priority = hashlib.sha256((SALT + ':' + spec['source_input_sha256'])
                                  .encode()).hexdigest()
        eligible.append((priority, case, spec))
    if len(eligible) < count:
        raise ValueError('Not enough new complete prefix inputs')
    chosen = sorted(eligible)[:count]
    cases = [dict(case=case, task=task, memory_ids=spec['ids'],
                  source_input_sha256=spec['source_input_sha256'],
                  snapshot_sha256=spec['snapshot_sha256'],
                  retrieval_sha256=spec['retrieval_sha256'],
                  source_row_sha256=spec['original_memrl_row_sha256'])
             for _, case, spec in chosen]
    return dict(origin=str(origin), origin_plan_sha256=sha(origin / 'plan.json'),
                selection_code_sha256=sha(Path(__file__)),
                prior_selection=str(prior), prior_selection_sha256=sha(prior),
                repeat=repeat, actor_repeats=list(ACTOR_REPEATS), cases=cases,
                eligible_count=len(eligible), salt=SALT,
                rule='Exclude old probes; rank complete first-20%-prefix Poker inputs with nonempty native retrieval by salted SHA-256 of input content; take six. Source rewards never used.',
                target='Fixed-snapshot full-versus-single-deletion official reward; three paired actor seeds.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin', type=Path, required=True)
    parser.add_argument('--prior-selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeat', type=int, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    selection = select(args.origin.resolve(), args.prior_selection.resolve(),
                       args.repeat)
    if output.exists() and read(output) != selection:
        raise ValueError('Existing frozen selection changed')
    save(output, selection)
    print([row['case'] for row in selection['cases']])


if __name__ == '__main__':
    main()
