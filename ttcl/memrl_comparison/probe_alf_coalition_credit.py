"""Read-only, frozen three-memory coalition probe on official ALFWorld train games.

Cases are selected after seeing an earlier result, so this is mechanism analysis,
not an independent performance estimate or a source of reviewed writer labels.
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


CASES = (
    'alfworld/pick_and_place_simple/92721/memrl/episode_009',
    'alfworld/pick_clean_then_place_in_recep/92721/memrl/episode_007',
)
REPEATS = (93101, 93102, 93103)


def design_for(origin: Path) -> dict:
    report = audit_source(origin)
    if not report['complete'] or report['missing'] or report['audited_pairs'] != 18:
        raise ValueError('Native source chain is incomplete')
    cases = []
    for case in CASES:
        spec, arms = memory_arms(origin, case)
        if spec['benchmark'] != 'alfworld' or len(spec['ids']) != 3:
            raise ValueError('Expected three retrieved ALFWorld memories')
        expected = {'full', 'none', *(f'only_{i}' for i in range(3)),
                    *(f'drop_{i}' for i in range(3))}
        if set(arms) != expected:
            raise ValueError('Unexpected coalition arms')
        cases.append(dict(case=case, input_sha256=spec['source_input_sha256'],
                          ids=spec['ids'], snapshot_sha256=spec['snapshot_sha256'],
                          retrieval_sha256=spec['retrieval_sha256'],
                          original_row_sha256=spec['original_memrl_row_sha256'],
                          context_sha256=spec['arm_context_sha256']))
    return dict(schema='alf_three_memory_coalition_v1', origin=str(origin),
                plan_sha256=sha(origin / 'plan.json'),
                source_design_sha256=sha(origin / 'design.json'),
                source_complete_sha256=sha(origin / 'complete.json'),
                runner_sha256=sha(Path(__file__)),
                credit_probe_sha256=sha(Path(__file__).with_name('credit_probe.py')),
                repeats=list(REPEATS), cases=cases,
                selection='Posthoc: two native train cases with full/drop-first both successful; diagnostic only',
                budget='9 arms per case and repeat; at most 50 environment actions per arm; no writer or Q updates')


def run(origin: Path, output: Path, url: str, prepare_only: bool) -> None:
    expected = design_for(origin)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'design.json'
    if path.exists():
        if read(path) != expected:
            raise ValueError('Frozen design or source changed')
    else:
        save(path, expected)
    if prepare_only:
        print(json.dumps(dict(prepared=str(path), design_sha256=sha(path))))
        return
    plan = read(origin / 'plan.json')
    plan['url'] = url
    plan['alf']['actor_url'] = url
    for case in CASES:
        spec, arms = memory_arms(origin, case)
        for repeat in REPEATS:
            names = [n for n in arms if n != 'full']
            random.Random(int(hashlib.sha256(f'{case}/{repeat}'.encode()).hexdigest(), 16)).shuffle(names)
            contexts = {'full': arms['full']}
            contexts.update((name, arms[name]) for name in names)
            contexts['full_repeat'] = arms['full']
            target = output / case / f'actor_repeat_{repeat}'
            target.mkdir(parents=True, exist_ok=True)
            order_path = target / 'order.json'
            order = dict(names=list(contexts), seed=repeat)
            if order_path.exists() and read(order_path) != order:
                raise ValueError('Frozen order changed')
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
