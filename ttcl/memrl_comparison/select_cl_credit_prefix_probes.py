"""Freeze outcome-blind CLBench calibration-prefix memory probe cases."""
from __future__ import annotations

import argparse
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import read, save, sha
from ttcl.memrl_comparison.credit_probe import memory_arms


TASKS = ('blind_spectrum_monitoring', 'exploitable_poker',
         'database_exploration', 'cohort_studies')
ACTOR_REPEATS = (92751, 92752, 92753)


def select(origin: Path, repeat: int) -> dict:
    plan = read(origin / 'plan.json')
    cases, missing = [], []
    for task in TASKS:
        eligible = []
        prefix = int(.2 * plan['tasks'][task])
        for index in range(1, prefix + 1):
            case = f'clbench/{task}/{repeat}/memrl/episode_{index:03d}'
            episode = origin / 'runs' / case
            row_path, retrieval_path = episode / 'row.json', episode / 'retrieval.json'
            if not row_path.exists() or not retrieval_path.exists():
                continue
            row, retrieval = read(row_path), read(retrieval_path)
            if (row['status'] == 'complete' and retrieval['ids'] and
                    (episode / 'update.json').exists()):
                eligible.append((case, len(retrieval['ids'])))
        if not eligible:
            missing.append(dict(task=task, reason='no_complete_prefix_memory_source'))
            continue
        chosen = [eligible[0][0], eligible[-1][0]]
        multiple = next((case for case, count in eligible if count >= 2), None)
        if multiple:
            chosen.append(multiple)
        for case in dict.fromkeys(chosen):
            spec, _ = memory_arms(origin, case)
            cases.append(dict(case=case, task=task, memory_ids=spec['ids'],
                              source_input_sha256=spec['source_input_sha256'],
                              snapshot_sha256=spec['snapshot_sha256'],
                              retrieval_sha256=spec['retrieval_sha256'],
                              source_row_sha256=spec['original_memrl_row_sha256']))
    return dict(origin=str(origin), origin_plan_sha256=sha(origin / 'plan.json'),
                selection_code_sha256=sha(Path(__file__)), repeat=repeat,
                actor_repeats=list(ACTOR_REPEATS), cases=cases, missing=missing,
                rule='Per domain: earliest and latest completed calibration-prefix episode with >=1 retrieved memory; also earliest with >=2 if any. Selection ignores rewards.',
                target='Paired official reward difference for full vs leave-one-out memory context, fixed source and actor seed.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeat', type=int, default=404)
    args = parser.parse_args()
    result = select(args.origin.resolve(), args.repeat)
    output = args.output.resolve()
    if output.exists() and read(output) != result:
        raise ValueError('Frozen CL prefix selection changed')
    save(output, result)
    print(f"Selected {len(result['cases'])} CL prefix memory cases across "
          f"{len(TASKS)-len(result['missing'])} domains")


if __name__ == '__main__':
    main()
