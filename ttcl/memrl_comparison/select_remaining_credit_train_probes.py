"""Freeze every unprobed eligible ALFWorld train-source memory case.

This second wave is selected by retrieval count and exclusion from the first
frozen wave only.  It does not inspect rewards or probe outcomes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .credit_probe import memory_arms


def select(source: Path) -> dict:
    complete = read(source / 'complete.json')
    design = read(source / 'design.json')
    first = read(source / 'selected_probe_cases.json')
    if (complete['completed'] != complete['expected'] or
            complete['design_sha256'] != sha(source / 'design.json') or
            first['source_complete_sha256'] != sha(source / 'complete.json')):
        raise ValueError('ALF train source or first selection changed')
    excluded = {item['case'] for item in first['cases']}
    cases, by_index = [], {}
    for family in design['families']:
        for index, item in enumerate(design['selected_games'][family], 1):
            case = f'alfworld/{family}/{design["repeat"]}/memrl/episode_{index:03d}'
            if case in excluded:
                continue
            episode = source / 'runs' / case
            row = read(episode / 'row.json')
            if row['status'] != 'complete' or row['input_sha256'] != item['sha256']:
                raise ValueError(f'Incomplete or changed source: {case}')
            if len(read(episode / 'retrieval_1.json')['ids']) < 2:
                continue
            spec, _ = memory_arms(source, case)
            cases.append(dict(case=case, family=family, ids=spec['ids'],
                              input_sha256=spec['source_input_sha256'],
                              retrieval_sha256=spec['retrieval_sha256'],
                              snapshot_sha256=spec['snapshot_sha256']))
            for position in range(len(spec['ids'])):
                by_index.setdefault(str(position), []).append(case)
    return dict(source=str(source), source_plan_sha256=sha(source / 'plan.json'),
                source_complete_sha256=sha(source / 'complete.json'),
                first_selection_sha256=sha(source / 'selected_probe_cases.json'),
                selection_code_sha256=sha(Path(__file__)),
                context_reconstruction_sha256=sha(Path(memory_arms.__code__.co_filename)),
                actor_repeats=first['actor_repeats'], cases=cases,
                by_index=by_index,
                rule='Every remaining completed official-train case with >=2 first-attempt retrieved memories; no reward inspection.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    selection = select(args.source.resolve())
    if output.exists() and read(output) != selection:
        raise ValueError('Remaining train-probe selection changed')
    save(output, selection)
    print(f"Selected {len(selection['cases'])} remaining train cases; "
          f"positions {[(key, len(value)) for key, value in selection['by_index'].items()]}")


if __name__ == '__main__':
    main()
