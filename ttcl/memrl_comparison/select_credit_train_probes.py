"""Materialize outcome-blind train-split per-memory probe cases from a frozen rule."""
from __future__ import annotations

import argparse
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import read, save, sha
from ttcl.memrl_comparison.credit_probe import memory_arms


def select(source: Path) -> dict:
    code_freeze = read(source / 'probe_selection_code_sha256.json')
    if sha(Path(__file__)) != code_freeze['sha256']:
        raise ValueError('Frozen train-probe selection code changed')
    if sha(Path(memory_arms.__code__.co_filename)) != code_freeze['credit_probe_sha256']:
        raise ValueError('Frozen train-probe context reconstruction changed')
    rule_path = source / 'probe_selection_rule.json'
    rule = read(rule_path)
    if sha(source / 'plan.json') != rule['source_plan_sha256']:
        raise ValueError('Train source plan changed')
    if sha(source / 'design.json') != rule['source_design_sha256']:
        raise ValueError('Train source design changed')
    complete = read(source / 'complete.json')
    if complete['completed'] != complete['expected'] or complete['design_sha256'] != rule['source_design_sha256']:
        raise ValueError('Train source is incomplete or design changed')
    design = read(source / 'design.json')
    cases, missing, by_index = [], [], {}
    for family in design['families']:
        eligible = []
        for index, item in enumerate(design['selected_games'][family], 1):
            case = f'alfworld/{family}/{design["repeat"]}/memrl/episode_{index:03d}'
            episode = source / 'runs' / case
            row = read(episode / 'row.json')
            if row['input_sha256'] != item['sha256']:
                raise ValueError(f'Train source input changed: {case}')
            if row['status'] != 'complete':
                missing.append(dict(case=case, reason='source_cell_failed'))
                continue
            retrieval = read(episode / 'retrieval_1.json')
            if len(retrieval['ids']) >= 2:
                eligible.append(case)
        if not eligible:
            missing.append(dict(family=family, reason='no_eligible_retrieval'))
            continue
        chosen = list(dict.fromkeys((eligible[0], eligible[-1])))
        for case in chosen:
            spec, _ = memory_arms(source, case)
            cases.append(dict(case=case, family=family, ids=spec['ids'],
                              input_sha256=spec['source_input_sha256'],
                              retrieval_sha256=spec['retrieval_sha256'],
                              snapshot_sha256=spec['snapshot_sha256']))
            for position in range(len(spec['ids'])):
                by_index.setdefault(str(position), []).append(case)
    return dict(rule_sha256=sha(rule_path), selection_code_sha256=code_freeze['sha256'],
                context_reconstruction_sha256=code_freeze['credit_probe_sha256'],
                source_plan_sha256=sha(source / 'plan.json'),
                source_complete_sha256=sha(source / 'complete.json'),
                actor_repeats=rule['actor_repeats'], cases=cases,
                by_index=by_index, missing=missing,
                note='Selection inspects source status and first-attempt retrieval counts, never source rewards.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    report = select(source)
    output = source / 'selected_probe_cases.json'
    if output.exists() and read(output) != report:
        raise ValueError('Frozen train-probe selection changed')
    save(output, report)
    print(f"Selected {len(report['cases'])} train source cases; "
          f"memory positions {[(key,len(value)) for key,value in report['by_index'].items()]}")


if __name__ == '__main__':
    main()
