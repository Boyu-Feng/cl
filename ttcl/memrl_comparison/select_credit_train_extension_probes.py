"""Select one new official-train memory probe per ALFWorld family by hash."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import read, save, sha

from .audit_credit_train_extension import audit as audit_source
from .credit_probe import memory_arms


ACTOR_REPEATS = (92831, 92832, 92833)
SALT = 'paired-credit-new-alf-inputs-v1'


def select(source: Path) -> dict:
    source_audit = audit_source(source)
    if (not source_audit['complete'] or source_audit['missing'] or
            source_audit['audited_pairs'] != source_audit['expected_pairs']):
        raise ValueError('New ALFWorld train source is incomplete')
    design = read(source / 'design.json')
    cases, by_index, missing = [], {}, []
    for family in design['families']:
        eligible = []
        boot = design['bootstrap'][family]['index']
        for offset, item in enumerate(design['selected_games'][family], 1):
            index = boot + offset
            case = f'alfworld/{family}/{design["repeat"]}/memrl/episode_{index:03d}'
            row = read(source / 'runs' / case / 'row.json')
            retrieval = read(source / 'runs' / case / 'retrieval_1.json')
            if row['status'] != 'complete' or row['input_sha256'] != item['sha256']:
                raise ValueError(f'New source cell changed: {case}')
            if len(retrieval['ids']) < 2:
                continue
            priority = hashlib.sha256((SALT + ':' + item['sha256'])
                                      .encode()).hexdigest()
            eligible.append((priority, case))
        if not eligible:
            missing.append(dict(family=family,
                                reason='no_new_first_attempt_multi_memory_case'))
            continue
        _, case = min(eligible)
        spec, _ = memory_arms(source, case)
        cases.append(dict(case=case, family=family, ids=spec['ids'],
                          input_sha256=spec['source_input_sha256'],
                          retrieval_sha256=spec['retrieval_sha256'],
                          snapshot_sha256=spec['snapshot_sha256']))
        for position in range(len(spec['ids'])):
            by_index.setdefault(str(position), []).append(case)
    return dict(source=str(source), source_plan_sha256=sha(source / 'plan.json'),
                source_design_sha256=sha(source / 'design.json'),
                source_complete_sha256=sha(source / 'complete.json'),
                selection_code_sha256=sha(Path(__file__)),
                context_reconstruction_sha256=sha(
                    Path(memory_arms.__code__.co_filename)),
                actor_repeats=list(ACTOR_REPEATS), salt=SALT,
                rule='Per family, take the SHA-256-minimal new official-train input among completed cases with at least two first-attempt native retrieved memories; no reward selection.',
                cases=cases, by_index=by_index, missing=missing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    result = select(args.source.resolve())
    if output.exists() and read(output) != result:
        raise ValueError('Frozen ALF extension selection changed')
    save(output, result)
    print(f"Selected {len(result['cases'])} new train inputs; "
          f"memory positions {[(key, len(value)) for key, value in result['by_index'].items()]}; "
          f"missing families {len(result['missing'])}")


if __name__ == '__main__':
    main()
