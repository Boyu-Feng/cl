"""Select new ALFWorld train retrievals for a frozen causal-credit cache.

Selection uses source content and retrieval only; new-game rewards are never
used to decide whether a case receives a cache intervention.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import read, save, sha

from .audit_credit_train_extension import audit as audit_source
from .causal_credit_cache import decision, fit
from .credit_probe import memory_arms


ACTOR_REPEATS = (92931, 92932, 92933)


def select(source: Path, cache_path: Path, donors: list[Path]) -> dict:
    audit = audit_source(source)
    if (not audit['complete'] or audit['missing'] or
            audit['audited_pairs'] != audit['expected_pairs']):
        raise ValueError('New ALFWorld source is incomplete')
    cache = read(cache_path)
    if (cache != fit(donors) or
            cache['source_code_sha256'] !=
            sha(Path(__file__).with_name('causal_credit_cache.py'))):
        raise ValueError('Donor causal-credit cache changed')
    donor_sources = {read(path / 'design.json')['origin'] for path in donors}
    if len(donor_sources) != 1:
        raise ValueError('Donor probes did not share one training source')
    donor_source = Path(next(iter(donor_sources)))
    donor_design = read(donor_source / 'design.json')
    donor_inputs = {item['sha256']
                    for games in donor_design['selected_games'].values()
                    for item in games}
    design = read(source / 'design.json')
    selected, by_index = [], {}
    multi_memory, inspected = 0, 0
    for family in design['families']:
        boot = design['bootstrap'][family]['index']
        for offset, item in enumerate(design['selected_games'][family], 1):
            index = boot + offset
            case = f'alfworld/{family}/{design["repeat"]}/memrl/episode_{index:03d}'
            row = read(source / 'runs' / case / 'row.json')
            retrieval = read(source / 'runs' / case / 'retrieval_1.json')
            if (row['status'] != 'complete' or
                    row['input_sha256'] != item['sha256'] or
                    item['sha256'] in donor_inputs):
                raise ValueError(f'New source overlaps donor or changed: {case}')
            inspected += 1
            if len(retrieval['ids']) < 2:
                continue
            multi_memory += 1
            spec, _ = memory_arms(source, case)
            result = decision(spec, cache)
            if result['drop_position'] is None:
                continue
            chosen = dict(case=case, family=family,
                          position=result['drop_position'],
                          memory_id=result['drop_memory_id'],
                          memory_text_sha256=spec['memory_text_sha256'][
                              result['drop_memory_id']],
                          input_sha256=spec['source_input_sha256'],
                          retrieval_sha256=spec['retrieval_sha256'],
                          snapshot_sha256=spec['snapshot_sha256'],
                          donor_case=result['donor']['source_case'])
            selected.append(chosen)
            by_index.setdefault(str(result['drop_position']), []).append(case)
    return dict(schema='causal_credit_cache_transfer_selection_v1',
                source=str(source), source_design_sha256=sha(source / 'design.json'),
                source_complete_sha256=sha(source / 'complete.json'),
                cache_path=str(cache_path), cache_sha256=sha(cache_path),
                donor_paths=[str(path) for path in donors],
                donor_source_design_sha256=sha(donor_source / 'design.json'),
                selection_code_sha256=sha(Path(__file__)),
                actor_repeats=list(ACTOR_REPEATS),
                inspected_games=inspected, multi_memory_games=multi_memory,
                selected=selected, by_index=by_index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--donors', nargs=3, type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = select(args.source.resolve(), args.cache.resolve(),
                    [path.resolve() for path in args.donors])
    if args.output.exists() and read(args.output) != result:
        raise ValueError('Frozen cache-transfer selection changed')
    save(args.output, result)
    print(json.dumps(dict(inspected=result['inspected_games'],
                          multi_memory=result['multi_memory_games'],
                          selected=len(result['selected']),
                          by_index={key:len(value) for key, value in
                                    result['by_index'].items()}), sort_keys=True))


if __name__ == '__main__':
    main()
