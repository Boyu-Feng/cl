"""Official replay and source audit for post-hoc same-prompt duplicates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .audit_memrl_claim_overlay import audit_episode
from .probe_memrl_claim_overlay import design_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    design = read(root / 'design.json')
    source = Path(design['source'])
    source_design = read(source / 'design.json')
    expected, static = design_for(Path(source_design['origin']),
        Path(source_design['dataset_path']), source_design['url'])
    if (design['schema'] != 'alf_memrl_claim_same_prompt_repeat_v1' or
            sha(source / 'design.json') != design['source_design_sha256'] or
            source_design != expected or
            design['static_memory'] != source_design['static_memory'] or
            design['url'] != source_design['url'] or
            sha(Path(__file__).with_name('probe_memrl_claim_repeat.py')) != design['runner_sha256']):
        raise ValueError('Frozen repeat source changed')
    rows = []
    for selection in design['cases']:
        case = source_design['cases'][selection['index']]
        repeat, index = selection['repeat'], selection['index']
        if case['input_sha256'] != selection['input_sha256']:
            raise ValueError('Repeat case content changed')
        stem = source / str(repeat) / f'{index:02d}_{case["input_sha256"][:12]}'
        base, candidate = stem / 'memrl/episode.json', stem / 'memrl_static/episode.json'
        if (sha(base) != selection['base_sha256'] or
                sha(candidate) != selection['static_sha256'] or
                read(base)['reward'] != 0 or read(candidate)['reward'] != 1):
            raise ValueError('Original post-hoc win changed')
        results = {}
        for arm, memory in [('memrl_repeat', None), ('memrl_static_repeat', static)]:
            path = root / str(repeat) / f'{index:02d}_{case["input_sha256"][:12]}' / arm / 'episode.json'
            results[arm] = audit_episode(source_design, case, repeat, arm, memory, path)
        rows.append(dict(repeat=repeat, index=index, input_sha256=case['input_sha256'],
                         original=dict(memrl=0, memrl_static=1), repeated=results))
    report = dict(schema='alf_memrl_claim_same_prompt_repeat_audit_v1',
        design_sha256=sha(root / 'design.json'), rows=rows,
        totals={arm: sum(row['repeated'][arm]['reward'] for row in rows)
                for arm in ('memrl_repeat', 'memrl_static_repeat')},
        caveat='Post-hoc selected original wins; duplicate controls test sampling noise, not independent benchmark efficacy')
    save(args.output, report)
    print(json.dumps(dict(cases=len(rows), totals=report['totals'])), flush=True)


if __name__ == '__main__':
    main()
