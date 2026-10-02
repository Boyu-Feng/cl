"""Summarize paired general-evidence pilot runs without filling failed cells."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from ttcl.icl_mem0_comparison.protocol import read, save, sha


def analyze(directories):
    out = {}
    for directory in directories:
        design = read(directory / 'design.json')
        benchmark, task, repeat = (design[k] for k in ('benchmark', 'task', 'repeat'))
        origin_plan = read(Path(design['origin']) / 'plan.json')
        threshold = int(.2 * origin_plan['tasks'][task]) if benchmark == 'clbench' else 0
        rows = []
        failures = []
        for index in range(design['limit']):
            folder = f'episode_{index+1:03d}'
            vanilla = read(directory / 'vanilla' / folder / 'row.json')
            improved = read(directory / 'general_evidence' / folder / 'row.json')
            if (vanilla['status'] != 'complete' or improved['status'] != 'complete'):
                failures.append(dict(index=index, vanilla=vanilla['status'],
                                     general_evidence=improved['status']))
                continue
            if index < threshold:
                continue
            measures = ('first_attempt', 'within_three') if benchmark == 'alfworld' else ('reward',)
            rows.append(dict(index=index,
                             **{measure: dict(vanilla=vanilla[measure],
                                              general_evidence=improved[measure])
                                for measure in measures}))
        measures = ('first_attempt', 'within_three') if benchmark == 'alfworld' else ('reward',)
        scores = {}
        for measure in measures:
            values = [r[measure] for r in rows]
            scores[measure] = dict(n=len(values),
                vanilla_mean=statistics.fmean(v['vanilla'] for v in values) if values else None,
                general_evidence_mean=statistics.fmean(v['general_evidence'] for v in values) if values else None,
                paired_difference=statistics.fmean(v['general_evidence']-v['vanilla'] for v in values)
                    if values else None,
                wins=sum(v['general_evidence'] > v['vanilla'] for v in values),
                losses=sum(v['general_evidence'] < v['vanilla'] for v in values),
                ties=sum(v['general_evidence'] == v['vanilla'] for v in values))
        key = f'{benchmark}/{task}/{repeat}'
        out[key] = dict(directory=str(directory), design_sha256=sha(directory / 'design.json'),
                        completed=2*design['limit'], failed_pairs=failures,
                        primary_from_index=threshold, scores=scores)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directories', type=Path, nargs='+')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = analyze([x.resolve() for x in args.directories])
    if args.output:
        save(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
