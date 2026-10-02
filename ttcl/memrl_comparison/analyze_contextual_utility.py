"""Summarize paired contextual-utility pilot runs without filling failed cells."""
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
            improved = read(directory / 'contextual_utility' / folder / 'row.json')
            if (vanilla['status'] != 'complete' or improved['status'] != 'complete'):
                failures.append(dict(index=index, vanilla=vanilla['status'],
                                     contextual_utility=improved['status']))
                continue
            if index < threshold:
                continue
            measures = ('first_attempt', 'within_three') if benchmark == 'alfworld' else ('reward',)
            rows.append(dict(index=index,
                             **{measure: dict(vanilla=vanilla[measure],
                                              contextual_utility=improved[measure])
                                for measure in measures}))
        measures = ('first_attempt', 'within_three') if benchmark == 'alfworld' else ('reward',)
        scores = {}
        for measure in measures:
            values = [r[measure] for r in rows]
            scores[measure] = dict(n=len(values),
                vanilla_mean=statistics.fmean(v['vanilla'] for v in values) if values else None,
                contextual_utility_mean=statistics.fmean(v['contextual_utility'] for v in values) if values else None,
                paired_difference=statistics.fmean(v['contextual_utility']-v['vanilla'] for v in values)
                    if values else None,
                wins=sum(v['contextual_utility'] > v['vanilla'] for v in values),
                losses=sum(v['contextual_utility'] < v['vanilla'] for v in values),
                ties=sum(v['contextual_utility'] == v['vanilla'] for v in values))
        gate_decisions = {}
        projections = {}
        for index in range(design['limit']):
            folder = directory / 'contextual_utility' / f'episode_{index+1:03d}'
            for retrieval in folder.glob('retrieval*.json'):
                decision = read(retrieval).get('gate', 'unknown')
                gate_decisions[decision] = gate_decisions.get(decision, 0) + 1
            projected = folder / 'policy_action.json'
            if projected.exists():
                operator = read(projected)['operator']
                projections[operator] = projections.get(operator, 0) + 1
        split = design.get('alf_split') or ('planned' if benchmark == 'alfworld' else 'canonical')
        key = f'{benchmark}/{task}/{repeat}/{split}'
        out[key] = dict(directory=str(directory), design_sha256=sha(directory / 'design.json'),
                        completed=sum((directory / arm / f'episode_{i+1:03d}' / 'row.json').exists()
                                      for arm in ('vanilla','contextual_utility')
                                      for i in range(design['limit'])), failed_pairs=failures,
                        gate_decisions=gate_decisions, projections=projections,
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
