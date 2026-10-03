"""Summarize one frozen paired grounded-evidence run without filling gaps."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from ttcl.icl_mem0_comparison.protocol import read, sha


POLICIES = ('vanilla', 'grounded_evidence')


def analyze(output: Path):
    design = read(output / 'design.json')
    origin = Path(design['origin'])
    if sha(origin / 'plan.json') != design['origin_plan_sha256']:
        raise ValueError('Origin plan changed')
    frozen = output / 'source' / 'grounded_evidence.py'
    if sha(frozen) != design['implementation_sha256']:
        raise ValueError('Frozen candidate changed')
    plan = read(origin / 'plan.json')
    threshold = (int(.2 * plan['tasks'][design['task']])
                 if design['benchmark'] == 'clbench' else 0)
    scored, failed, coverage = [], [], []
    for index, binding in enumerate(design['bindings']):
        rows = []
        for policy in POLICIES:
            episode = output / policy / f'episode_{index+1:03d}'
            path = episode / 'row.json'
            if not path.exists():
                rows = []
                break
            row = read(path)
            expected = ('input_sha256' if design['benchmark'] == 'alfworld'
                        else 'initial_query_sha256')
            if row[expected] != binding:
                raise ValueError(f'Input binding changed: {path}')
            if sha(episode / 'memory_after.json') != row['memory_after_sha256']:
                raise ValueError(f'Memory snapshot changed: {path}')
            rows.append(row)
        if len(rows) != 2:
            continue
        coverage.append(index)
        if index < threshold:
            continue
        if not all(row['status'] == 'complete' and
                   isinstance(row.get('reward'), (int, float)) and
                   math.isfinite(row['reward']) for row in rows):
            failed.append(dict(index=index, statuses=[r['status'] for r in rows]))
            continue
        scored.append(dict(index=index, vanilla=rows[0]['reward'],
                           grounded_evidence=rows[1]['reward'],
                           actor_calls=[row['actor_calls'] for row in rows],
                           actor_input_tokens=[row['actor_input_tokens'] for row in rows]))
    deltas = [r['grounded_evidence'] - r['vanilla'] for r in scored]
    return dict(benchmark=design['benchmark'], task=design['task'],
                repeat=design['repeat'], recorded_pairs=len(coverage),
                planned_pairs=design['limit'], scored_pairs=len(scored),
                planned_suffix_pairs=max(0, design['limit'] - threshold),
                complete=len(coverage) == design['limit'] and
                         (output / 'rows.json').exists(),
                vanilla=statistics.fmean(r['vanilla'] for r in scored) if scored else None,
                grounded_evidence=statistics.fmean(r['grounded_evidence'] for r in scored)
                                  if scored else None,
                difference=statistics.fmean(deltas) if deltas else None,
                wins=sum(x > 0 for x in deltas), losses=sum(x < 0 for x in deltas),
                ties=sum(x == 0 for x in deltas), failures=failed,
                paired_actor_calls=[sum(r['actor_calls'][i] for r in scored)
                                    for i in (0, 1)],
                paired_actor_input_tokens=[sum(r['actor_input_tokens'][i] for r in scored)
                                           for i in (0, 1)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(analyze(args.output.resolve()), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
