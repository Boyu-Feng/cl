"""Summarize one-policy CLBench paired evaluation on common valid suffix cells."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from ttcl.icl_mem0_comparison.protocol import read, save, sha


def summarize(origin: Path, output: Path):
    plan = read(origin / 'plan.json')
    result = {}
    for task, count in plan['tasks'].items():
        pairs, failures, runs = [], [], {}
        for repeat in plan['repeats']:
            folder = output / f'{task}_{repeat}'
            design_path = folder / 'design.json'
            if not design_path.exists():
                runs[str(repeat)] = dict(recorded=0, expected=2*count, complete=False)
                continue
            design = read(design_path)
            if (design['task'], design['repeat'], design['limit']) != (task, repeat, count):
                raise ValueError(f'Run design mismatch: {folder}')
            if design['origin_plan_sha256'] != sha(origin / 'plan.json'):
                raise ValueError(f'Origin plan mismatch: {folder}')
            frozen = folder / 'source' / 'universal_evidence.py'
            if frozen.exists() and sha(frozen) != design['implementation_sha256']:
                raise ValueError(f'Frozen method changed: {folder}')
            recorded = 0
            for index in range(count):
                rows = []
                for policy in ('vanilla', 'universal_evidence'):
                    path = folder / policy / f'episode_{index+1:03d}' / 'row.json'
                    if path.exists():
                        row = read(path)
                        if row['initial_query_sha256'] != design['bindings'][index]:
                            raise ValueError(f'Input binding changed: {path}')
                        recorded += 1
                        rows.append(row)
                if index < int(.2*count) or len(rows) != 2:
                    continue
                if not all(r['status'] == 'complete' and isinstance(r.get('reward'), (int, float))
                           and math.isfinite(r['reward']) for r in rows):
                    failures.append(dict(repeat=repeat, index=index,
                                         statuses=[r['status'] for r in rows]))
                    continue
                pairs.append(dict(repeat=repeat, index=index, vanilla=rows[0]['reward'],
                                  universal_evidence=rows[1]['reward']))
            runs[str(repeat)] = dict(recorded=recorded, expected=2*count,
                                     complete=recorded == 2*count and (folder/'rows.json').exists())
        deltas = [r['universal_evidence']-r['vanilla'] for r in pairs]
        result[task] = dict(scored_pairs=len(pairs), expected_pairs=len(plan['repeats'])*
                            (count-int(.2*count)),
                            vanilla=statistics.fmean(r['vanilla'] for r in pairs) if pairs else None,
                            universal_evidence=statistics.fmean(r['universal_evidence'] for r in pairs) if pairs else None,
                            difference=statistics.fmean(deltas) if deltas else None,
                            wins=sum(d>0 for d in deltas), losses=sum(d<0 for d in deltas),
                            ties=sum(d==0 for d in deltas), failures=failures, runs=runs)
    result['complete'] = all(run['complete'] for task in plan['tasks']
                             for run in result[task]['runs'].values())
    save(output / 'summary.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--origin', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(summarize(args.origin.resolve(), args.output.resolve()), indent=2))


if __name__ == '__main__':
    main()
