"""Audit complete paired Poker runs and expose heavy-tail sensitivity."""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import statistics

from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .analyze_typed_grounded import analyze


def report(outputs: list[Path], draws: int = 20000) -> dict:
    if not outputs or draws < 100:
        raise ValueError('Need complete Poker runs and enough bootstrap draws')
    runs = []
    for output in outputs:
        summary = analyze(output)
        if (summary['benchmark'] != 'clbench' or
                summary['task'] != 'exploitable_poker' or
                not summary['complete'] or summary['failures'] or
                summary['scored_pairs'] != summary['planned_suffix_pairs']):
            raise ValueError(f'Incomplete or failed Poker run: {output}')
        design = read(output / 'design.json')
        origin = Path(design['origin'])
        plan = read(origin / 'plan.json')
        threshold = int(.2 * plan['tasks']['exploitable_poker'])
        if len(design['bindings']) != plan['tasks']['exploitable_poker']:
            raise ValueError('Poker run has a non-full task budget')
        pairs = []
        for index in range(threshold, design['limit']):
            rows = [read(output / arm / f'episode_{index+1:03d}' / 'row.json')
                    for arm in ('vanilla', 'typed_grounded')]
            if (any(row['status'] != 'complete' for row in rows) or
                    rows[0]['initial_query_sha256'] != rows[1]['initial_query_sha256'] or
                    rows[0]['initial_query_sha256'] != design['bindings'][index]):
                raise ValueError('Poker scored-pair binding changed')
            pairs.append(dict(index=index, vanilla=rows[0]['reward'],
                              candidate=rows[1]['reward'],
                              delta=rows[1]['reward']-rows[0]['reward'],
                              calls=[row['actor_calls'] for row in rows],
                              input_tokens=[row['actor_input_tokens'] for row in rows]))
        runs.append(dict(output=str(output), repeat=design['repeat'],
                         design_sha256=sha(output / 'design.json'),
                         source_implementation_sha256=design['implementation_sha256'],
                         n=len(pairs), mean_delta=statistics.fmean(p['delta'] for p in pairs),
                         wins=sum(p['delta'] > 0 for p in pairs),
                         losses=sum(p['delta'] < 0 for p in pairs),
                         ties=sum(p['delta'] == 0 for p in pairs),
                         sum_delta=sum(p['delta'] for p in pairs),
                         calls=[sum(p['calls'][i] for p in pairs) for i in (0,1)],
                         input_tokens=[sum(p['input_tokens'][i] for p in pairs)
                                       for i in (0,1)],
                         worst=sorted(pairs,key=lambda p:p['delta'])[:5],
                         best=sorted(pairs,key=lambda p:p['delta'],reverse=True)[:5],
                         pairs=pairs))
    if len({run['repeat'] for run in runs}) != len(runs):
        raise ValueError('Repeated Poker seed')
    if len({run['source_implementation_sha256'] for run in runs}) != 1:
        raise ValueError('Different candidate implementations cannot be pooled')
    indices = [p['index'] for p in runs[0]['pairs']]
    if any([p['index'] for p in run['pairs']] != indices for run in runs):
        raise ValueError('Poker scored indices differ')
    by_index = {index: [next(p['delta'] for p in run['pairs'] if p['index']==index)
                        for run in runs] for index in indices}
    rng = random.Random(20261003)
    bootstrap = []
    for _ in range(draws):
        sample = [rng.choice(indices) for _ in indices]
        bootstrap.append(statistics.fmean(
            delta for index in sample for delta in by_index[index]))
    bootstrap.sort()
    all_pairs = [p for run in runs for p in run['pairs']]
    deltas = [p['delta'] for p in all_pairs]
    return dict(schema='poker_heavy_tail_audit_v1', repeats=[r['repeat'] for r in runs],
                implementation_sha256=runs[0]['source_implementation_sha256'],
                runs=runs, pooled=dict(n=len(deltas), mean_delta=statistics.fmean(deltas),
                    median_delta=statistics.median(deltas),
                    wins=sum(d > 0 for d in deltas), losses=sum(d < 0 for d in deltas),
                    ties=sum(d == 0 for d in deltas),
                    three_largest_positive_sum=sum(sorted((d for d in deltas if d > 0),
                                                          reverse=True)[:3]),
                    three_largest_negative_sum=sum(sorted(d for d in deltas if d < 0)[:3]),
                    canonical_index_cluster_bootstrap_95=[
                        bootstrap[int(.025 * draws)], bootstrap[int(.975 * draws)]],
                    bootstrap_draws=draws, bootstrap_seed=20261003),
                caveat='Official mean is primary. Bootstrap resamples shared hand indices only and cannot create independent online chains; median and top-tail sums are descriptive.')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = report([p.resolve() for p in a.runs])
    save(a.output.resolve(), result)
    print(result['pooled'])


if __name__ == '__main__':
    main()
