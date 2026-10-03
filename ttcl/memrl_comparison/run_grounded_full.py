"""Run and summarize the full paired ALFWorld/CLBench grounded-evidence evaluation."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from ttcl.icl_mem0_comparison.protocol import read, save, sha


POLICIES = ('vanilla', 'grounded_evidence')


def summarize(origin: Path, output: Path):
    plan = read(origin / 'plan.json')
    results = {}
    expected = 0
    completed = 0
    failed = 0
    grouped = {}
    for benchmark, task, repeat in jobs(plan):
        folder = output / f'{benchmark}_{task}_{repeat}'
        limit = (len(next(s['tasks'] for s in plan['alf']['sequences']
                          if s['family'] == task and s['repeat'] == repeat))
                 if benchmark == 'alfworld' else plan['tasks'][task])
        expected += 2 * limit
        arms = {}
        for policy in POLICIES:
            arms[policy] = {}
            for index in range(limit):
                path = folder / policy / f'episode_{index+1:03d}' / 'row.json'
                if not path.exists():
                    continue
                row = read(path)
                completed += 1
                if row['status'] != 'complete':
                    failed += 1
                arms[policy][index] = row
        threshold = int(plan['tasks'][task] * .2) if benchmark == 'clbench' else 0
        common = [i for i in range(threshold, limit)
                  if all(i in arms[p] and arms[p][i]['status'] == 'complete'
                         and isinstance(arms[p][i].get('reward'), (int, float))
                         and math.isfinite(arms[p][i]['reward']) for p in POLICIES)]
        for i in common:
            if arms['vanilla'][i]['initial_query_sha256'] != arms['grounded_evidence'][i]['initial_query_sha256']:
                raise ValueError(f'Paired input mismatch: {folder} episode {i+1}')
        measures = ('first_attempt', 'within_three') if benchmark == 'alfworld' else ('reward',)
        metrics = {name: {p: statistics.fmean(arms[p][i][name] for i in common) if common else None
                          for p in POLICIES} for name in measures}
        results[f'{benchmark}/{task}/{repeat}'] = dict(scored_pairs=len(common),
            expected_pairs=limit - threshold, metrics=metrics,
            failed_cells=sum(row['status'] != 'complete' for arm in arms.values() for row in arm.values()))
        group = grouped.setdefault(f'{benchmark}/{task}', {name: {p: [] for p in POLICIES}
                             for name in measures})
        for name in measures:
            for policy in POLICIES:
                group[name][policy].extend(arms[policy][i][name] for i in common)
        if benchmark == 'alfworld':
            overall = grouped.setdefault('alfworld/all', {name: {p: [] for p in POLICIES}
                                           for name in measures})
            for name in measures:
                for policy in POLICIES:
                    overall[name][policy].extend(arms[policy][i][name] for i in common)
    aggregates = {group: {name: dict(n=len(values['vanilla']),
                   means={p: statistics.fmean(values[p]) if values[p] else None for p in POLICIES})
                   for name, values in metrics.items()} for group, metrics in grouped.items()}
    result = dict(expected_cells=expected, recorded_cells=completed, failed_cells=failed,
                  results=results, aggregates=aggregates, updated_at=time.time())
    save(output / 'summary.json', result)
    return result


def jobs(plan):
    alf = [('alfworld', s['family'], s['repeat']) for s in plan['alf']['sequences']]
    cl = [('clbench', task, repeat) for repeat in plan['repeats'] for task in plan['tasks']]
    ordered = []
    while alf or cl:
        if alf:
            ordered.append(alf.pop(0))
        if cl:
            ordered.append(cl.pop(0))
    return ordered


def run(origin: Path, output: Path, url: str, workers: int, temperature: float):
    plan = read(origin / 'plan.json')
    output.mkdir(parents=True, exist_ok=True)
    design = dict(origin=str(origin), origin_plan_sha256=sha(origin / 'plan.json'),
                  runner_sha256=sha(Path(__file__)),
                  implementation_sha256=sha(Path(__file__).with_name('grounded_evidence.py')),
                  typed_projection_sha256=sha(Path(__file__).with_name('typed_projection.py')),
                  url=url, workers=workers, temperature=temperature,
                  jobs=[list(job) for job in jobs(plan)], policies=list(POLICIES))
    if (output / 'design.json').exists() and read(output / 'design.json') != design:
        raise ValueError('Frozen full-run design changed')
    save(output / 'design.json', design)
    env = dict(os.environ, TTCL_BENCH=str(origin / 'source' / 'bench'),
               PYTHONPATH=os.pathsep.join((str(Path(__file__).resolve().parents[2]),
                   str(origin / 'source' / 'bench'),
                   str(Path(__file__).resolve().parents[2] / 'ttcl/.runtime/structured_memory_deps'),
                   str(Path(__file__).resolve().parents[2] / 'ttcl/.runtime/deltamem_benchmark_deps'))),
               HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2',
               PYTHONUNBUFFERED='1')
    pending = jobs(plan)
    active = []
    done = []
    (output / 'logs').mkdir(exist_ok=True)
    while pending or active:
        while pending and len(active) < workers:
            benchmark, task, repeat = pending.pop(0)
            target = output / f'{benchmark}_{task}_{repeat}'
            command = [sys.executable, '-m', 'ttcl.memrl_comparison.evaluate_grounded_evidence',
                       '--origin', str(origin), '--output', str(target), '--benchmark', benchmark,
                       '--task', task, '--repeat', str(repeat), '--url', url,
                       '--temperature', str(temperature)]
            log = (output / 'logs' / f'{benchmark}_{task}_{repeat}.log').open('a')
            process = subprocess.Popen(command, env=env, cwd=Path(__file__).resolve().parents[2],
                                       stdout=log, stderr=subprocess.STDOUT)
            log.close()
            active.append((process, benchmark, task, repeat))
        still = []
        for process, benchmark, task, repeat in active:
            if process.poll() is None:
                still.append((process, benchmark, task, repeat))
            else:
                done.append(dict(benchmark=benchmark, task=task, repeat=repeat,
                                 exit_code=process.returncode))
        active = still
        summary = summarize(origin, output)
        save(output / 'status.json', dict(phase='running' if pending or active else
             ('complete' if all(x['exit_code'] == 0 for x in done) else 'finished_with_failures'),
             pending=pending, active=[dict(pid=p.pid, benchmark=b, task=t, repeat=r)
                                      for p, b, t, r in active], done=done,
             recorded_cells=summary['recorded_cells'], expected_cells=summary['expected_cells'],
             updated_at=time.time()))
        if pending or active:
            time.sleep(15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--temperature', type=float, default=.7)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error('workers must be between 1 and 4')
    run(args.origin.resolve(), args.output.resolve(), args.url, args.workers, args.temperature)


if __name__ == '__main__':
    main()
