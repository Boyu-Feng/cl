"""Run and report the frozen four-domain universal-evidence comparison."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .report_universal_evidence import summarize


def jobs(plan):
    return [(task, repeat) for repeat in plan['repeats'] for task in plan['tasks']]


def run(origin: Path, output: Path, url: str, workers: int, temperature: float):
    plan = read(origin / 'plan.json')
    output.mkdir(parents=True, exist_ok=True)
    design = dict(origin=str(origin), origin_plan_sha256=sha(origin/'plan.json'),
                  runner_sha256=sha(Path(__file__)),
                  evaluator_sha256=sha(Path(__file__).with_name('evaluate_universal_evidence.py')),
                  method_sha256=sha(Path(__file__).with_name('universal_evidence.py')),
                  url=url, workers=workers, temperature=temperature,
                  jobs=[list(x) for x in jobs(plan)], policies=['vanilla', 'universal_evidence'])
    if (output/'design.json').exists() and read(output/'design.json') != design:
        raise ValueError('Frozen full-run design differs')
    save(output/'design.json', design)
    workspace = Path(__file__).resolve().parents[2]
    env = dict(os.environ, TTCL_BENCH=str(origin/'source'/'bench'),
               PYTHONPATH=os.pathsep.join((str(workspace), str(origin/'source'/'bench'),
                   str(workspace/'ttcl/.runtime/structured_memory_deps'),
                   str(workspace/'ttcl/.runtime/deltamem_benchmark_deps'))),
               HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2',
               PYTHONUNBUFFERED='1')
    pending, active, done = jobs(plan), [], []
    (output/'logs').mkdir(exist_ok=True)
    while pending or active:
        while pending and len(active) < workers:
            task, repeat = pending.pop(0)
            command = [sys.executable, '-m', 'ttcl.memrl_comparison.evaluate_universal_evidence',
                       '--origin', str(origin), '--output', str(output/f'{task}_{repeat}'),
                       '--benchmark', 'clbench', '--task', task, '--repeat', str(repeat),
                       '--url', url, '--temperature', str(temperature)]
            with (output/'logs'/f'{task}_{repeat}.log').open('a') as log:
                process = subprocess.Popen(command, env=env, cwd=workspace,
                                           stdout=log, stderr=subprocess.STDOUT)
            active.append((process, task, repeat))
        still = []
        for process, task, repeat in active:
            if process.poll() is None:
                still.append((process, task, repeat))
            else:
                done.append(dict(task=task, repeat=repeat, exit_code=process.returncode))
        active = still
        summary = summarize(origin, output)
        save(output/'status.json', dict(
            phase='running' if pending or active else
                  ('complete' if all(x['exit_code']==0 for x in done) else 'finished_with_failures'),
            pending=pending, active=[dict(pid=p.pid, task=t, repeat=r) for p,t,r in active],
            done=done, summary_complete=summary['complete'], updated_at=time.time()))
        if pending or active:
            time.sleep(15)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--origin', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--url', required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--temperature', type=float, default=.7)
    args = p.parse_args()
    if not 1 <= args.workers <= 2:
        p.error('workers must be 1 or 2')
    run(args.origin.resolve(), args.output.resolve(), args.url, args.workers, args.temperature)


if __name__ == '__main__':
    main()
