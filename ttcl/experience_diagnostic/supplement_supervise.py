"""Run the evidence-format sensitivity check as original workers release GPUs."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main(parent):
    root = parent / 'supplement_raw_actions'
    workspace = Path('/home/fengboyu/cl')
    python = '/home/fengboyu/miniconda3/envs/seal_env/bin/python'
    env = dict(os.environ, PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
               TTCL_BENCH=str(workspace / 'current_work/continual-learning-bench'),
               PYTHONPATH=os.pathsep.join([str(workspace / 'ttcl/.runtime/structured_memory_deps'), str(parent / 'source'), str(workspace / 'current_work/continual-learning-bench')]))
    dependencies = {
        0: 'subset_cohort_studies_505.json', 2: 'subset_cohort_studies_606.json',
        6: 'prefix_subset_cohort_studies_606.json', 1: 'cohort_studies_505.json',
        3: 'cohort_studies_606.json',
    }
    pending = [(t, e, s) for t in ['cohort_studies', 'database_exploration'] for e in [13, 17] for s in [505, 606]]
    jobs, active = [], {}
    while pending or active:
        for gpu, dependency in dependencies.items():
            path = parent / 'progress' / dependency
            if gpu in active or not pending or not path.exists() or json.loads(path.read_text())['phase'] != 'complete':
                continue
            task, episode, repeat = pending.pop(0)
            name = f'{task}_{episode}_{repeat}'
            cmd = [python, '-m', 'ttcl.experience_diagnostic.raw_actions', 'score', str(root), task, str(repeat), str(episode)]
            logfile = root / 'logs' / f'{name}.log'
            logfile.parent.mkdir(exist_ok=True)
            with logfile.open('w') as handle:
                process = subprocess.Popen(cmd, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)), cwd=parent / 'source', stdout=handle, stderr=subprocess.STDOUT)
            job = dict(name=name, pid=process.pid, gpu=gpu, command=cmd, status='running', started=time.time())
            jobs.append(job)
            active[gpu] = process, job
            print('started', name, gpu, flush=True)
        for gpu, (process, job) in list(active.items()):
            code = process.poll()
            if code is not None:
                job.update(status='complete' if code == 0 else 'failed', exit_code=code, finished=time.time())
                del active[gpu]
                print(job['status'], job['name'], flush=True)
        tmp = root / 'jobs.tmp'
        tmp.write_text(json.dumps({'jobs': jobs, 'pending': pending, 'updated': time.time()}, indent=2))
        tmp.replace(root / 'jobs.json')
        if pending or active:
            time.sleep(5)
    if any(j['status'] == 'failed' for j in jobs):
        raise RuntimeError('Supplement worker failed; original results retained')
    print('All 16 supplementary cells attempted.', flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
