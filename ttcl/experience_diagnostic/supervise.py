"""Run isolated GPU workers and keep an inspectable process/job ledger."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

PYTHON = '/home/fengboyu/miniconda3/envs/seal_env/bin/python'
WORKSPACE = Path('/home/fengboyu/cl')


def main(root):
    source = root / 'source'
    env = dict(os.environ, PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
               TTCL_BENCH=str(WORKSPACE / 'current_work/continual-learning-bench'),
               PYTHONPATH=os.pathsep.join([str(WORKSPACE / 'ttcl/.runtime/structured_memory_deps'),
                                          str(source), str(WORKSPACE / 'current_work/continual-learning-bench')]))
    def command(phase, *extra):
        return [PYTHON, '-m', 'ttcl.experience_diagnostic.run', phase, '--root', str(root), *extra]
    jobs = []
    def persist():
        path = root / 'jobs.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps({'pid': os.getpid(), 'updated': time.time(), 'jobs': jobs}, indent=2))
        tmp.replace(path)
    def phase(specifications, gpus):
        pending = list(specifications)
        active = {}
        while pending or active:
            for gpu in gpus:
                if gpu in active or not pending:
                    continue
                name, cmd = pending.pop(0)
                logfile = root / 'logs' / f'{name}.log'
                logfile.parent.mkdir(exist_ok=True)
                handle = logfile.open('a')
                process = subprocess.Popen(cmd, cwd=source, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)),
                                           stdout=handle, stderr=subprocess.STDOUT)
                handle.close()
                job = dict(name=name, command=cmd, gpu=gpu, pid=process.pid, status='running', started=time.time())
                jobs.append(job)
                active[gpu] = process, job
                print('started', name, process.pid, flush=True)
            for gpu, (process, job) in list(active.items()):
                code = process.poll()
                if code is not None:
                    job.update(status='complete' if code == 0 else 'failed', code=code, finished=time.time())
                    del active[gpu]
                    print(job['status'], job['name'], flush=True)
            persist()
            if pending or active:
                time.sleep(5)
        if any(j['status'] == 'failed' for j in jobs):
            raise RuntimeError('Worker failure retained in logs; dependent stages not started')
    phase([(f'generate_{task}', command('generate', '--task', task))
           for task in ['database_exploration', 'cohort_studies']], [0, 1])
    subprocess.run(command('freeze'), cwd=source, env=env, check=True)
    phase([(f'score_{task}_{repeat}', command('score', '--task', task, '--repeat', str(repeat)))
           for task, repeat in [('database_exploration', 505), ('cohort_studies', 505),
                                ('cohort_studies', 606), ('database_exploration', 606)]], [0, 1, 3])
    subprocess.run(command('report'), cwd=source, env=env, check=True)
    print('All diagnostic scoring and integrity checks finished.', flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
