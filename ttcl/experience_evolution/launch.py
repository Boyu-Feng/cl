"""Run fixed training arms and controls; fail on any worker failure."""
import os
import subprocess
import time
from .core import save, workspace_root, python_executable


WORKSPACE = workspace_root()
ROOT = WORKSPACE / 'ttcl/results/experience_evolution/alfworld_delta_20260922'
PYTHON = str(python_executable(WORKSPACE))


def start(command, arm, gpu):
    log = (ROOT / f'{command}_{arm}.log').open('w')
    env = dict(os.environ, TTCL_WORKSPACE=str(WORKSPACE), CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    p = subprocess.Popen([PYTHON, '-m', 'ttcl.experience_evolution.run', command,
                          '--root', str(ROOT), '--arm', arm], env=env, stdout=log, stderr=subprocess.STDOUT)
    log.close()
    print(f'START {command}/{arm} pid={p.pid} gpu={gpu}', flush=True)
    return p


if __name__ == '__main__':
    assert (ROOT / 'freeze.json').exists()
    queues = {0: [('train', 'delta'), ('evaluate', 'delta'), ('evaluate', 'delta_reset')],
              3: [('train', 'absolute'), ('evaluate', 'absolute')],
              2: [('evaluate', 'none'), ('evaluate', 'untrained')]}
    active, finished = {}, []
    try:
        while queues or active:
            for gpu in list(queues):
                if gpu not in active:
                    if not queues[gpu]:
                        del queues[gpu]
                        continue
                    command, arm = queues[gpu].pop(0)
                    active[gpu] = (start(command, arm, gpu), command, arm)
            for gpu, (p, command, arm) in list(active.items()):
                code = p.poll()
                if code is None:
                    continue
                if code:
                    raise RuntimeError(f'{command}/{arm} exited {code}; see log')
                finished.append(f'{command}/{arm}')
                print('FINISHED', finished[-1], flush=True)
                del active[gpu]
            save(ROOT / 'status.json', {'phase': 'running', 'finished': finished,
                 'workers': {str(g): {'pid': v[0].pid, 'command': v[1], 'arm': v[2]} for g,v in active.items()}})
            time.sleep(3)
        subprocess.run([PYTHON, '-m', 'ttcl.experience_evolution.analyze', '--root', str(ROOT)], check=True)
        save(ROOT / 'status.json', {'phase': 'complete', 'finished': finished})
    except BaseException as e:
        for p, _, _ in active.values():
            if p.poll() is None:
                p.terminate()
        save(ROOT / 'status.json', {'phase': 'failed', 'finished': finished, 'error': str(e)})
        raise
