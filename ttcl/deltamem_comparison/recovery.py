"""Audited CL-only infrastructure recovery; original frozen source stays intact."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from ttcl.experience_evolution.core import read, save, workspace_root
from .run import environment, sha, verify


def verify_recovery(root, recovery):
    verify(root)
    for path, expected in read(recovery/'hashes.json').items():
        if sha(path) != expected:
            raise ValueError('Recovery input changed: '+path)
    for path, expected in read(recovery/'completed_attempt_hashes.json').items():
        if sha(path) != expected:
            raise ValueError('Previously completed attempt changed: '+path)


def prepare(root, recovery, *, reason=None, wait_max_memory_mib=None):
    verify(root)
    if read(root/'workers/clbench.json')['phase'] != 'failed':
        raise ValueError('Only a failed CL worker may use this recovery')
    if recovery.exists():
        raise FileExistsError(recovery)
    recovery.mkdir(parents=True)
    shutil.copytree(root/'source/ttcl', recovery/'source/ttcl', symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    replacement = workspace_root()/'ttcl/deltamem_comparison'
    changes = {}
    for name in ['worker.py', 'server.py', 'recovery.py']:
        dest = recovery/'source/ttcl/deltamem_comparison'/name
        changes[name] = {'before': sha(dest) if dest.exists() else None,
                         'after': sha(replacement/name)}
        shutil.copy2(replacement/name, dest)
    completed = {}
    for marker in (root/'clbench').rglob('done.json'):
        for path in marker.parent.iterdir():
            if path.is_file():
                completed[str(path)] = sha(path)
    save(recovery/'completed_attempt_hashes.json', completed)
    save(recovery/'recovery.json', dict(root=str(root), reason=reason or 'Official poker can finish before any actor call; absent response log and empty state are valid only for no-public-input completed episodes.',
        wait_max_memory_mib=wait_max_memory_mib,
        changed_files=changes, original_frozen_source_unchanged=True,
        plan_changed=False, completed_attempts=len(list((root/'clbench').rglob('done.json'))),
        created_at=time.time()))
    for path in [root/'workers/clbench.json', root/'clbench_supervisor_status.json',
                 root/'clbench_processes.json', root/'clbench_supervisor_pid.json']:
        if path.exists():
            shutil.copy2(path, recovery/('previous_'+path.name))
    paths = [p for p in (recovery/'source').rglob('*') if p.is_file()]
    paths += [recovery/'recovery.json', recovery/'completed_attempt_hashes.json']
    save(recovery/'hashes.json', {str(p): sha(p) for p in paths})


def env_for(root, recovery):
    env = environment(root)
    env['PYTHONPATH'] = str(recovery/'source')+os.pathsep+env['PYTHONPATH']
    return env


def supervise(root, recovery, gpu, port):
    import requests
    verify_recovery(root, recovery)
    # Keep inference numerics and all evaluation budgets fixed; only defer startup.
    limit = read(recovery/'recovery.json').get('wait_max_memory_mib')
    if limit is not None:
        while True:
            used = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
            if used <= limit:
                break
            save(root/'clbench_supervisor_status.json', dict(phase='waiting_for_gpu',
                recovery=str(recovery), gpu=gpu, used_mib=used, maximum_used_mib=limit,
                updated_at=time.time()))
            time.sleep(30)
        verify_recovery(root, recovery)
    plan = read(root/'plan.json')
    env = env_for(root, recovery)
    server_env = dict(env, CUDA_VISIBLE_DEVICES=str(gpu),
        LD_LIBRARY_PATH=str(Path(plan['delta_python']).parent.parent/'lib')+os.pathsep+os.environ.get('LD_LIBRARY_PATH', ''))
    server_command = [plan['delta_python'], '-m', 'ttcl.deltamem_comparison.server',
        '--model', plan['model'], '--adapter', plan['adapter'], '--output', str(root), '--port', str(port)]
    worker_command = [plan['benchmark_python'], '-m', 'ttcl.deltamem_comparison.worker',
        '--root', str(root), '--suite', 'clbench', '--url', f'http://127.0.0.1:{port}']
    http = requests.Session(); http.trust_env = False
    server = None
    save(root/'clbench_supervisor_status.json', {'phase': 'recovering', 'recovery': str(recovery)})
    try:
        with (recovery/'server.log').open('a') as log:
            server = subprocess.Popen(server_command, cwd=recovery/'source', env=server_env,
                                      stdout=log, stderr=subprocess.STDOUT)
        save(root/'clbench_processes.json', dict(supervisor_pid=os.getpid(), server_pid=server.pid,
            gpu=gpu, port=port, server_command=server_command, worker_command=worker_command,
            recovery=str(recovery)))
        for _ in range(300):
            if server.poll() is not None:
                raise RuntimeError('Recovery inference server exited')
            try:
                if http.get(f'http://127.0.0.1:{port}/health', timeout=2).json().get('ready'):
                    break
            except (requests.RequestException, ValueError):
                pass
            time.sleep(2)
        else:
            raise TimeoutError('Recovery server startup timeout')
        # No task, no inference: verify zero-action reset state is serializable.
        response = http.post(f'http://127.0.0.1:{port}/begin', json={'arm': 'deltamem_reset'}, timeout=30)
        response.raise_for_status()
        response = http.post(f'http://127.0.0.1:{port}/finish', json={
            'destination': str(recovery/'empty_state_preflight.pt'), 'messages': []}, timeout=30)
        response.raise_for_status()
        check = response.json()
        if check['state_bytes'] != 0 or check['state_modules'] != 0:
            raise ValueError('Zero-action initial state must stay empty')
        save(recovery/'empty_state_preflight.json', check)
        with (recovery/'worker.log').open('a') as log:
            worker = subprocess.Popen(worker_command, cwd=recovery/'source', env=env,
                                      stdout=log, stderr=subprocess.STDOUT)
            save(root/'clbench_worker_pid.json', {'pid': worker.pid, 'recovery': str(recovery)})
            save(root/'clbench_supervisor_status.json', {'phase': 'running', 'recovery': str(recovery)})
            code = worker.wait()
        if code:
            raise RuntimeError(f'Recovery worker exited with {code}')
        save(root/'clbench_supervisor_status.json', {'phase': 'stopped', 'recovery': str(recovery)})
    except Exception as error:
        save(root/'clbench_supervisor_status.json', {'phase': 'failed', 'recovery': str(recovery), 'error': repr(error)})
        raise
    finally:
        if server and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill(); server.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'launch', 'supervise'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--recovery', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--port', type=int, default=18307)
    parser.add_argument('--reason', help='Recorded infrastructure recovery reason (prepare only)')
    parser.add_argument('--wait-max-memory-mib', type=int, help='Freeze a maximum GPU occupancy for startup (prepare only)')
    args = parser.parse_args()
    if args.wait_max_memory_mib is not None and args.wait_max_memory_mib < 0:
        parser.error('--wait-max-memory-mib must be nonnegative')
    root, recovery = args.root.resolve(), args.recovery.resolve()
    if args.action == 'prepare':
        prepare(root, recovery, reason=args.reason, wait_max_memory_mib=args.wait_max_memory_mib)
    elif args.action == 'supervise':
        supervise(root, recovery, args.gpu, args.port)
    else:
        prior = read(root/'clbench_supervisor_pid.json')
        if prior:
            try:
                os.kill(prior['pid'], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError('Previous CL supervisor is still alive')
        command = [read(root/'plan.json')['benchmark_python'], '-m', 'ttcl.deltamem_comparison.recovery',
            'supervise', '--root', str(root), '--recovery', str(recovery), '--gpu', str(args.gpu), '--port', str(args.port)]
        with (recovery/'supervisor.log').open('a') as log:
            process = subprocess.Popen(command, cwd=recovery/'source', env=env_for(root, recovery),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(root/'clbench_supervisor_pid.json', {'pid': process.pid, 'recovery': str(recovery)})
        print(json.dumps({'pid': process.pid, 'recovery': str(recovery)}))


if __name__ == '__main__':
    main()
