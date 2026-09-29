"""Prepare immutable inputs, launch isolated workers, and summarize paired results."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from ttcl.experience_evolution.core import read, save, workspace_root, python_executable
from .worker import ARMS

WORKSPACE = workspace_root()
DEFAULT_ROOT = WORKSPACE/'results/deltamem_comparison/20260926'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def docker_status():
    try:
        process = subprocess.run(['docker', 'info', '--format', '{{.ServerVersion}}'],
                                 capture_output=True, text=True, timeout=20)
        docker = process.returncode == 0
        message = (process.stdout if docker else process.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        docker, message = False, repr(error)
    mini = importlib.util.find_spec('minisweagent') is not None
    return dict(available=docker and mini, docker_available=docker,
                minisweagent_available=mini, detail=message, checked_at=time.time())


def prepare(root, delta_python):
    if root.exists():
        raise FileExistsError('Choose a new directory; frozen inputs are never overwritten')
    old_alf = WORKSPACE/'ttcl/results/alfworld_comparison/20260924_parserfix'
    old_cl = WORKSPACE/'ttcl/results/reflexion_expel_clbench/20260923'
    alf, cl = read(old_alf/'plan.json'), read(old_cl/'plan.json')
    if not alf or not cl:
        raise FileNotFoundError('Historical plans are required for this paired replication; see EXPERIMENTS.md')
    audit = read(old_alf/'split_manifest.json')
    if not audit['checks_passed'] or audit['training_audit']['historical_training_evaluation_overlap']:
        raise ValueError('Invalid historical train/test audit')
    tasks = dict(cl['tasks'])
    for name, count in [('sales_prediction', 12), ('codebase_adaptation', 19)]:
        tasks[name] = dict(total=count, train=[], test=list(range(count)))
    model = WORKSPACE/'current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'
    adapter = WORKSPACE/'current_work/delta-Mem/model/delta-mem_qwen3_4b-instruct'
    for path in (model/'config.json', adapter/'delta_mem_adapter.pt', Path(delta_python)):
        if not path.is_file():
            raise FileNotFoundError(path)
    data = Path(alf['data_root'])
    for sequence in alf['sequences']:
        for task in sequence['tasks']:
            if '/valid_unseen/' not in task['path'] or sha(data/task['path']) != task['sha256']:
                raise ValueError('Changed ALFWorld task')
    settings = {k: alf[k] for k in ('data_root', 'max_steps', 'max_attempts', 'context',
                 'actor_temperature', 'actor_max_tokens', 'eval_seeds', 'sequences', 'test_tasks')}
    plan = dict(schema_version=1, created_at=time.time(), arms=list(ARMS), model=str(model),
        adapter=str(adapter), adapter_source='declare-lab/delta-mem_qwen3_4b-instruct',
        adapter_revision='c46dc31155608e412d44bf56638d5a6f856f2e7e',
        checkpoint_selection='official released Qasper TSW rank-8; fixed before evaluation',
        training=False, context=65536, delta_python=str(Path(delta_python).absolute()),
        benchmark_python=str(python_executable()),
        clbench=dict(tasks=tasks, repeats=cl['repeats'], max_attempts=3,
                     first_attempt_separate=True, extra_domains_schedule='default'),
        alfworld=settings, historical_alfworld=str(old_alf), historical_clbench=str(old_cl),
        expected_clbench_cells=sum(len(t['test']) for t in tasks.values())*len(cl['repeats'])*len(ARMS),
        expected_alfworld_cells=alf['test_tasks']*len(alf['eval_seeds'])*len(ARMS),
        policy=dict(reset_text_and_kv='every attempt', online_state='within domain/seed or ALF family/seed',
                    reset_ablation='reset at each new task; preserve state across its retries',
                    public_feedback_only=True, no_test_checkpoint_selection=True,
                    historical_results='descriptive cross-backend comparison; same-backend base rerun is primary',
                    scalar_feedback='not written into CL memory; only official public observations',
                    no_long_text_history=True, no_best_of_n=True))
    root.mkdir(parents=True)
    save(root/'plan.json', plan)
    save(root/'historical_split_audit.json', audit)
    save(root/'docker_preflight.json', docker_status())
    source = root/'source'
    (source/'ttcl').mkdir(parents=True)
    (source/'ttcl/__init__.py').write_text('')
    for name in ('deltamem_comparison', 'experience_evolution', 'experience_v2',
                 'experience_feedback', 'alfworld_comparison', 'structured_memory',
                 'llm_memory', 'common'):
        shutil.copytree(WORKSPACE/'ttcl'/name, source/'ttcl'/name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(WORKSPACE/'ttcl/paths.py', source/'ttcl/paths.py')
    shutil.copytree(WORKSPACE/'current_work/delta-Mem/deltamem', source/'deltamem',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    bench = WORKSPACE/'current_work/continual-learning-bench'
    shutil.copytree(bench/'src', source/'bench/src',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (source/'bench/data').symlink_to(bench/'data', target_is_directory=True)
    paths = [p for p in source.rglob('*') if p.is_file()]
    paths += [root/'plan.json', old_alf/'plan.json', old_cl/'plan.json', root/'historical_split_audit.json']
    paths += [p for p in model.iterdir() if p.is_file() and p.suffix in ('.json', '.safetensors', '.jinja')]
    paths += [adapter/'delta_mem_config.json', adapter/'delta_mem_adapter.pt']
    paths += [p for p in (bench/'data').rglob('*') if p.is_file()]
    save(root/'input_hashes.json', {str(p): sha(p) for p in paths})
    save(root/'prepared.json', {'prepared_at': time.time(), 'frozen_files': len(paths),
         'manifest_sha256': sha(root/'input_hashes.json')})
    return plan


def verify(root):
    manifest = read(root/'input_hashes.json')
    if sha(root/'input_hashes.json') != read(root/'prepared.json')['manifest_sha256']:
        raise ValueError('Input manifest changed')
    for path, expected in manifest.items():
        if sha(path) != expected:
            raise ValueError(f'Frozen input changed: {path}')


def environment(root):
    return dict(os.environ, TTCL_WORKSPACE=str(WORKSPACE),
        TTCL_BENCH=str(root/'source/bench'), ALFWORLD_DATA=str(WORKSPACE/'ttcl/data/alfworld_delta'),
        PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
        MSWEA_GLOBAL_CONFIG_DIR=str(root/'runtime/minisweagent'), MSWEA_SILENT_STARTUP='1',
        PYTHONPATH=os.pathsep.join(map(str, [root/'source', root/'source/bench',
                                         WORKSPACE/'ttcl/.runtime/deltamem_benchmark_deps',
                                         WORKSPACE/'ttcl/.runtime/structured_memory_deps'])))


def supervise(root, suite, gpu, port):
    import requests
    plan = read(root/'plan.json')
    verify(root)
    env = environment(root)
    server_env = dict(env, CUDA_VISIBLE_DEVICES=str(gpu))
    library = Path(plan['delta_python']).parent.parent/'lib'
    server_env['LD_LIBRARY_PATH'] = str(library)+os.pathsep+os.environ.get('LD_LIBRARY_PATH', '')
    server_command = [plan['delta_python'], '-m', 'ttcl.deltamem_comparison.server',
        '--model', plan['model'], '--adapter', plan['adapter'], '--output', str(root), '--port', str(port)]
    worker_command = [plan['benchmark_python'], '-m', 'ttcl.deltamem_comparison.worker',
        '--root', str(root), '--suite', suite, '--url', f'http://127.0.0.1:{port}']
    http = requests.Session(); http.trust_env = False
    server = None
    try:
        with (root/f'{suite}_server.log').open('a') as log:
            server = subprocess.Popen(server_command, env=server_env, cwd=root/'source',
                                      stdout=log, stderr=subprocess.STDOUT)
        save(root/f'{suite}_processes.json', dict(supervisor_pid=os.getpid(), server_pid=server.pid,
             gpu=gpu, port=port, server_command=server_command, worker_command=worker_command))
        for _ in range(300):
            if server.poll() is not None:
                raise RuntimeError('Inference server exited; see server log')
            try:
                if http.get(f'http://127.0.0.1:{port}/health', timeout=2).json().get('ready'):
                    break
            except (requests.RequestException, ValueError):
                pass
            time.sleep(2)
        else:
            raise TimeoutError('Model startup timeout')
        with (root/f'{suite}_worker.log').open('a') as log:
            worker = subprocess.Popen(worker_command, env=env, cwd=root/'source',
                                      stdout=log, stderr=subprocess.STDOUT)
            save(root/f'{suite}_worker_pid.json', {'pid': worker.pid})
            code = worker.wait()
        if code:
            raise RuntimeError(f'{suite} worker exited with {code}; failure records retained')
    except Exception as error:
        save(root/f'{suite}_supervisor_status.json', {'phase': 'failed', 'error': repr(error)})
        raise
    finally:
        if server and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill(); server.wait()
    save(root/f'{suite}_supervisor_status.json', {'phase': 'stopped', 'updated_at': time.time()})


def launch(root, suites, cl_gpu=0, alf_gpu=6):
    plan = read(root/'plan.json')
    for suite, gpu, port in [('clbench', cl_gpu, 18307), ('alfworld', alf_gpu, 18308)]:
        if suite not in suites:
            continue
        pid_path = root/f'{suite}_supervisor_pid.json'
        if pid_path.exists():
            try:
                os.kill(read(pid_path)['pid'], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(f'{suite} supervisor is still alive')
        command = [plan['benchmark_python'], '-m', 'ttcl.deltamem_comparison.run',
                   'supervise', '--root', str(root), '--suite', suite, '--gpu', str(gpu), '--port', str(port)]
        with (root/f'{suite}_supervisor.log').open('a') as log:
            p = subprocess.Popen(command, cwd=root/'source', env=environment(root),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(pid_path, {'pid': p.pid, 'command': command, 'launched_at': time.time()})
        print(json.dumps({'suite': suite, 'gpu': gpu, 'pid': p.pid}), flush=True)


def aggregate(rows, key, field):
    groups = {}
    for row in rows:
        groups.setdefault(row['arm'], {})[key(row)] = row
    common = set.intersection(*(set(groups.get(a, {})) for a in ARMS))
    scored = {k for k in common if all(groups[a][k].get(field) is not None for a in ARMS)}
    return dict(recorded={a: len(groups.get(a, {})) for a in ARMS}, paired_records=len(common),
                paired_scored=len(scored), paired_unscored=len(common)-len(scored),
                means={a: statistics.mean(groups[a][k][field] for k in scored) if scored else None for a in ARMS},
                actor_calls={a: sum(r.get('actor_calls', 0) for r in groups.get(a, {}).values()) for a in ARMS})


def historical_comparison(root, plan, alf, cl):
    def paired(new, old, key, new_field, old_field):
        left = {key(r): r for r in new if r['arm'] == 'deltamem_online'}
        right = {key(r): r for r in old}
        keys = [k for k in left.keys() & right.keys()
                if left[k].get(new_field) is not None and right[k].get(old_field) is not None]
        if any(left[k].get('instance_id') != right[k].get('instance_id') for k in keys):
            raise ValueError('Historical CL canonical identities differ')
        a = [left[k][new_field] for k in keys]
        b = [right[k][old_field] for k in keys]
        return dict(paired=len(keys), deltamem=statistics.mean(a) if a else None,
                    historical=statistics.mean(b) if b else None,
                    difference=statistics.mean(x-y for x, y in zip(a, b)) if a else None)

    result = {'interpretation': 'Descriptive only: historical vLLM versus current HF; primary control is current HF base. CL old Delta saw the earlier prefix and used one attempt; other historical budgets are labeled.',
              'alfworld': {}, 'clbench': {}}
    old_alf = [read(p) for p in (Path(plan['historical_alfworld'])/'evaluation').glob('*/*/*/*/result.json')]
    for arm in ('retry_none', 'untrained', 'delta', 'reflexion', 'expel'):
        rows = [r for r in old_alf if r['arm'] == arm]
        result['alfworld'][arm] = {field: paired(alf, rows, lambda r: (r['game'], r['repeat']), field, field)
                                   for field in ('first_success', 'success')}
    for domain in plan['clbench']['tasks']:
        current = [r for r in cl if r['task'] == domain]
        comparisons = {}
        directory = Path(plan['historical_clbench'])/'runs'/domain
        for arm in ('none', 'retry_none', 'reflexion', 'expel'):
            old = []
            for p in directory.glob(f'*/evaluation/*/{arm}/result.json'):
                row = read(p)
                row['repeat'] = int(p.parents[3].name)
                old.append(row)
            comparisons[arm] = paired(current, old, lambda r: (r['canonical_index'], r['repeat']),
                'first_reward' if arm in ('none', 'expel') else 'reward', 'reward')
        feedback = WORKSPACE/'ttcl/results/experience_feedback/20260923/feedback_transfer/clbench'/domain
        old_delta = [read(p) for p in feedback.glob('*/delta/*/row.json')]
        comparisons['old_delta_one_attempt_full_prefix'] = paired(current, old_delta,
            lambda r: (r['canonical_index'], r['repeat']), 'first_reward', 'reward')
        result['clbench'][domain] = comparisons
    save(root/'historical_comparison.json', result)
    return result


def status(root):
    plan = read(root/'plan.json')
    alf = [read(p) for p in (root/'alfworld').glob('*/*/*/*/result.json')]
    cl = [read(p) for p in (root/'clbench').glob('*/*/*/*/result.json')]
    result = dict(updated_at=time.time(), alfworld=dict(completed=len(alf), expected=plan['expected_alfworld_cells']),
                  clbench=dict(completed=len(cl), expected=plan['expected_clbench_cells'], tasks={}),
                  workers={p.stem: read(p) for p in (root/'workers').glob('*.json')})
    result['alfworld'].update(first_attempt=aggregate(alf, lambda r: (r['game'], r['repeat']), 'first_success'),
                             within_three=aggregate(alf, lambda r: (r['game'], r['repeat']), 'success'))
    for domain in plan['clbench']['tasks']:
        rows = [r for r in cl if r['task'] == domain]
        result['clbench']['tasks'][domain] = dict(
            first_attempt=aggregate(rows, lambda r: (r['canonical_index'], r['repeat']), 'first_reward'),
            within_three=aggregate(rows, lambda r: (r['canonical_index'], r['repeat']), 'reward'))
    result['historical_sources'] = {k: plan[k] for k in ('historical_alfworld', 'historical_clbench')}
    historical_comparison(root, plan, alf, cl)
    save(root/'summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'launch', 'supervise', 'status', 'verify'])
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--delta-python', default=os.environ.get('TTCL_DELTA_PYTHON'))
    parser.add_argument('--suite', choices=['clbench', 'alfworld'])
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--cl-gpu', type=int, default=0)
    parser.add_argument('--alf-gpu', type=int, default=6)
    parser.add_argument('--port', type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'prepare':
        if not args.delta_python:
            parser.error('Set --delta-python or TTCL_DELTA_PYTHON to the isolated Delta-Mem interpreter')
        plan = prepare(root, args.delta_python)
        print(json.dumps({k: plan[k] for k in ('expected_clbench_cells', 'expected_alfworld_cells')}))
    elif args.action == 'launch':
        launch(root, [args.suite] if args.suite else ['clbench', 'alfworld'], args.cl_gpu, args.alf_gpu)
    elif args.action == 'supervise':
        supervise(root, args.suite, args.gpu, args.port)
    elif args.action == 'verify':
        verify(root)
    else:
        status(root)


if __name__ == '__main__':
    main()
