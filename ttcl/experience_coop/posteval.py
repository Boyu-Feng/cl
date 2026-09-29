"""Queue a fixed-final evaluation without changing a running frozen PPO experiment."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from ttcl.experience_evolution.core import read, save, workspace_root, python_executable


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_optional(path):
    return read(path) if Path(path).exists() else None


def next_action(status, training_finished, test_summary):
    if status.get('phase') == 'failed':
        return 'failed'
    if not training_finished:
        return 'failed' if status.get('phase') == 'complete' else 'wait'
    if status.get('phase') != 'complete':
        return 'wait'
    if test_summary and test_summary.get('complete'):
        return 'reuse'
    return 'run_full_test'


def final_adapters(origin, plan):
    last = plan['blocks'][-1]['id']
    for arm in plan['arms']:
        for block in plan['blocks']:
            for role in (['writer', 'reader'] if arm == 'dual' else ['writer']):
                status = read_optional(origin/'training'/arm/block['id']/role/'status.json')
                if not status or status.get('phase') != 'complete':
                    return None
    expected = {'original_delta': plan['initial_adapter'],
        'writer_only': str(origin/'training/writer_only'/last/'writer/adapter'),
        'dual_writer': str(origin/'training/dual'/last/'writer/adapter'),
        'dual_reader': str(origin/'training/dual'/last/'reader/adapter')}
    latest = read_optional(origin/'latest_checkpoints.json')
    if not latest:
        return None
    for arm, role, key in [('writer_only','writer','writer_only'),
                          ('dual','writer','dual_writer'),('dual','reader','dual_reader')]:
        if Path(latest[arm][role]).resolve() != Path(expected[key]).resolve():
            return None
    for path in expected.values():
        directory = Path(path)
        if not (directory/'adapter_config.json').is_file() or not list(directory.glob('*.safetensors')):
            raise ValueError('Incomplete final adapter: '+str(directory))
    return expected


def prepare(root, origin):
    if root.exists():
        raise FileExistsError('Post-evaluation queue already exists')
    plan = read(origin/'plan.json')
    if not plan:
        raise FileNotFoundError(origin/'plan.json')
    root.mkdir(parents=True)
    shutil.copy2(Path(__file__), root/'runner.py')
    (root/'source').symlink_to(origin/'source', target_is_directory=True)
    counts = {stage:sum(len(s['tasks']) for s in plan[key])*len(plan[seeds])*len(plan['evaluation_routes'])
              for stage,key,seeds in [('development','development','development_seeds'),
                                      ('test','final_sequences','final_seeds')]}
    request = dict(origin=str(origin), created_at=time.time(),
        origin_plan_sha256=sha(origin/'plan.json'), runner_sha256=sha(root/'runner.py'),
        user_request='PPO-8 training completion must automatically be followed by evaluation of the trained model.',
        policy='Keep original development evaluation. Reuse its full test if run; otherwise run the same full test once, regardless of development gate.',
        checkpoint_rule='Only the predeclared final block of both training branches; never select by evaluation reward.',
        final_block=plan['blocks'][-1]['id'], expected_records=counts,
        gpu=plan['gpu'], port=plan['port'], test_seeds=plan['final_seeds'],
        routes=plan['evaluation_routes'], test_sequences=plan['final_sequences'],
        interpretation='User-requested fixed-final diagnostic if development gate fails; not a claim of passing that gate.',
        poll_seconds=30, no_training_updates=True, no_test_tuning=True)
    save(root/'request.json', request)
    save(root/'queue_integrity.json', {'request_sha256':sha(root/'request.json')})
    save(root/'status.json', {'phase':'queued', 'expected_test_records':counts['test']})
    return counts


def verify_queue(root, request):
    if sha(root/'request.json') != read(root/'queue_integrity.json')['request_sha256']:
        raise ValueError('Queued evaluation request changed')
    if sha(root/'runner.py') != request['runner_sha256']:
        raise ValueError('Queued runner changed')
    if sha(Path(request['origin'])/'plan.json') != request['origin_plan_sha256']:
        raise ValueError('Original training plan changed')


def acquire_gpu(root, plan):
    lock = Path(f'/tmp/ttcl_experience_coop_gpu_{plan["gpu"]}.lock').open('a')
    acquired = False
    while True:
        if not acquired:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                pass
        result = subprocess.run(['nvidia-smi',f'--id={plan["gpu"]}',
            '--query-gpu=memory.used','--format=csv,noheader,nounits'],
            capture_output=True,text=True,check=True)
        used = int(result.stdout.strip())
        if acquired and used < 1000:
            return lock
        save(root/'status.json',{'phase':'waiting_for_gpu','gpu':plan['gpu'],
            'used_mib':used,'original_gpu_lock_available':acquired,'updated_at':time.time()})
        time.sleep(30)


def run_full_test(root, origin, request, adapters):
    # Imported from the unchanged original frozen source via launch PYTHONPATH.
    from ttcl.experience_coop.run import verify, checkpoint_hashes, open_server, evaluate
    from ttcl.experience_v2.common import stop_server
    verify(origin)
    plan = copy.deepcopy(read(origin/'plan.json'))
    if plan['final_sequences'] != request['test_sequences'] or plan['evaluation_routes'] != request['routes']:
        raise ValueError('Predeclared evaluation grid differs')
    decision = read(origin/'development_decision.json')
    plan.update(post_training_origin=str(origin), post_training_request=str(root/'request.json'),
        final_rule='User-requested diagnostic evaluation of fixed final checkpoints; development gate outcome is retained.')
    save(root/'plan.json', plan)
    save(root/'development_decision.json', dict(decision or {},
        original_decision_source=str(origin/'development_decision.json'),
        full_test_requested_independently=True))
    hashes = dict(read(origin/'input_hashes.json'))
    hashes.update(checkpoint_hashes(adapters.values()))
    hashes.update({str(root/'plan.json'):sha(root/'plan.json'),
                   str(root/'request.json'):sha(root/'request.json'),
                   str(root/'runner.py'):sha(root/'runner.py')})
    save(root/'input_hashes.json', hashes)
    save(root/'final_checkpoint_bindings.json', {'adapters':adapters,
        'adapter_hashes':checkpoint_hashes(adapters.values()), 'selected_by':'fixed final block'})
    verify(root)
    gpu_lock = acquire_gpu(root, plan)
    server = None
    try:
        save(root/'status.json',{'phase':'starting_test_server','gpu':plan['gpu'],'time':time.time()})
        server = open_server(root, 'user_requested_final', adapters)
        save(root/'status.json',{'phase':'test','expected':request['expected_records']['test'],'time':time.time()})
        summary = evaluate(root, 'test', adapters)
        verify(root)
        save(root/'final_result.json',{'summary':summary,
            'qualified_on_development':(decision or {}).get('qualified', []),
            'test_trigger':'user requested full fixed-final evaluation, independent of the development gate',
            'no_further_tuning_from_this_test':True})
        save(root/'status.json',{'phase':'complete','result':str(root/'final_result.json'),'time':time.time()})
    finally:
        stop_server(server)
        gpu_lock.close()


def supervise(root):
    request = read(root/'request.json'); origin = Path(request['origin'])
    lock = (root/'queue.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_queue(root, request)
    try:
        while True:
            verify_queue(root, request)
            plan = read(origin/'plan.json')
            status = read(origin/'status.json') or {}
            adapters = final_adapters(origin, plan)
            summary_path = origin/'evaluation/test/summary.json'
            summary = read_optional(summary_path)
            action = next_action(status, adapters is not None, summary)
            if action == 'failed':
                raise RuntimeError('Original run failed or reports completion without all final checkpoints; no incomplete model will be evaluated')
            if action == 'reuse':
                if summary.get('completed') != request['expected_records']['test']:
                    raise ValueError('Original test has a different record count')
                save(root/'final_result.json',{'reused_from':str(summary_path),
                    'summary_sha256':sha(summary_path),'summary':summary,'duplicate_test_started':False})
                save(root/'status.json',{'phase':'complete','reused_from':str(summary_path),'time':time.time()})
                return
            if action == 'run_full_test':
                run_full_test(root, origin, request, adapters)
                return
            save(root/'status.json',{'phase':'waiting_for_original_run','original_phase':status.get('phase'),
                'original_arm':status.get('arm'),'original_block':status.get('block'),
                'final_checkpoints_ready':adapters is not None,
                'development_is_automatic':True,'full_test_is_queued':True,'updated_at':time.time()})
            time.sleep(request['poll_seconds'])
    except BaseException as error:
        save(root/'status.json',{'phase':'failed','error':repr(error),
            'traceback':traceback.format_exc(),'time':time.time()})
        raise


def launch(root):
    request = read(root/'request.json'); verify_queue(root, request)
    if read(root/'status.json')['phase'] != 'queued':
        raise ValueError('Launch requires a fresh prepared queue')
    from ttcl.experience_v2.common import environment
    env = environment(root)
    command = [str(python_executable()), str(root/'runner.py'), 'supervise','--root',str(root)]
    with (root/'supervisor.log').open('a') as log:
        process = subprocess.Popen(command,cwd=root/'source',env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(root/'supervisor_pid.json',{'pid':process.pid,'command':command,'started_at':time.time()})
    return process.pid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','launch','supervise','status'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--origin',type=Path)
    args = parser.parse_args();root=args.root.resolve()
    if args.action == 'prepare':
        if args.origin is None:parser.error('--origin is required')
        print(prepare(root,args.origin.resolve()))
    elif args.action == 'launch':print(launch(root))
    elif args.action == 'supervise':supervise(root)
    else:print(json.dumps(read(root/'status.json'),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
