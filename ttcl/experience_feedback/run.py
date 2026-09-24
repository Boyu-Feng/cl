from __future__ import annotations
import argparse
from collections import deque
import fcntl
import os
from pathlib import Path
import signal
import time
import traceback
from ttcl.experience_v2.common import read, save, sha_file, start_server, stop_server
from ttcl.experience_v2.launch import process
from .evaluate import clbench
from .report import report, audit_chain


def verify(root):
    for path, expected in read(root/'input_hashes.json').items():
        if sha_file(path)!=expected: raise ValueError(f'Frozen input changed: {path}')


def run_suite(root,suite,concurrency=4):
    settings=read(root/'plan.json')['suites'][suite]
    queue=deque((task,repeat) for task in settings['tasks'] for repeat in settings['repeats'])
    active=[]; finished=[]; failed=[]; last_report=0
    try:
        while queue or active:
            while queue and len(active)<concurrency:
                task,repeat=queue.popleft(); name=f'{task}_{repeat}'
                status=root/suite/'workers'/f'{name}.json'
                if status.exists() and read(status).get('phase')=='complete':
                    audit_chain(root,suite,task,repeat); finished.append(name);continue
                p=process(root,'ttcl.experience_feedback.run',
                    ['worker','--root',root,'--suite',suite,'--task',task,'--repeat',repeat],f'{suite}/logs/{name}.log')
                active.append((name,p))
            for name,p in list(active):
                code=p.poll()
                if code is None: continue
                (failed if code else finished).append(name); active.remove((name,p))
            save(root/f'{suite}_status.json',{'phase':'running','active':[{'name':n,'pid':p.pid} for n,p in active],
                 'finished':finished,'failed':failed,'queued':len(queue),'updated_at':time.time()})
            if time.monotonic()-last_report>30: report(root,suite);last_report=time.monotonic()
            if failed: raise RuntimeError(f'Worker failed: {failed}; inspect logs')
            if queue or active: time.sleep(3)
        report(root,suite)
        save(root/f'{suite}_status.json',{'phase':'complete','finished':finished,'failed':[],'active':[],'queued':0})
    finally:
        for _,p in active:
            if p.poll() is None: p.terminate()
        for _,p in active:
            try: p.wait(timeout=10)
            except Exception: p.kill();p.wait()


def supervise(root):
    lock=(root/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify(root)
    server=None
    def stopping(signum, frame): raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,stopping)
    try:
        save(root/'status.json',{'phase':'starting_server','supervisor_pid':os.getpid(),'stage':1})
        settings=read(root/'plan.json')['suites']['feedback_transfer']
        adapters={name:root/'adapters'/name for name in settings['arms'].values() if name!='frozen-actor'}
        server=start_server(root,0,18237,adapters)
        for suite in ['smoke','feedback_transfer']:
            save(root/'status.json',{'phase':'running','stage':1,'suite':suite,'supervisor_pid':os.getpid()})
            run_suite(root,suite)
        save(root/'status.json',{'phase':'complete','stage':1,'next_stage':'matched_training_prepared_not_started'})
    except BaseException as exc:
        save(root/'status.json',{'phase':'stopped' if isinstance(exc,KeyboardInterrupt) else 'failed',
                                'stage':1,'traceback':traceback.format_exc()})
        raise
    finally: stop_server(server)


def training_pipeline(root):
    parent=root.parent if root.name=='failure_mix' else root
    verify(parent)
    if read(parent/'status.json').get('phase')!='complete':
        raise RuntimeError('Stage 1 feedback evaluation must finish before training')
    if root!=parent and read(parent/'training_pipeline_status.json').get('phase')!='complete':
        raise RuntimeError('Stage 2 must finish before failure-mixture ablation')
    lock=(parent/'training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(root/'training_plan.json');server=None;worker=None
    try:
        server=start_server(root,1,18238,{},context=32768)
        for arm in plan['training_arms']:
            save(root/'training_pipeline_status.json',{'phase':'training','arm':arm})
            worker=process(root,'ttcl.experience_feedback.run',['train','--root',root,'--arm',arm],f'logs/train_{arm}.log',3)
            if worker.wait(): raise RuntimeError(f'Training failed: {arm}')
        save(root/'training_pipeline_status.json',{'phase':'complete','arms':list(plan['training_arms'])})
    except BaseException:
        save(root/'training_pipeline_status.json',{'phase':'failed','traceback':traceback.format_exc()});raise
    finally:
        if worker and worker.poll() is None:
            worker.terminate()
            try: worker.wait(timeout=10)
            except Exception: worker.kill();worker.wait()
        stop_server(server)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['supervise','worker','report','training','train'])
    p.add_argument('--root',required=True,type=Path);p.add_argument('--suite',default='feedback_transfer')
    p.add_argument('--task');p.add_argument('--repeat',type=int);p.add_argument('--arm')
    a=p.parse_args()
    if a.command=='supervise': supervise(a.root)
    elif a.command=='worker':
        clbench(a.root,a.suite,a.task,a.repeat);audit_chain(a.root,a.suite,a.task,a.repeat)
    elif a.command=='report': report(a.root,a.suite)
    elif a.command=='training': training_pipeline(a.root)
    else:
        from .train import train
        train(a.root,a.arm)

if __name__=='__main__':main()
