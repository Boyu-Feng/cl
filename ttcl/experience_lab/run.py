"""Bounded autonomous train/develop cycles, with immutable sources and no overwrites."""
from __future__ import annotations

import argparse
from collections import deque
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from ttcl.experience_evolution.core import read, save
from ttcl.experience_v2.common import PYTHON, environment, sha_file, start_server, stop_server
from .prepare import DEFAULT_ROOT, prepare
from .collect import collect_shard, build_dataset, shard_for
from .evaluate import evaluate_shard, summarize, promote


def verify(root):
    for path, expected in read(Path(root)/'input_hashes.json').items():
        if sha_file(path)!=expected: raise ValueError(f'Frozen exploration input changed: {path}')


def child(root, command, arguments, log_name, gpu=None):
    env=environment(root)
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    args=[str(PYTHON),'-m','ttcl.experience_lab.run',command,'--root',str(root),*map(str,arguments)]
    log=root/'logs'/f'{log_name}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as handle:
        process=subprocess.Popen(args,cwd=root/'source',env=env,stdout=handle,
                                 stderr=subprocess.STDOUT,start_new_session=True)
    save(root/'processes'/f'{log_name}.json',{'pid':process.pid,'command':args})
    return process


def run_workers(root, command, jobs, stage, parallel=4):
    pending=deque(jobs);active=[];completed=[]
    try:
        while pending or active:
            while pending and len(active)<parallel:
                name,args=pending.popleft()
                active.append((name,child(root,command,args,f'{stage}_{name}')))
            for name,process in active.copy():
                code=process.poll()
                if code is None:continue
                active.remove((name,process))
                if code:raise RuntimeError(f'{stage}/{name} failed; inspect its log')
                completed.append(name)
            save(root/'worker_status.json',{'stage':stage,'completed':completed,
                'active':[{'name':n,'pid':p.pid} for n,p in active], 'queued':len(pending),'time':time.time()})
            if pending or active:time.sleep(3)
    finally:
        for _,process in active:stop_server(process)


def wait_gpu(root, gpu):
    while True:
        result=subprocess.run(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used',
                               '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        if int(result.stdout.strip())<1000:return
        save(root/'status.json',{'phase':'waiting_for_idle_gpu','gpu':gpu,'time':time.time()})
        time.sleep(15)


def evaluate(root, run_id, stage, adapters):
    plan=read(root/'plan.json');directory=root/run_id/stage
    settings={'stage':stage,'arms':['none','untrained','original_delta',*adapters],
              'seeds':plan['development_seeds'] if stage=='development' else plan['final_seeds'],
              'sequences':plan['development'] if stage=='development' else plan['final_sequences'],
              'candidate_adapters':{k:str(v) for k,v in adapters.items()},
              'original_adapter':plan['initial_adapter']}
    if (directory/'evaluation_plan.json').exists():raise FileExistsError('Evaluation already exists')
    save(directory/'evaluation_plan.json',settings)
    save(directory/'adapter_hashes.json',{str(p):sha_file(p) for adapter in
         [Path(plan['initial_adapter']),*map(Path,adapters.values())] for p in adapter.rglob('*') if p.is_file()})
    jobs=[(s['id'],['--round',run_id,'--stage',stage,'--shard',s['id']]) for s in settings['sequences']]
    run_workers(root,'evaluate-shard',jobs,f'{run_id}_{stage}')
    result=summarize(directory)
    if not result['complete']:raise RuntimeError('Incomplete evaluation grid')
    for p,h in read(directory/'adapter_hashes.json').items():
        if sha_file(p)!=h:raise ValueError('Evaluation writer checkpoint changed')
    return result


def open_server(root, label, adapters):
    plan=read(root/'plan.json')
    directory=root/'services'/label
    directory.mkdir(parents=True,exist_ok=False)
    (directory/'source').symlink_to(root/'source',target_is_directory=True)
    return start_server(directory,plan['gpu'],plan['port'],adapters,context=plan['context'])


def supervise(root):
    root=Path(root);plan=read(root/'plan.json')
    lock=(root/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify(root);server=None;worker=None
    current=Path(plan['initial_adapter']);champion=None;decisions=[]
    try:
        wait_gpu(root,plan['gpu'])
        for rp in plan['rounds']:
            rid=rp['id'];directory=root/rid;directory.mkdir(exist_ok=False)
            save(directory/'round_input.json',{'parent_adapter':str(current),
                'parent_hashes':{str(p):sha_file(p) for p in current.rglob('*') if p.is_file()},
                'round':rid,'time':time.time()})
            save(root/'status.json',{'phase':'collecting','round':rid,'gpu':plan['gpu'],
                'supervisor_pid':os.getpid(),'expected_histories':len(rp['histories']),'time':time.time()})
            server=open_server(root,f'{rid}_collection',{'parent':current})
            # One independently seeded process per task family/domain; environment
            # locks are local and no global CLBench RNG is shared between threads.
            preferred=['pick_and_place_simple','exploitable_poker','look_at_obj_in_light',
                       'database_exploration','blind_spectrum_monitoring','cohort_studies',
                       'pick_clean_then_place_in_recep','pick_cool_then_place_in_recep',
                       'pick_heat_then_place_in_recep','pick_two_obj_and_place']
            available={shard_for(h) for h in rp['histories']}
            shards=[s for s in preferred if s in available]
            jobs=[(s,['--round',rid,'--shard',s]) for s in shards]
            run_workers(root,'collect-shard',jobs,f'{rid}_collect')
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(plan['model'],local_files_only=True)
            rows=build_dataset(root,rid,tokenizer)
            stop_server(server);server=None
            if len(rows)<plan['min_pairs']:
                decisions.append({'round':rid,'accepted_pairs':len(rows),'promoted':None,
                                  'reason':'insufficient_confirmed_training_signal'})
                save(root/'decisions.json',decisions)
                continue
            for arm in plan['training_arms']:
                save(root/'status.json',{'phase':'training','round':rid,'arm':arm,
                    'examples':len(rows),'supervisor_pid':os.getpid(),'time':time.time()})
                worker=child(root,'train',['--round',rid,'--arm',arm],f'{rid}_train_{arm}',plan['gpu'])
                if worker.wait():raise RuntimeError(f'Training failed: {rid}/{arm}')
                worker=None
            adapters={a:directory/'training'/a/'adapter' for a in plan['training_arms']}
            if champion is not None:adapters['incumbent']=current
            server=open_server(root,f'{rid}_development',
                {'original_delta':Path(plan['initial_adapter']),**adapters})
            save(root/'status.json',{'phase':'development','round':rid,'time':time.time()})
            summary=evaluate(root,rid,'development',adapters)
            winner=promote(summary,list(adapters))
            decisions.append({'round':rid,'accepted_pairs':len(rows),'promoted':winner,
                'reason':'development_gate_passed' if winner else 'no_dual_benchmark_development_gain'})
            save(root/'decisions.json',decisions)
            if winner:
                current=adapters[winner];champion={'round':rid,'arm':winner,'adapter':str(current)}
            stop_server(server);server=None
            verify(root)
        if champion:
            save(root/'selected.json',dict(champion,selection_only_uses_development=True))
            server=open_server(root,'final_confirmation',
                {'original_delta':Path(plan['initial_adapter']),'selected':current})
            save(root/'status.json',{'phase':'final_confirmation','selected':champion,'time':time.time()})
            summary=evaluate(root,'confirmation','test',{'selected':current})
            verified=promote(summary,['selected']) is not None
            save(root/'final_result.json',{'selected':champion,'dual_gain_point_estimate':verified,
                'summary':summary,'further_tuning_on_this_test_forbidden':True,
                'interpretation':'Point estimates are not statistical significance; retain per-task paired uncertainty.'})
        verify(root)
        save(root/'status.json',{'phase':'complete','selected':champion,
             'outcome':'confirmation_finished' if champion else 'no_candidate_passed_development',
             'finished_at':time.time()})
    except BaseException:
        save(root/'status.json',{'phase':'failed','time':time.time(),'traceback':traceback.format_exc()})
        raise
    finally:
        stop_server(worker);stop_server(server)


def launch(root):
    root=Path(root);verify(root)
    if read(root/'status.json')['phase']!='prepared':raise ValueError('Launch requires a fresh prepared cycle')
    process=child(root,'supervise',[],'supervisor')
    save(root/'supervisor_pid.json',{'pid':process.pid})
    return {'pid':process.pid,'root':str(root)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['prepare','launch','supervise','collect-shard','train','evaluate-shard','status','verify'])
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--gpu',type=int,default=1);p.add_argument('--port',type=int,default=18277)
    p.add_argument('--rounds',type=int,default=2);p.add_argument('--round');p.add_argument('--shard')
    p.add_argument('--stage',choices=['development','test']);p.add_argument('--arm')
    a=p.parse_args();root=a.root.resolve()
    if a.command=='prepare':print(prepare(root,a.gpu,a.port,a.rounds))
    elif a.command=='launch':print(launch(root))
    elif a.command=='collect-shard':collect_shard(root,a.round,a.shard)
    elif a.command=='train':
        from .learning import train
        train(root,a.round,a.arm)
    elif a.command=='evaluate-shard':evaluate_shard(root,a.round,a.stage,a.shard)
    elif a.command=='verify':verify(root)
    elif a.command=='status':print(read(root/'status.json'))
    else:
        def stopping(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
        signal.signal(signal.SIGTERM,stopping)
        supervise(root)


if __name__=='__main__':main()
