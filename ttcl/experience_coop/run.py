"""Single-GPU queued supervisor; older runs, sources and adapters are never edited."""
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
from ttcl.experience_lab.evaluate import promote
from .prepare import DEFAULT_ROOT, DEFAULT_SPLITS, prepare
from .collect import source_shard, freeze_sources, collect_history, freeze_block
from .evaluate import evaluate_shard, summarize


def verify(root):
    root=Path(root)
    manifests=[root/'input_hashes.json']
    if (root/'reader_initial_hashes.json').exists(): manifests.append(root/'reader_initial_hashes.json')
    for manifest in manifests:
        for path,h in read(manifest).items():
            if sha_file(path)!=h: raise ValueError('Frozen experiment input changed: '+path)


def child(root, command, args, label, gpu=None):
    env=environment(root)
    if gpu is not None: env['CUDA_VISIBLE_DEVICES']=str(gpu)
    else: env['CUDA_VISIBLE_DEVICES']=''
    cmd=[str(PYTHON),'-m','ttcl.experience_coop.run',command,'--root',str(root),*map(str,args)]
    path=root/'logs'/f'{label}.log';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as log:
        process=subprocess.Popen(cmd,cwd=root/'source',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(root/'processes'/f'{label}.json',{'pid':process.pid,'command':cmd,'created':time.time()})
    return process


def workers(root, command, jobs, label):
    pending=deque(jobs);active=[];completed=[]
    try:
        while pending or active:
            while pending and len(active)<4:
                name,args=pending.popleft();active.append((name,child(root,command,args,f'{label}_{name}')))
            for name,process in active.copy():
                code=process.poll()
                if code is None: continue
                active.remove((name,process))
                if code: raise RuntimeError(f'{label}/{name} failed; inspect its log')
                completed.append(name)
            save(root/'worker_status.json',{'stage':label,'completed':completed,
                'active':[{'name':n,'pid':p.pid} for n,p in active],'queued':len(pending),'time':time.time()})
            if active or pending: time.sleep(3)
    finally:
        for _,process in active: stop_server(process)


def wait_for_gpu(root, plan):
    idle=0
    while idle<3:
        predecessor=plan.get('predecessor'); predecessor_phase=None
        if predecessor:
            p=Path(predecessor)/'status.json'
            predecessor_phase=read(p).get('phase') if p.exists() else 'missing_status'
        eligible=not predecessor or predecessor_phase in ['complete','failed']
        result=subprocess.run(['nvidia-smi',f'--id={plan["gpu"]}',
            '--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        used=int(result.stdout.strip())
        idle=idle+1 if eligible and used<1000 else 0
        save(root/'status.json',{'phase':'waiting_for_gpu','gpu':plan['gpu'],'used_mib':used,
            'predecessor':predecessor,'predecessor_phase':predecessor_phase,
            'supervisor_pid':os.getpid(),'time':time.time()})
        if idle<3: time.sleep(15)


def open_server(root, label, adapters):
    plan=read(root/'plan.json');directory=root/'services'/label
    directory.mkdir(parents=True,exist_ok=False)
    (directory/'source').symlink_to(root/'source',target_is_directory=True)
    return start_server(directory,plan['gpu'],plan['port'],adapters,context=plan['context'])


def checkpoint_hashes(paths):
    result={}
    for path in paths:
        if path is None: continue
        path=Path(path)
        if path.is_file(): result[str(path)]=sha_file(path)
        else:
            for p in path.rglob('*'):
                if p.is_file(): result[str(p)]=sha_file(p)
    return result


def evaluate(root, stage, adapters):
    plan=read(root/'plan.json');directory=root/'evaluation'/stage
    settings={'arms':list(plan['evaluation_routes']),'routes':plan['evaluation_routes'],
        'seeds':plan['development_seeds'] if stage=='development' else plan['final_seeds'],
        'sequences':plan['development'] if stage=='development' else plan['final_sequences'],
        'adapters':{k:str(v) for k,v in adapters.items()},'weights_frozen':True}
    save(directory/'evaluation_plan.json',settings)
    hashes=checkpoint_hashes(adapters.values());save(directory/'adapter_hashes.json',hashes)
    jobs=[(s['id'],['--stage',stage,'--shard',s['id']]) for s in settings['sequences']]
    workers(root,'evaluate-shard',jobs,f'eval_{stage}')
    summary=summarize(directory)
    if not summary['complete']: raise RuntimeError('Incomplete evaluation grid')
    for p,h in hashes.items():
        if sha_file(p)!=h: raise ValueError('Evaluation checkpoint changed')
    return summary


def server_preflight(root, plan):
    from .client import Client
    client=Client(plan)
    try:
        messages=[{'role':'user','content':'Return the single word READY.'}]
        samples={}
        for name in ['frozen-actor','original_delta','reader_initial']:
            samples[name]=client.complete(messages,name,92710,12,1.,1.,capture=True)
        save(root/'preflight/server_protocol.json',{'passed':True,'samples':samples,
            'task_data_used':False,'exact_sampled_token_ids':True})
    finally: client.session.close()


def supervise(root):
    root=Path(root);plan=read(root/'plan.json')
    lock=(root/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu_lock=Path(f'/tmp/ttcl_experience_coop_gpu_{plan["gpu"]}.lock').open('a')
    fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify(root)
    if not (root/'reader_initial_hashes.json').exists(): raise ValueError('Initialize reader before launch')
    server=None;process=None
    try:
        wait_for_gpu(root,plan)
        save(root/'status.json',{'phase':'server_preflight','gpu':plan['gpu'],'time':time.time()})
        server=open_server(root,'source_collection',{'original_delta':plan['initial_adapter'],
                                                     'reader_initial':plan['initial_reader']})
        server_preflight(root,plan)
        save(root/'status.json',{'phase':'collecting_shared_sources','expected':len(plan['histories']),
                                'supervisor_pid':os.getpid(),'time':time.time()})
        shards=sorted({h.get('family',h['domain']) for h in plan['histories']})
        workers(root,'source-shard',[(s,['--shard',s]) for s in shards],'sources')
        freeze_sources(root);stop_server(server);server=None
        states={arm:{'writer':plan['initial_adapter'],'reader':plan['initial_reader'],
                     'writer_optimizer':None,'reader_optimizer':None} for arm in plan['arms']}
        # Interleave conditions at the same block, each retaining its own optimizer.
        for block in plan['blocks']:
            bid=block['id']
            for arm in plan['arms']:
                directory=root/'training'/arm/bid;directory.mkdir(parents=True,exist_ok=False)
                state=states[arm]
                save(directory/'block_input.json',dict(state,arm=arm,block=bid,
                    hashes=checkpoint_hashes([*state.values(),root/'source_audit.json']),
                    routing='Writer and reader are separate alternatives on a frozen shared base'))
                adapters={'writer_current':state['writer']}
                if arm=='dual': adapters['reader_current']=state['reader']
                server=open_server(root,f'{arm}_{bid}',adapters)
                save(root/'status.json',{'phase':'ppo_rollout','arm':arm,'block':bid,
                    'blocks_per_arm':len(plan['blocks']),'rollout_per_history':8,'time':time.time()})
                jobs=[(hid,['--arm',arm,'--block',bid,'--history',hid]) for hid in block['histories']]
                workers(root,'collect-history',jobs,f'{arm}_{bid}_rollout')
                counts=freeze_block(root,arm,bid)
                stop_server(server);server=None
                for role in (['writer','reader'] if arm=='dual' else ['writer']):
                    save(root/'status.json',{'phase':'ppo_update','arm':arm,'block':bid,'role':role,
                                            'examples':counts[role],'time':time.time()})
                    process=child(root,'train',['--arm',arm,'--block',bid,'--role',role],
                                  f'{arm}_{bid}_{role}_train',plan['gpu'])
                    if process.wait(): raise RuntimeError(f'{arm}/{bid}/{role} training failed')
                    process=None
                    state[role]=str(directory/role/'adapter')
                    state[role+'_optimizer']=str(directory/role/'optimizer.pt')
                save(root/'latest_checkpoints.json',states)
            verify(root)
        adapters={'original_delta':plan['initial_adapter'],'writer_only':states['writer_only']['writer'],
                  'dual_writer':states['dual']['writer'],'dual_reader':states['dual']['reader']}
        server=open_server(root,'development',adapters)
        save(root/'status.json',{'phase':'development','time':time.time()})
        development=evaluate(root,'development',adapters)
        qualified=[a for a in ['ppo8_writer','dual'] if promote(development,[a]) is not None]
        save(root/'development_decision.json',{'qualified':qualified,
            'rule':'ALF gain with nonnegative seeds; all four CL domains nonnegative and at least one improves',
            'fixed_final_checkpoints':True,'not_a_significance_claim':True})
        stop_server(server);server=None
        if qualified:
            server=open_server(root,'final_comparison',adapters)
            save(root/'status.json',{'phase':'final_comparison','qualified':qualified,'time':time.time()})
            final=evaluate(root,'test',adapters)
            save(root/'final_result.json',{'qualified_on_development':qualified,'summary':final,
                 'no_further_tuning_from_this_test':True})
        verify(root)
        save(root/'status.json',{'phase':'complete','qualified':qualified,
            'outcome':'final_comparison_finished' if qualified else 'no_dual_benchmark_development_gain',
            'time':time.time()})
    except BaseException:
        save(root/'status.json',{'phase':'failed','time':time.time(),'traceback':traceback.format_exc()})
        raise
    finally:
        stop_server(process);stop_server(server)


def launch(root):
    root=Path(root);verify(root)
    if read(root/'status.json')['phase']!='prepared': raise ValueError('Launch requires a fresh experiment')
    if not (root/'reader_initial_hashes.json').exists(): raise ValueError('Initialize reader first')
    process=child(root,'supervise',[],'supervisor')
    save(root/'supervisor_pid.json',{'pid':process.pid})
    return {'pid':process.pid,'root':str(root)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['prepare','init-reader','launch','supervise','source-shard',
        'collect-history','train','evaluate-shard','status','verify'])
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--split-plan',type=Path,default=DEFAULT_SPLITS)
    p.add_argument('--gpu',type=int,default=3);p.add_argument('--port',type=int,default=18287)
    p.add_argument('--predecessor',type=Path)
    p.add_argument('--arm',choices=['writer_only','dual']);p.add_argument('--role',choices=['writer','reader'])
    p.add_argument('--block');p.add_argument('--history');p.add_argument('--shard')
    p.add_argument('--stage',choices=['development','test'])
    a=p.parse_args();root=a.root.resolve()
    if a.command=='prepare': print(prepare(root,a.split_plan,a.gpu,a.port,a.predecessor))
    elif a.command=='init-reader':
        from .learning import initialize_reader
        verify(root);print(initialize_reader(root))
    elif a.command=='launch': print(launch(root))
    elif a.command=='source-shard': source_shard(root,a.shard)
    elif a.command=='collect-history': collect_history(root,a.arm,a.block,a.history)
    elif a.command=='train':
        from .learning import train
        train(root,a.arm,a.block,a.role)
    elif a.command=='evaluate-shard': evaluate_shard(root,a.stage,a.shard)
    elif a.command=='verify': verify(root)
    elif a.command=='status': print(read(root/'status.json'))
    else:
        def stop(signum,frame): raise KeyboardInterrupt(f'Signal {signum}')
        signal.signal(signal.SIGTERM,stop);supervise(root)


if __name__=='__main__': main()
