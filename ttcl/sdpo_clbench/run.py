from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

from .protocol import ARMS,TASKS,UPSTREAM_COMMIT,binding,read,save,sha


def workspace():
    return Path(os.environ.get('TTCL_WORKSPACE',Path(__file__).resolve().parents[2])).resolve()


def defaults():
    w=workspace()
    model=w/'models/delta_mem/Qwen3-4B-Instruct-2507'
    if not (model/'config.json').exists():
        model=w/'current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'
    return dict(model=str(model.resolve()),device='cuda:0',
        task_seed=42,context=65536,actor_max_tokens=4096,actor_temperature=1.,actor_top_p=1.,
        lora_rank=8,lora_alpha=16,lora_targets=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        learning_rate=1e-5,teacher_update_rate=.05,actions_per_update=2,max_grad_norm=1.,
        training_context=24576,feedback_chars=12000,logit_chunk=64,
        loss='official SDPO top100 + tail, JSD alpha=0.5',optimizer_steps_per_episode=1)


def prepare(root,args):
    if root.exists():raise FileExistsError('Frozen runs must never be overwritten')
    w=workspace();bench=w/'current_work/continual-learning-bench';model=Path(defaults()['model'])
    for p in [model/'config.json',bench/'data/database_exploration/products.db',w/'current_work/SDPO_reference/LICENSE']:
        if not p.is_file():raise FileNotFoundError(p)
    root.mkdir(parents=True)
    plan=dict(defaults(),created_at=time.time(),tasks=TASKS,repeats=[303,404],arms=list(ARMS),
        expected_records=sum(TASKS.values())*4,gpu=args.gpu,python=str(Path(sys.executable).absolute()),
        checkpoint_root=str(w/'models/sdpo_clbench'/root.name),upstream_commit=UPSTREAM_COMMIT,
        upstream_url='https://github.com/lasgroup/SDPO',paper='https://arxiv.org/abs/2601.20802',
        protocol=dict(scope='CLBench online LoRA transfer, not original paper benchmark reproduction',
          scoring='Every task scored once before update, including first task; official scores unchanged',
          memory='Student/EMA teacher/Adam persist within a domain-seed chain; reset between chains',
          isolation='No historical Delta/PPO weights, no offline fitting on benchmark tasks, no future episode input',
          feedback='Public tool/format feedback and completed episode scalar only; enhanced scalar feedback protocol',
          reward_role='Condition self-teacher on scalar reward; no separate scalar policy-gradient term',
          teacher='EMA adapter, frozen during each step; unchanged official top100+tail JSD loss',
          collection='temperature=1 top_p=1; one attempt; up to two schema retries; all calls retained',
          selection='Two uniformly sampled generated actions per episode, independent of outcomes/length; exclusions logged',
          budget='No extra environment rollouts; one gradient step per episode except last; all compute logged',
          context='No inference truncation; selected training actions over 24576 tokens excluded with counts',
          missing='Unscored failures retained, never converted to zero; common scored pairs and failure rates reported',
          omitted_domains={'sales_prediction':'Docker-dependent domain outside current four-domain suite',
                           'codebase_adaptation':'Docker-dependent domain outside current four-domain suite'},
          deltas_from_upstream=['Qwen3-4B-Instruct-2507 + LoRA replaces full-parameter training',
             'Persistent cross-task adapter instead of resetting per coding question',
             'One rollout per task; no successful sibling rollouts',
             'Continuous rewards included as feedback, no binary success threshold',
             'Two actions per episode; single-device PyTorch replaces distributed veRL',
             'LoRA learning rate 1e-5; EMA 0.05; JSD 0.5; top100+tail retained']))
    if Path(plan['checkpoint_root']).exists():raise FileExistsError('Checkpoint destination already exists')
    save(root/'plan.json',plan)
    source=root/'source';(source/'ttcl').mkdir(parents=True);(source/'ttcl/__init__.py').write_text('')
    for name in ['sdpo_clbench','structured_memory','llm_memory','common']:
        shutil.copytree(w/'ttcl'/name,source/'ttcl'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(w/'current_work/SDPO_reference',source/'current_work/SDPO_reference')
    shutil.copytree(bench/'src',source/'bench/src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (source/'bench/data').symlink_to((bench/'data').resolve(),target_is_directory=True)
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    save(root/'packages.json',packages)
    paths=[p for p in source.rglob('*') if p.is_file()]
    paths += [p for p in (bench/'data').rglob('*') if p.is_file()]
    paths += [p for p in model.iterdir() if p.is_file() and p.suffix in ['.json','.safetensors','.jinja']]
    paths += [root/'plan.json',root/'packages.json']
    save(root/'input_hashes.json',{str(p):sha(p) for p in paths})
    save(root/'prepared.json',{'manifest_sha256':sha(root/'input_hashes.json'),'files':len(paths)})
    print(json.dumps({'root':str(root),'expected_records':plan['expected_records'],'gpu':args.gpu}))


def verify(root):
    if sha(root/'input_hashes.json')!=read(root/'prepared.json')['manifest_sha256']:raise ValueError('Manifest changed')
    for path,expected in read(root/'input_hashes.json').items():
        if sha(path)!=expected:raise ValueError('Frozen input changed: '+path)


def environment(root):
    w=workspace();plan=read(root/'plan.json')
    return dict(os.environ,TTCL_WORKSPACE=str(w),TTCL_BENCH=str(root/'source/bench'),
        PYTHONPATH=os.pathsep.join(map(str,[root/'source',root/'source/bench',w/'ttcl/.runtime/structured_memory_deps'])),
        CUDA_VISIBLE_DEVICES=str(plan['gpu']),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',
        PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')


def summarize(root):
    plan=read(root/'plan.json');rows=[read(p) for p in (root/'runs').glob('*/*/*/episode_*/row.json')]
    tables={}
    for task in TASKS:
        groups={a:{(r['repeat'],r['canonical_index']):r for r in rows if r['task']==task and r['arm']==a and r['status']=='complete'} for a in ARMS}
        common=set(groups[ARMS[0]]) & set(groups[ARMS[1]])
        for k in common:
            if len({(g[k]['instance_id'],g[k]['initial_query_sha256']) for g in groups.values()})!=1:raise ValueError('Pair mismatch')
        table={}
        for label,keys in [('all',common),('after_first',{k for k in common if k[1]>0})]:
            means={a:statistics.mean(groups[a][k]['reward'] for k in keys) if keys else None for a in ARMS}
            table[label]={'paired_n':len(keys),'means':means,
                'sdpo_minus_frozen':means['sdpo_online']-means['frozen'] if keys else None}
        table['by_seed']={str(s):{'paired_n':sum(k[0]==s for k in common),
            'delta':statistics.mean(groups['sdpo_online'][k]['reward']-groups['frozen'][k]['reward'] for k in common if k[0]==s) if any(k[0]==s for k in common) else None} for s in plan['repeats']}
        tables[task]=table
    updates=[read(p) for p in (root/'runs').glob('*/*/sdpo_online/episode_*/update/audit.json')]
    result={'updated_at':time.time(),'expected_records':plan['expected_records'],'recorded_records':len(rows),
        'scored_records':sum(r['status']=='complete' for r in rows),'failed_records':sum(r['status']!='complete' for r in rows),
        'parameter_updates':sum(u['status']=='updated' for u in updates),'skipped_updates':sum(u['status']!='updated' for u in updates),
        'domains':tables,'costs':{a:{k:sum(r.get(k,0) for r in rows if r['arm']==a) for k in ['actor_calls','actor_input_tokens','actor_output_tokens','actor_seconds']} for a in ARMS},
        'update_seconds':sum(u.get('seconds',0) for u in updates)}
    save(root/'summary.json',result);return result


def supervise(root):
    import fcntl
    lock=(root/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(root/'plan.json');verify(root);env=environment(root);(root/'logs').mkdir(exist_ok=True)
    completed=[];failed=[]
    try:
        for repeat in plan['repeats']:
            for task in plan['tasks']:
                cmd=[plan['python'],'-m','ttcl.sdpo_clbench.run','worker','--root',str(root),'--task',task,'--repeat',str(repeat)]
                with (root/'logs'/f'{task}_{repeat}.log').open('x') as log:
                    p=subprocess.Popen(cmd,env=env,cwd=str(root/'source'),stdout=log,stderr=subprocess.STDOUT)
                save(root/'status.json',{'phase':'running','task':task,'repeat':repeat,'pid':p.pid,'completed':completed,'failed':failed,'time':time.time()})
                while p.poll() is None:
                    summarize(root);time.sleep(10)
                entry={'task':task,'repeat':repeat,'returncode':p.returncode}
                (completed if p.returncode==0 else failed).append(entry);summarize(root)
        save(root/'status.json',{'phase':'complete' if not failed else 'finished_with_failures',
            'completed':completed,'failed':failed,'time':time.time()})
    except Exception as exc:
        save(root/'status.json',{'phase':'failed','error':repr(exc),'traceback':traceback.format_exc(),'time':time.time()});raise


def smoke(root):
    from .model import OnlineModel
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True);plan=defaults();plan.update(actor_max_tokens=32,context=2048,training_context=4096)
    save(root/'plan.json',plan);model=OnlineModel(plan,92727);model.arm='sdpo_online'
    messages=[{'role':'user','content':'Return a JSON object with the key answer set to the integer 2.'}]
    sample=model.generate(messages,42)
    event={'messages':messages,'raw_response':sample['raw_response'],'parse_error':None}
    before=model.fingerprint('base')
    update=model.update(model.captures,[event],[{'step':1,'public_feedback':'The required value of answer is 2. This is synthetic development feedback.'}],0.,[0],root/'update')
    after=model.fingerprint('base')
    if not update['adapter_changed'] or before!=after:raise AssertionError('Smoke invariants failed')
    save(root/'audit.json',{'passed':True,'real_qwen4b_update':True,'base_unchanged':True,'update':update,
        'data':'synthetic schema task; no CLBench test outcomes used'})
    print(json.dumps(read(root/'audit.json')))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','verify','supervise','launch','status','worker','smoke'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',type=int,default=2)
    p.add_argument('--task',choices=TASKS);p.add_argument('--repeat',type=int)
    args=p.parse_args();root=args.root.resolve()
    if args.command=='prepare':prepare(root,args)
    elif args.command=='verify':verify(root);print('Frozen inputs verified')
    elif args.command=='supervise':supervise(root)
    elif args.command=='status':print(json.dumps({'status':read(root/'status.json') if (root/'status.json').exists() else None,'summary':summarize(root)},indent=2))
    elif args.command=='worker':
        from .worker import work
        work(root,args.task,args.repeat)
    elif args.command=='smoke':smoke(root)
    elif args.command=='launch':
        if (root/'launched.json').exists():raise FileExistsError('Already launched; preserve run')
        plan=read(root/'plan.json')
        cmd=[plan['python'],'-m','ttcl.sdpo_clbench.run','supervise','--root',str(root)]
        with (root/'supervisor.log').open('x') as log:
            proc=subprocess.Popen(cmd,env=environment(root),cwd=str(root/'source'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(root/'launched.json',{'pid':proc.pid,'command':cmd,'time':time.time()});print('Supervisor PID',proc.pid)


if __name__=='__main__':main()
