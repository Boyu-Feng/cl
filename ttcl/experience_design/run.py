from __future__ import annotations

import argparse
from collections import defaultdict
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import threading
import time
import traceback

from ttcl.experience_evolution.core import read,save,append,seed,writer_messages
from ttcl.experience_v2.common import Client,PYTHON,environment,sha_file,start_server,stop_server
from ttcl.experience_repair.run import matched,validate_episode,episode_metrics,generate
from .data import ROOT,prepare
from .environments import run_local
from .learning import reward_label,make_writer,update,train_sft,adapter_hash


def verify(root):
    bad=[p for p,h in read(root/'input_hashes.json').items() if sha_file(p)!=h]
    if bad:raise ValueError(f'Frozen input changes: {bad}')
    path=root/'training_data/hashes.json'
    if path.exists():
        for p,h in read(path).items():
            if sha_file(p)!=h:raise ValueError('SFT input changed')


class MixedActor:
    def __init__(self,plan):
        from ttcl.experience_evolution.environment import Actor
        self.plan=plan;self.alf=Actor(plan)
        self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=16);self.local=threading.local()

    def custom(self,job):
        if not hasattr(self.local,'client'):
            self.local.client=Client(self.plan['actor_url'],context=self.plan['context'])
        ep=run_local(job['spec'],job['memory'],job['seed'],self.local.client)
        save(Path(job['output'])/'episode.json',ep)
        return ep

    def run_many(self,jobs):
        missing=[]
        for job in jobs:
            dest=Path(job['output'])
            if (dest/'episode.json').exists():validate_episode(read(dest/'episode.json'),job)
            else:
                if dest.exists():dest.rename(dest.with_name(dest.name+f'.interrupted_{time.time_ns()}'))
                missing.append(job)
        # Bound concurrency; every persisted job has a deterministic identity.
        for start in range(0,len(missing),16):
            batch=missing[start:start+16]
            alf=[j for j in batch if j['spec']['domain']=='alfworld']
            custom=[self.pool.submit(self.custom,j) for j in batch if j['spec']['domain']!='alfworld']
            if alf:self.alf.run_many(alf)
            for f in custom:f.result()
        results=[read(Path(j['output'])/'episode.json') for j in jobs]
        for ep,j in zip(results,jobs):validate_episode(ep,j)
        return results

    def close(self):
        self.pool.shutdown(wait=False,cancel_futures=True);self.alf.pool.shutdown(wait=False,cancel_futures=True)


def actor_job(spec,memory,repeat,path):
    return {'game':spec['id'],'spec':spec,'memory':memory,'seed':repeat,'output':str(path)}


def collect(root):
    plan=read(root/'plan.json');client=Client(plan['actor_url'],context=plan['context']);values={a:[] for a in plan['sft_arms']}
    for index,row in enumerate(read(root/'dataset.json')):
        text=generate(root,client,row['messages'],'delta',root/'raw_generations'/row['id'],seed(92431,row['id'],'raw'))
        for arm,target in [('raw_sft',text),('balanced_sft',row['target'])]:
            values[arm].append({'id':row['id'],'messages':row['messages'],'target':target,
                                'domain':row['domain'],'category':row['category']})
        save(root/'collection_status.json',{'phase':'running','completed':index+1,'expected':128})
    paths=[]
    for arm,rows in values.items():
        p=root/'training_data'/f'{arm}.json';save(p,rows);paths.append(p)
    save(root/'training_data/hashes.json',{str(p):sha_file(p) for p in paths})
    save(root/'collection_status.json',{'phase':'complete','completed':128,'expected':128})


def train_rl(root,arm):
    plan=read(root/'plan.json');objective=plan['rl_arms'][arm];out=root/'training'/arm
    if (out/'status.json').exists() and read(out/'status.json')['phase']=='complete':return
    if out.exists():raise RuntimeError('Partial RL needs explicit checkpoint recovery')
    out.mkdir(parents=True);save(out/'status.json',{'phase':'loading','objective':objective})
    writer=make_writer(plan,root/'training/balanced_sft/adapter');actor=MixedActor(plan)
    histories=read(root/'dataset.json');random.Random(92431).shuffle(histories)
    totals=[];scored=0
    try:
        for bi,start in enumerate(range(0,len(histories),4)):
            batch=histories[start:start+4];directory=out/f'batch_{bi:03}'
            save(out/'status.json',{'phase':'generating','batch':bi,'expected_batches':32,'actor_episodes':scored})
            messages=[h['messages'] for h in batch for _ in range(2)]
            paths=[directory/h['id']/f'candidate_{k}' for h in batch for k in range(2)]
            samples=writer.generate(messages,seed(92431,'rl_writer',bi),paths)
            metrics=[]
            for i,history in enumerate(batch):
                candidates=samples[i*2:i*2+2]
                memories={'candidate_0':candidates[0]['text'],'candidate_1':candidates[1]['text'],
                          'previous':history['previous'],'empty':''}
                jobs=[];keys=[]
                for ti,spec in enumerate(history['probes']):
                    for repeat in plan['probe_repeats']:
                        for branch,memory in memories.items():
                            jobs.append(actor_job(spec,memory,seed(repeat,history['id'],ti),
                                        directory/history['id']/'probes'/str(ti)/str(repeat)/branch))
                            keys.append((ti,repeat,branch))
                episodes=actor.run_many(jobs);scored+=len(episodes)
                for j in range(0,len(episodes),4):matched(episodes[j:j+4])
                by_branch=defaultdict(list)
                for ep,(ti,repeat,branch) in zip(episodes,keys):
                    by_branch[branch].append(ep['reward'])
                    metrics.append({'history':history['id'],'domain':history['domain'],'category':history['category'],
                                    'target':ti,'repeat':repeat,'branch':branch,**episode_metrics(ep)})
                for k,sample in enumerate(candidates):
                    advantage=reward_label(by_branch[f'candidate_{k}'],by_branch['previous'],by_branch['empty'],objective)
                    sample.update(advantage=advantage,delta=reward_label(by_branch[f'candidate_{k}'],by_branch['previous'],by_branch['empty'],'empty'))
                    save(paths[i*2+k]/'label.json',{'objective':objective,'advantage':advantage,
                         'candidate_rewards':by_branch[f'candidate_{k}'],'previous_rewards':by_branch['previous'],
                         'empty_rewards':by_branch['empty'],'vs_previous':reward_label(by_branch[f'candidate_{k}'],by_branch['previous'],by_branch['empty'],'previous'),
                         'vs_empty':sample['delta'],'history':history['id'],'domain':history['domain'],'category':history['category']})
                save(out/'status.json',{'phase':'probing','batch':bi,'expected_batches':32,'actor_episodes':scored,
                                       'expected_actor_episodes':2048,'updated_at':time.time()})
            save(directory/'probe_rows.json',metrics)
            value=update(writer,samples);value.update(batch=bi,objective=objective,actor_episodes=scored,updated_at=time.time())
            totals.append(value);append(out/'training.jsonl',value);save(out/'status.json',dict(phase='training',**value));print(json.dumps(value),flush=True)
            if (bi+1)%8==0:writer.save(out/f'checkpoint_{bi+1:03}')
        audit=writer.save(out)
        save(out/'reference_audit.json',{'reference_unchanged':adapter_hash(writer.model,'reference')==writer.reference_hash,
                                      'reference_sha256':writer.reference_hash,'reference_checkpoint':str(root/'training/balanced_sft/adapter')})
        save(out/'status.json',{'phase':'complete','objective':objective,'writer_actions':256,'optimizer_steps':64,
                               'actor_episodes':scored,'positive':sum(x['positive'] for x in totals),
                               'negative':sum(x['negative'] for x in totals),'zero':sum(x['zero'] for x in totals),'audit':audit})
    finally:actor.close()


def evaluation(root):
    plan=read(root/'plan.json');actor=MixedActor(plan);client=Client(plan['actor_url'],context=plan['context']);rows=[]
    try:
        for sequence in plan['evaluation_alf']:
            for repeat in plan['eval_repeats']:
                dest=root/'evaluation/alfworld'/sequence['id']/str(repeat)
                spec={'domain':'alfworld','id':sequence['games'][0]}
                source=actor.run_many([actor_job(spec,'',seed(repeat,sequence['id'],0),dest/'source')])[0]
                memories={'none':''}
                for arm in plan['evaluation_arms'][1:]:
                    memories[arm]=generate(root,client,writer_messages('',source),arm,dest/arm/'writer',seed(repeat,sequence['id'],'writer'))
                spec={'domain':'alfworld','id':sequence['games'][1]}
                results=actor.run_many([actor_job(spec,memories[arm],seed(repeat,sequence['id'],1),dest/arm/'target') for arm in plan['evaluation_arms']])
                matched(results)
                for arm,ep in zip(plan['evaluation_arms'],results):
                    rows.append({'domain':'alfworld','sequence':sequence['id'],'repeat':repeat,'arm':arm,'position':1,
                                 'phase':'transfer','family':sequence['family'],**episode_metrics(ep)})
                save(root/'evaluation_rows.json',rows);report(root)
                save(root/'evaluation_status.json',{'phase':'running','completed':len(rows),'expected':1288})
        for chain in plan['evaluation_chains']:
            for repeat in plan['eval_repeats']:
                memories={a:'' for a in plan['evaluation_arms']}
                for position,spec in enumerate(chain['tasks']):
                    dest=root/'evaluation'/chain['id']/str(repeat)/f'task_{position:02}'
                    episodes=actor.run_many([actor_job(spec,memories[arm],seed(repeat,chain['id'],position),dest/arm) for arm in plan['evaluation_arms']])
                    matched(episodes)
                    for arm,ep in zip(plan['evaluation_arms'],episodes):
                        rows.append({'domain':chain['domain'],'sequence':chain['id'],'repeat':repeat,'arm':arm,
                                     'position':position,'phase':chain['phases'][position],**episode_metrics(ep)})
                        if arm!='none':
                            memories[arm]=generate(root,client,writer_messages(memories[arm],ep),arm,dest/arm/'writer',seed(repeat,chain['id'],position,'writer'))
                    save(root/'evaluation_rows.json',rows);report(root)
                    save(root/'evaluation_status.json',{'phase':'running','completed':len(rows),'expected':1288})
        save(root/'evaluation_status.json',{'phase':'complete','completed':len(rows),'expected':1288})
    finally:actor.close()


def report(root):
    lines=['# Data and reward experiment','',
           'Independent mechanism tests; synthetic SQL/API scores are not CLBench results. See PROTOCOL.md.','']
    summary={}
    if (root/'evaluation_rows.json').exists():
        rows=read(root/'evaluation_rows.json')
        for domain in sorted({r['domain'] for r in rows}):
            selected=[r for r in rows if r['domain']==domain and r['position']>0]
            groups=defaultdict(dict)
            for r in selected:groups[r['arm']][(r['sequence'],r['repeat'],r['position'])]=r
            if not groups:continue
            common=set.intersection(*(set(g) for g in groups.values()))
            lines += [f'## {domain}: {len(common)} shared post-first records per arm','',
                      '| arm | success | delta none | A-return success | mean steps |','|---|---:|---:|---:|---:|']
            summary[domain]={}
            for arm,g in groups.items():
                values=[g[k] for k in common];returned=[r for r in values if r['phase']=='A_return']
                v={'n':len(values),'mean_reward':statistics.mean(r['reward'] for r in values),
                   'vs_none':statistics.mean(g[k]['reward']-groups['none'][k]['reward'] for k in common),
                   'return_reward':statistics.mean(r['reward'] for r in returned) if returned else None,
                   'mean_steps':statistics.mean(r['steps'] for r in values),
                   'by_seed':{str(s):statistics.mean(r['reward'] for r in values if r['repeat']==s) for s in {r['repeat'] for r in values}},
                   'input_tokens':sum(r['input_tokens'] for r in values),'output_tokens':sum(r['output_tokens'] for r in values),
                   'invalid_commands':sum(r['invalid_commands'] for r in values)}
                if domain=='alfworld':
                    v['by_family']={f:statistics.mean(r['reward'] for r in values if r['family']==f) for f in {r['family'] for r in values}}
                    v['macro_family']=statistics.mean(v['by_family'].values())
                summary[domain][arm]=v
                ret='—' if v['return_reward'] is None else f'{v["return_reward"]:.4f}'
                lines.append(f'| {arm} | {v["mean_reward"]:.4f} | {v["vs_none"]:+.4f} | {ret} | {v["mean_steps"]:.2f} |')
            lines.append('')
    if all(domain in summary for domain in ['alfworld','sql','tools']):
        shared=set.intersection(*(set(summary[d]) for d in ['alfworld','sql','tools']))
        summary['macro_domain']={a:statistics.mean(summary[d][a]['mean_reward'] for d in ['alfworld','sql','tools']) for a in shared}
    save(root/'summary.json',summary);(root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def worker(root,command,arm=None):
    env=environment(root);env['CUDA_VISIBLE_DEVICES']=str(read(root/'plan.json')['training_gpu'])
    args=[str(PYTHON),'-m','ttcl.experience_design.run',command,'--root',str(root)]
    if arm:args+=['--arm',arm]
    path=root/'logs'/f'{command}_{arm or "main"}.log';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as log:p=subprocess.Popen(args,env=env,cwd=root/'source',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(root/'processes'/f'{command}_{arm or "main"}.json',{'pid':p.pid,'command':args})
    return p


def supervise(root):
    lock=(root/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify(root);plan=read(root/'plan.json');server=None;child=None
    try:
        save(root/'status.json',{'phase':'starting_server','supervisor_pid':os.getpid(),'updated_at':time.time()})
        server=start_server(root,plan['server_gpu'],plan['port'],{'delta':root/'adapters/delta'},context=plan['context'])
        save(root/'status.json',{'phase':'collecting_raw_sft_control','supervisor_pid':os.getpid(),'updated_at':time.time()})
        collect(root)
        for arm in plan['sft_arms']:
            child=worker(root,'sft',arm)
            save(root/'status.json',{'phase':'sft','arm':arm,'supervisor_pid':os.getpid(),'worker_pid':child.pid,'updated_at':time.time()})
            if child.wait():raise RuntimeError(f'SFT failed: {arm}')
        for arm in plan['rl_arms']:
            child=worker(root,'rl',arm)
            save(root/'status.json',{'phase':'rl','arm':arm,'supervisor_pid':os.getpid(),'worker_pid':child.pid,'updated_at':time.time()})
            if child.wait():raise RuntimeError(f'RL failed: {arm}')
        stop_server(server);server=None
        adapters={'delta':root/'adapters/delta',**{a:root/'training'/a/'adapter' for a in [*plan['sft_arms'],*plan['rl_arms']]}}
        server=start_server(root,plan['server_gpu'],plan['port'],adapters,context=plan['context'])
        save(root/'status.json',{'phase':'evaluation','supervisor_pid':os.getpid(),'updated_at':time.time()})
        evaluation(root);verify(root);report(root)
        save(root/'status.json',{'phase':'complete','finished_at':time.time(),'expected':plan['expected']})
    except BaseException as exc:
        save(root/'status.json',{'phase':'stopped' if isinstance(exc,KeyboardInterrupt) else 'failed','updated_at':time.time(),'traceback':traceback.format_exc()});raise
    finally:
        if child is not None and child.poll() is None:stop_server(child)
        stop_server(server)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','supervise','sft','rl','report'])
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--arm');a=p.parse_args()
    def stop(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    try:
        if a.command=='prepare':prepare(a.root)
        elif a.command=='supervise':supervise(a.root)
        elif a.command=='sft':verify(a.root);train_sft(a.root,a.arm)
        elif a.command=='rl':verify(a.root);train_rl(a.root,a.arm)
        else:report(a.root)
    except BaseException as exc:
        if a.command in {'sft','rl'}:
            save(a.root/'training'/a.arm/'status.json',{'phase':'stopped' if isinstance(exc,KeyboardInterrupt) else 'failed','traceback':traceback.format_exc()})
        raise


if __name__=='__main__':main()
