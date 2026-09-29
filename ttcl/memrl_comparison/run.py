from __future__ import annotations
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import sys
import time
import traceback
from ttcl.icl_mem0_comparison.protocol import TASKS, read, save, sha


def workspace():
    return Path(os.environ.get('TTCL_WORKSPACE',Path(__file__).resolve().parents[2])).resolve()


def environment(root=None):
    w=workspace();source=Path(root)/'source' if root else w
    bench=source/'bench' if root else w/'current_work/continual-learning-bench'
    return dict(os.environ,TTCL_WORKSPACE=str(w),TTCL_BENCH=str(bench),
        PYTHONPATH=os.pathsep.join(map(str,[source,bench,w/'ttcl/.runtime/structured_memory_deps',w/'ttcl/.runtime/deltamem_benchmark_deps',w/'models/nlp'])),
        HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',PYTHONUNBUFFERED='1')


def calibrate(output):
    import numpy as np
    from ttcl.memrl_comparison.memory import Embedder
    from ttcl.memrl_comparison.worker import alf_query
    from ttcl.alfworld_comparison.environment import make_env, concurrent_preflight
    from ttcl.icl_mem0_comparison.worker import make_task,base
    from ttcl.icl_mem0_comparison.protocol import normalize_prompt
    if output.exists():raise FileExistsError(output)
    w=workspace();old=read(w/'ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json')
    embedder=Embedder(w/'models/embedding/bge-m3')
    games=list(dict.fromkeys(g for group in old['training'] for g in group['games']))
    evidence={};queries=[]
    cachepath=output.with_suffix('.prompts.partial')
    cached=read(cachepath) if cachepath.exists() else {}
    for position,game in enumerate(games):
        if '/train/' not in game:raise ValueError('Calibration requires training games')
        path=Path(old['data_root'])/game;binding=sha(path)
        if game in cached:
            if cached[game]['game_sha256']!=binding:raise ValueError('Calibration input changed')
            queries.append(cached[game]['public_query'])
        else:
            env=make_env(path)
            try:queries.append(alf_query(str(env.reset()['feedback'])))
            finally:env.close()
            cached[game]=dict(game_sha256=binding,public_query=queries[-1])
            save(cachepath,cached)
        evidence[str(path)]=binding
        if (position+1)%24==0:print('Read ALF train prompts',position+1,'/',len(games),flush=True)
    def stats(texts):
        embeddings=np.asarray(embedder.embed(texts))
        sims=embeddings@embeddings.T
        values=np.clip(sims[np.triu_indices(len(texts),k=1)],-1.,1.)
        if len(values)<1:raise ValueError('Insufficient calibration prompts')
        return dict(mean=float(values.mean()),std=max(float(values.std()),1.e-6),
            threshold=min(.999999,float(np.quantile(values,.8))),prompt_count=len(texts),pairs=len(values),
            public_prompts=texts,rule='mean/std and 80th percentile of distinct calibration prompt pairs; no rewards')
    result={'alfworld':stats(queries)}
    print('ALFWorld calibration complete',len(queries),flush=True)
    os.chdir(os.environ['TTCL_BENCH'])
    for name,count in TASKS.items():
        prompts=[]
        for index in range(int(count*.2)):
            task=make_task(name,42)
            try:
                query=task.reset_baseline_instance(index);brief=task.get_agent_brief()
                brief=base.format_task_agent_brief(brief) if brief else ''
                prompts.append(normalize_prompt(query.prompt,brief,index,count))
            finally:
                conn=getattr(task,'_conn',None)
                if conn is not None:conn.close()
        result[name]=stats(prompts)
        print(name,'calibration complete',len(prompts),flush=True)
    save(output,dict(created_at=time.time(),calibration=result,alf_training_bindings=evidence,
        alf_preflight=concurrent_preflight(old['data_root'],games,workers=3,repeats=2,steps=2),
        policy='ALF training inputs only; CLBench first 20% public prompts only, primary scores on last 80%; no labels/outcomes'))


def prepare(root,args):
    import yaml
    if root.exists():raise FileExistsError('Use a new immutable run directory')
    w=workspace();oldroot=w/'ttcl/results/alfworld_comparison/20260924_parserfix'
    alf=read(oldroot/'plan.json');split=read(oldroot/'split_manifest.json')
    if not split['checks_passed']:raise ValueError('Historical split audit failed')
    calibration=read(args.calibration)
    for seq in alf['sequences']:
        for task in seq['tasks']:
            if task['split']!='valid_unseen' or sha(Path(alf['data_root'])/task['path'])!=task['sha256']:
                raise ValueError('ALFWorld frozen data mismatch')
    root.mkdir(parents=True)
    source=root/'source';(source/'ttcl').mkdir(parents=True)
    (source/'ttcl/__init__.py').write_text('')
    for name in ('memrl_comparison','icl_mem0_comparison','structured_memory','common',
                 'alfworld_comparison','experience_evolution','reflexion_expel'):
        shutil.copytree(w/'ttcl'/name,source/'ttcl'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(w/'current_work/continual-learning-bench/src',source/'bench/src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (source/'bench/data').symlink_to(w/'current_work/continual-learning-bench/data',target_is_directory=True)
    shutil.copytree(w/'current_work/MemRL/memrl',source/'upstream/memrl',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(w/'current_work/MemRL/LICENSE',source/'upstream/LICENSE')
    shutil.copy2(w/'current_work/MemRL/configs/rl_alf_config.yaml',source/'upstream/rl_alf_config.yaml')
    shutil.copy2(oldroot/'split_manifest.json',root/'split_manifest.json')
    shutil.copy2(args.calibration,root/'calibration.json')
    cfg=yaml.safe_load((source/'upstream/rl_alf_config.yaml').read_text())['rl_config'];cfg.pop('q_min_threshold')
    slimcal={name:{k:v for k,v in entry.items() if k!='public_prompts'} for name,entry in calibration['calibration'].items()}
    alf={k:alf[k] for k in ('data_root','model','context','actor_temperature','actor_max_tokens','max_steps','max_attempts','eval_seeds','sequences','test_tasks')}
    alf['actor_url']=f'http://127.0.0.1:{args.port}'
    plan=dict(schema_version=1,created_at=time.time(),model=alf['model'],upstream=str(source/'upstream'),
        upstream_provenance=read(w/'config/upstreams.json')['repositories']['MemRL'],alf=alf,
        embedding=str(w/'models/embedding/bge-m3'),embedding_revision='5617a9f61b028005a4858fdac845db406aefb181',
        arms=['none','memrl'],tasks=TASKS,repeats=[303,404],task_seed=42,context=65536,
        memory_tokens=2048,writer_max_tokens=768,action_retries=2,max_turns_per_instance=64,
        rl_config=cfg,q_min_thresholds={'alfworld':-10,'clbench':None},calibration=slimcal,expected_cells={'alfworld':134*3*2,'clbench':sum(TASKS.values())*2*2},
        url=f'http://127.0.0.1:{args.port}',port=args.port,gpu_candidates=args.gpus,workers=4,
        worker_python=str(Path(sys.executable).absolute()),server_python=str(Path(args.server_python).absolute()),
        dtype='bfloat16',kv_cache_dtype='auto',gpu_memory_utilization=.80,
        policy=dict(method='Official unchanged retrieval, Q updater, builder/reflection ASTs; local JSON storage and BGE transport',
            scope='Budget-matched online method reproduction; not paper 3553-training-task x 10-epoch reproduction',
            memory_view='Store full native trajectory; expose only complete task+script/reflection entries within 2048 tokens; credit only exposed memories',
            writer_limit='Retain nonempty text at the declared 768-token generation cap, like native provider; flag every length finish; no extension/retry',
            rewards='ALF official success mapped to +1/-1; CL official continuous reward unchanged. Scalar goes only to Q; official success bit selects native writer branch, including failure instruction. No grader metadata or ground-truth labels in writer text',
            cl_success_none='No inferred binary label: use script builder and retain success=null metadata',
            cl_q_threshold='Disable ALF -10 Q cutoff for unbounded continuous rewards; retain native scoring and Q update',
            updates='After each completed ALF attempt or CL instance; memory shared only inside domain/family and seed chain',
            calibration='ALF historical train prompts; CL first 20% public prompts. No reward-based parameter selection',
            primary='ALF first-attempt and within-three; CL paired last 80%, after-first/all secondary; failures separately, never zero-filled',
            backend='New none baseline primary. Historical Mem0 uses 262144 context/FP8 KV and different feedback; descriptive only',
            unavailable_domains={'sales_prediction':'Docker environment unavailable in existing protocol','codebase_adaptation':'Docker environment unavailable in existing protocol'},
            resume='Completed cells immutable; interrupted partial episode requires explicit reviewed recovery, never silent replay'))
    save(root/'plan.json',plan)
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    save(root/'worker_packages.json',packages)
    (root/'server_packages.txt').write_text(subprocess.check_output([plan['server_python'],'-m','pip','freeze'],text=True))
    paths=[p for p in source.rglob('*') if p.is_file()]
    paths += [p for p in (w/'current_work/continual-learning-bench/data').rglob('*') if p.is_file()]
    paths += [p for p in Path(plan['model']).iterdir() if p.is_file() and p.suffix in ('.json','.safetensors','.jinja')]
    paths += [p for p in Path(plan['embedding']).rglob('*') if p.is_file() and '.cache' not in p.parts]
    paths += [p for folder in (w/'ttcl/.runtime/structured_memory_deps',w/'ttcl/.runtime/deltamem_benchmark_deps')
              for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths += [root/'plan.json',root/'split_manifest.json',root/'calibration.json',root/'worker_packages.json',root/'server_packages.txt']
    bindings={str(p):sha(p) for p in paths}
    for seq in alf['sequences']:
        for item in seq['tasks']:bindings[str(Path(alf['data_root'])/item['path'])]=item['sha256']
    bindings.update(calibration['alf_training_bindings'])
    save(root/'input_hashes.json',bindings)
    save(root/'prepared.json',dict(manifest_sha256=sha(root/'input_hashes.json'),files=len(bindings)))
    save(root/'status.json',dict(phase='prepared',recorded_cells=0,updated_at=time.time()))
    print(json.dumps({'root':str(root),'expected_cells':plan['expected_cells'],'files':len(bindings)},indent=2))


def verify(root):
    if sha(root/'input_hashes.json')!=read(root/'prepared.json')['manifest_sha256']:raise ValueError('Manifest changed')
    for path,expected in read(root/'input_hashes.json').items():
        if sha(path)!=expected:raise ValueError(f'Frozen input changed: {path}')


def summarize(root):
    import math
    plan=read(root/'plan.json')
    rows=[read(p) for p in (root/'runs').glob('*/*/*/*/episode_*/row.json')]
    scores={}
    for benchmark in ('alfworld','clbench'):
        domains=sorted({r['task'] for r in rows if r['benchmark']==benchmark})
        if benchmark=='alfworld' and domains:domains.append('all')
        for task in domains:
            groups={a:{(r['repeat'],r.get('game',r.get('canonical_index'))):r for r in rows if r['benchmark']==benchmark and (r['task']==task or task=='all') and r['arm']==a
                and r['status']=='complete' and isinstance(r.get('reward'),(int,float)) and math.isfinite(r['reward'])} for a in plan['arms']}
            common=set(groups['none'])&set(groups['memrl']);table={}
            selections={'all':common}
            if benchmark=='clbench':
                selections.update(after_first={k for k in common if k[1]>0},last_80_percent={k for k in common if k[1]>=int(plan['tasks'][task]*.2)})
            for label,keys in selections.items():
                for k in keys:
                    if groups['none'][k]['initial_query_sha256']!=groups['memrl'][k]['initial_query_sha256']:raise ValueError('Mismatched paired inputs')
                metrics=('first_attempt','within_three') if benchmark=='alfworld' else ('reward',)
                table[label]={metric:dict(n=len(keys),means={a:statistics.fmean(groups[a][k][metric] for k in keys) if keys else None for a in plan['arms']}) for metric in metrics}
            scores[f'{benchmark}/{task}']=table
    result=dict(updated_at=time.time(),recorded_cells=len(rows),expected_cells=sum(plan['expected_cells'].values()),
        scored_cells=sum(r['status']=='complete' for r in rows),failed_cells=sum(r['status']!='complete' for r in rows),
        memory_update_failures=sum(r.get('memory_update_status')=='failed' for r in rows),scores=scores)
    result['writer_token_limit_hits']=sum(r.get('writer_token_limit_hits',0) for r in rows)
    save(root/'summary.json',result)
    lines=['# MemRL 运行结果','',f"已记录 {result['recorded_cells']}/{result['expected_cells']}；可计分 {result['scored_cells']}；未计分失败 {result['failed_cells']}。",
        '','这是同预算的方法适配，不是论文完整训练结果。等待 GPU 时正式结果为 0；训练预检不计入分数。',
        '','|基准/领域|口径|指标|共同有效样本|无记忆|MemRL|','|---|---|---|---:|---:|---:|']
    for name,selections in scores.items():
        for label,metrics in selections.items():
            if name.startswith('clbench/') and label!='last_80_percent':continue
            for metric,d in metrics.items():
                if d['n']:lines.append(f"|{name}|{label}|{metric}|{d['n']}|{d['means']['none']:.6f}|{d['means']['memrl']:.6f}|")
    lines+=['','ALFWorld 沿用 134 道 valid_unseen、三个种子、每题最多三次；CLBench 主指标排除前 20% 校准前缀。',
            '失败不补零；两臂共同有效样本计分。阶段和任务异常见 status.json，逐条调用和新记忆内容绑定见 runs/。',
            '完整适配差异见 source/ttcl/memrl_comparison/README.md 和 plan.json。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


def server_command(plan):
    return [plan['server_python'],'-m','vllm.entrypoints.openai.api_server','--model',plan['model'],
        '--served-model-name','frozen-actor','--host','127.0.0.1','--port',str(plan['port']),
        '--dtype','bfloat16','--max-model-len',str(plan['context']),'--kv-cache-dtype','auto',
        '--gpu-memory-utilization','.80','--max-num-seqs','4','--enable-prefix-caching',
        '--enable-chunked-prefill','--max-num-batched-tokens','4096','--enforce-eager','--disable-log-requests']


def supervise(root):
    import requests
    plan=read(root/'plan.json');verify(root)
    env=environment(root);server=None;active=[]
    lock=(root/'supervisor.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu_lock=None
    try:
        consecutive={gpu:0 for gpu in plan['gpu_candidates']}
        while gpu_lock is None:
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            usage={int(line.split(',')[0]):int(line.split(',')[1]) for line in raw.splitlines()}
            for gpu in plan['gpu_candidates']:
                consecutive[gpu]=consecutive[gpu]+1 if usage[gpu]<1500 else 0
                if consecutive[gpu]<3:continue
                candidate=(workspace()/f'ttcl/.runtime/memrl_gpu_{gpu}.lock').open('w')
                try:fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:candidate.close();continue
                gpu_lock=candidate;chosen=gpu;break
            save(root/'status.json',dict(phase='waiting_for_gpu' if gpu_lock is None else 'starting_server',pid=os.getpid(),
                gpu_usage_mb=usage,gpu_candidates=plan['gpu_candidates'],updated_at=time.time(),recorded_cells=summarize(root)['recorded_cells']))
            if gpu_lock is None:time.sleep(30)
        verify(root)
        latest=int(subprocess.check_output(['nvidia-smi','--id',str(chosen),
            '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if latest>=1500:raise RuntimeError('GPU became occupied during verification; no server was launched')
        with socket.socket() as sock:sock.bind(('127.0.0.1',plan['port']))
        with (root/'server.log').open('a') as log:
            server=subprocess.Popen(server_command(plan),env=dict(env,CUDA_VISIBLE_DEVICES=str(chosen),
                VLLM_USE_V1='0',VLLM_WORKER_MULTIPROC_METHOD='spawn'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(root/'server.json',dict(pid=server.pid,gpu=chosen,command=server_command(plan)))
        session=requests.Session();session.trust_env=False
        for _ in range(180):
            if server.poll() is not None:raise RuntimeError('Own vLLM server exited during startup')
            try:
                if session.get(plan['url']+'/v1/models',timeout=3).ok:break
            except requests.RequestException:pass
            time.sleep(5)
        else:raise RuntimeError('Server readiness timeout')
        save(root/'status.json',dict(phase='preflight',pid=os.getpid(),gpu=chosen,updated_at=time.time(),recorded_cells=0))
        # This runs on the exact frozen runtime, before any evaluation worker.
        subprocess.run([plan['worker_python'],'-m','ttcl.memrl_comparison.run','smoke','--root',str(root),
                        '--output',str(root/'preflight')],env=env,cwd=root/'source',check=True)
        alf_jobs=[('alfworld',s['family'],s['repeat']) for s in plan['alf']['sequences']]
        cl_jobs=[('clbench',t,r) for r in plan['repeats'] for t in plan['tasks']]
        # Interleave both benchmarks so either gets evidence as soon as resources are available.
        jobs=[]
        while alf_jobs or cl_jobs:
            if alf_jobs:jobs.append(alf_jobs.pop(0))
            if cl_jobs:jobs.append(cl_jobs.pop(0))
        completed=[];failed=[];(root/'logs').mkdir(exist_ok=True)
        while jobs or active:
            if server.poll() is not None:raise RuntimeError('Own vLLM server exited')
            while jobs and len(active)<plan['workers']:
                benchmark,task,repeat=jobs.pop(0)
                command=[plan['worker_python'],'-m','ttcl.memrl_comparison.run','worker','--root',str(root),
                         '--benchmark',benchmark,'--task',task,'--repeat',str(repeat)]
                with (root/'logs'/f'{benchmark}_{task}_{repeat}.log').open('a') as log:
                    p=subprocess.Popen(command,env=env,cwd=root/'source',stdout=log,stderr=subprocess.STDOUT)
                active.append((p,benchmark,task,repeat))
            still=[]
            for p,b,t,r in active:
                if p.poll() is None:still.append((p,b,t,r))
                else:(completed if p.returncode==0 else failed).append(dict(benchmark=b,task=t,repeat=r,exit_code=p.returncode))
            active=still;summary=summarize(root)
            save(root/'status.json',dict(phase='running' if jobs or active else ('complete' if not failed else 'finished_with_failures'),
                pid=os.getpid(),gpu=chosen,queued=jobs,active=[dict(pid=p.pid,benchmark=b,task=t,repeat=r) for p,b,t,r in active],
                completed=completed,failed=failed,recorded_cells=summary['recorded_cells'],updated_at=time.time()))
            if jobs or active:time.sleep(15)
    except Exception as exc:
        save(root/'status.json',dict(phase='failed',pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc(),updated_at=time.time()))
        raise
    finally:
        # Only children created by this supervisor are touched.
        for p,*_ in active:
            if p.poll() is None:p.terminate()
        if server and server.poll() is None:server.terminate()
        if gpu_lock:gpu_lock.close()
        lock.close()


def smoke(root,output,url=None):
    from ttcl.memrl_comparison.memory import Memory
    from ttcl.memrl_comparison.worker import alf_cell,cl_cell
    from ttcl.icl_mem0_comparison.protocol import Client
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    plan=read(root/'plan.json');plan['url']=url or plan['url'];plan['alf']['max_steps']=3;plan['alf']['max_attempts']=1
    client=Client(plan,303)
    calibration=read(root/'calibration.json')
    path=next(iter(calibration['alf_training_bindings']))
    game=str(Path(path).relative_to(plan['alf']['data_root']))
    if '/train/' not in game:raise ValueError('Preflight requires train split')
    m=Memory(plan,client,output/'memory',plan['calibration']['alfworld'])
    probes=['Put a clean mug on the table.','Find an apple and place it in the fridge.']
    embedder=m.service.embedding_provider
    batched=embedder.embed(probes)
    for text in probes:embedder.cache.pop(text,None)
    individual=[embedder.embed([text])[0] for text in probes]
    embedding_error=max(abs(x-y) for a,b in zip(batched,individual) for x,y in zip(a,b))
    if embedding_error>1.e-5:raise ValueError('Embedding batching changed vectors')
    row=alf_cell(plan,client,m,dict(path=game,sha256=sha(path),family='train_preflight'),303,'memrl',output/'alf_train')
    if row['status']!='complete':raise RuntimeError(row)
    if len(m.store.items)!=1:raise ValueError('Smoke write missing')
    query=next(iter(m.service.dict_memory));retrieval=m.retrieve(query)
    if not retrieval['ids']:raise ValueError('Smoke retrieval empty')
    value=m.service.update_value(retrieval['ids'][0],1.)
    if abs(value-.3)>1.e-6:raise ValueError('Smoke Q update failed')
    os.chdir(os.environ['TTCL_BENCH'])
    # First CL instance belongs to declared calibration prefix, never primary test.
    clmemory=Memory(dict(plan,q_min_threshold=None),client,output/'cl_memory',plan['calibration']['blind_spectrum_monitoring'],embedder=m.service.embedding_provider)
    clrow=cl_cell(plan,client,clmemory,'blind_spectrum_monitoring',0,'memrl',output/'cl_calibration')
    if clrow['status']!='complete' or clrow.get('memory_update_status')!='complete':raise RuntimeError(clrow)
    clquery=next(iter(clmemory.service.dict_memory))
    if not clmemory.retrieve(clquery)['ids']:raise ValueError('CL retrieval failed')
    save(output/'report.json',dict(passed=True,url=plan['url'],alf_train_steps=row['actor_calls'],
        cl_calibration_scored=True,official_q_update=value,embedding_batch_max_error=embedding_error,formal_evaluation_used=False,
        note='Ephemeral memories discarded; preflight rewards are not experiment results'))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['calibrate','prepare','verify','supervise','worker','status','smoke'])
    p.add_argument('--root',type=Path,default=workspace()/'results/memrl_comparison/20260928')
    p.add_argument('--output',type=Path);p.add_argument('--calibration',type=Path)
    p.add_argument('--port',type=int,default=18527);p.add_argument('--gpus',type=int,nargs='+',default=[6,0])
    p.add_argument('--server-python',default=os.environ.get('TTCL_PYTHON',str(workspace()/'ttcl/.runtime/alf_delta_env/bin/python')))
    p.add_argument('--benchmark',choices=['alfworld','clbench']);p.add_argument('--task');p.add_argument('--repeat',type=int);p.add_argument('--url')
    a=p.parse_args();root=a.root.resolve()
    if a.command=='calibrate':calibrate(a.output.resolve())
    elif a.command=='prepare':prepare(root,a)
    elif a.command=='verify':verify(root)
    elif a.command=='supervise':supervise(root)
    elif a.command=='status':print(json.dumps(summarize(root),indent=2))
    elif a.command=='smoke':smoke(root,a.output.resolve(),a.url)
    elif a.command=='worker':
        from .worker import work
        work(root,a.benchmark,a.task,a.repeat)


if __name__=='__main__':main()
