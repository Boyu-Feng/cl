"""Frozen Qwen4B APEx pipeline: prerequisites -> full GRPO -> CLBench."""
from __future__ import annotations
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback
from .common import binding,read,save,sha,status,workspace


def prepare(root,args):
    w=workspace();root.mkdir(parents=True,exist_ok=True)
    if (root/'plan.json').exists() or (root/'source').exists():raise FileExistsError('Never overwrite a frozen experiment')
    spec=read(w/'config/apex-qwen4b.json')
    base=w/'models/delta_mem/Qwen3-4B-Instruct-2507'
    if not (base/'config.json').exists():base=w/'current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'
    if not (base/'config.json').exists():raise FileNotFoundError(base)
    if not (root/'asset_plan.json').exists():shutil.copy2(w/'config/apex-qwen4b-assets.json',root/'asset_plan.json')
    if not (root/'dataset.json').exists():
        from .data import prepare as prepare_data
        prepare_data(root)
    spec.update(base_model=str(base.resolve()),python=str(Path(sys.executable).absolute()),
        model_output=str(w/'models/apex_qwen4b'/root.name),gpu=args.gpu,judge_gpus=args.judge_gpus,
        judge_url=f'http://127.0.0.1:{args.judge_port}/v1',retrieval_url=f'http://127.0.0.1:{args.retrieval_port}',
        judge_port=args.judge_port,retrieval_port=args.retrieval_port,created_at=time.time())
    if Path(spec['model_output']).exists():raise FileExistsError('New checkpoint directory required')
    save(root/'plan.json',spec)
    source=root/'source';(source/'ttcl').mkdir(parents=True);(source/'ttcl/__init__.py').write_text('')
    for name in ['apex_qwen4b','structured_memory','llm_memory','common']:
        shutil.copytree(w/'ttcl'/name,source/'ttcl'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (source/'ttcl/icl_mem0_comparison').mkdir()
    for name in ['__init__.py','protocol.py']:
        shutil.copy2(w/'ttcl/icl_mem0_comparison'/name,source/'ttcl/icl_mem0_comparison'/name)
    shutil.copytree(w/'current_work/APEx',source/'ttcl/apex_qwen4b/official',ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
    bench=w/'current_work/continual-learning-bench'
    shutil.copytree(bench/'src',source/'bench/src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (source/'bench/data').symlink_to((bench/'data').resolve(),target_is_directory=True)
    for name in ['LICENSE','AGENTS.md']:
        if (bench/name).is_file():shutil.copy2(bench/name,source/'bench'/name)
    save(root/'packages.json',{d.metadata['Name']:d.version for d in importlib.metadata.distributions()})
    paths=[p for p in source.rglob('*') if p.is_file()]
    paths += [p for p in (bench/'data').rglob('*') if p.is_file()]
    paths += [p for p in base.iterdir() if p.is_file() and p.suffix in ['.json','.safetensors','.jinja','.txt']]
    paths += [Path(p) for p in read(root/'dataset.json')['files']]
    paths += [root/name for name in ['plan.json','asset_plan.json','dataset.json','packages.json']]
    save(root/'input_hashes.json',{str(p):sha(p) for p in paths})
    save(root/'prepared.json',{'manifest_sha256':sha(root/'input_hashes.json'),'files':len(paths),'time':time.time()})
    status(root,'frozen_ready_for_launch');print(json.dumps({'root':str(root),'files':len(paths)}))


def verify(root):
    if sha(root/'input_hashes.json')!=read(root/'prepared.json')['manifest_sha256']:raise ValueError('Frozen manifest changed')
    for filename,expected in read(root/'input_hashes.json').items():
        if sha(filename)!=expected:raise ValueError('Frozen input changed: '+filename)


def environment(root,gpus=None,offline=True):
    w=workspace();p=read(root/'plan.json')
    return dict(os.environ,TTCL_WORKSPACE=str(w),TTCL_BENCH=str(root/'source/bench'),
        PYTHONPATH=os.pathsep.join(map(str,[root/'source',root/'source/bench',w/'ttcl/.runtime/structured_memory_deps',w/'ttcl/.runtime/deltamem_benchmark_deps'])),
        CUDA_VISIBLE_DEVICES=str(p['gpu']) if gpus is None else gpus,
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONUNBUFFERED='1',
        HF_HUB_OFFLINE='1' if offline else '0',HF_HUB_DISABLE_XET='1',HF_HUB_DOWNLOAD_TIMEOUT='120',
        HF_HUB_ETAG_TIMEOUT='60',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        NO_PROXY='127.0.0.1,localhost',no_proxy='127.0.0.1,localhost')


def health(url):
    import requests
    try:
        s=requests.Session();s.trust_env=False
        return s.get(url,timeout=5).status_code==200
    except requests.RequestException:return False


def wait_gpu(root,gpus):
    while True:
        r=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True)
        used={int(a.strip()):int(b.strip()) for a,b in (line.split(',') for line in r.stdout.splitlines())}
        if all(used[int(g)]<1500 for g in gpus.split(',')):return
        save(root/'supervisor_status.json',{'phase':'waiting_for_gpu','devices':gpus,'memory_mib':used,'time':time.time()})
        time.sleep(20)


def summary(root):
    plan=read(root/'plan.json');cfg=plan['clbench']
    rows=[read(p) for p in (root/'clbench').glob('*/*/*/episode_*/result.json')]
    domains={}
    for domain in cfg['tasks']:
        groups={arm:{(r['repeat'],r['canonical_index']):r for r in rows if r['task']==domain and r['arm']==arm and r['status']=='complete'} for arm in cfg['arms']}
        common=set.intersection(*(set(g) for g in groups.values()))
        for key in common:
            if len({(g[key]['instance_id'],g[key]['initial_query_binding']) for g in groups.values()})!=1:
                raise ValueError('Canonical pairing mismatch')
        table={}
        for label,keys in [('all_tasks',common),('after_first_task',{k for k in common if k[1]>0})]:
            means={arm:statistics.mean(groups[arm][k]['reward'] for k in keys) if keys else None for arm in cfg['arms']}
            table[label]={'paired_n':len(keys),'mean_official_reward':means,
                'delta_vs_base':{arm:means[arm]-means['base'] if keys else None for arm in cfg['arms'] if arm!='base'}}
        table['recorded']={arm:sum(r['task']==domain and r['arm']==arm for r in rows) for arm in cfg['arms']}
        table['failures']={arm:sum(r['task']==domain and r['arm']==arm and r['status']!='complete' for r in rows) for arm in cfg['arms']}
        domains[domain]=table
    candidates=[read(p) for p in (root/'clbench').glob('*/*/*/episode_*/candidate_*/row.json')]
    result={'time':time.time(),'expected_primary_records':sum(cfg['tasks'].values())*len(cfg['repeats'])*len(cfg['arms']),
        'primary_records':len(rows),'all_rollout_records':len(candidates),'domains':domains,
        'primary_selection':'candidate 0, fixed before sampling; not best of four',
        'costs':{arm:{k:sum(r.get(k,0) for r in candidates if r['arm']==arm) for k in
             ['actor_calls','actor_input_tokens','actor_output_tokens','planner_calls','planner_output_tokens']} for arm in cfg['arms']}}
    save(root/'summary.json',result);return result


def supervise(root):
    lock=(root/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify(root);plan=read(root/'plan.json');owned={};logs=root/'logs';logs.mkdir(exist_ok=True)
    def spawn(name,cmd,env):
        with (logs/(name+'.log')).open('x') as out:
            proc=subprocess.Popen(cmd,cwd=root/'source',env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        owned[name]=proc
        save(root/'processes'/f'{name}.json',{'pid':proc.pid,'command':cmd,'time':time.time()})
        return proc
    def ensure_alive(name):
        if owned[name].poll() is not None:raise RuntimeError(f'{name} exited with {owned[name].returncode}; see its preserved log')
    def wait_worker(proc):
        while proc.poll() is None:
            ensure_alive('judge')
            if 'retrieval' in owned:ensure_alive('retrieval')
            time.sleep(10)
        return proc.returncode
    py=plan['python'];module=lambda name:[py,'-m','ttcl.apex_qwen4b.'+name,'--root',str(root)]
    try:
        save(root/'supervisor_status.json',{'phase':'waiting_for_verified_assets','pid':os.getpid(),'time':time.time()})
        if not (root/'assets_ready.json').exists():
            existing=read(root/'asset_process.json') if (root/'asset_process.json').exists() else {}
            pid=existing.get('pid');alive=False
            if pid:
                try:alive='ttcl.apex_qwen4b.assets' in Path(f'/proc/{pid}/cmdline').read_text()
                except FileNotFoundError:pass
            if not alive:spawn('assets_supervised',module('assets'),environment(root,offline=False))
        # E5 indexing can overlap the much larger 32B judge download.
        while True:
            progress=read(root/'asset_progress.json') if (root/'asset_progress.json').exists() else {}
            if progress.get('errors'):raise RuntimeError('Asset errors; see asset_progress.json')
            done=progress.get('completed',[]);spec=read(root/'asset_plan.json')
            ready=all(any(d.get('file')==f['path'] for d in done) for f in spec['wiki']['files'])
            ready=ready and any(d.get('repo')=='intfloat/e5-base-v2' for d in done)
            if ready:break
            time.sleep(10)
        wait_gpu(root,str(plan['gpu']))
        if health(plan['retrieval_url']+'/health'):raise RuntimeError('Retrieval port already occupied')
        spawn('retrieval',module('retrieval')+['--port',str(plan['retrieval_port'])],environment(root))
        save(root/'supervisor_status.json',{'phase':'indexing_and_waiting_for_judge_weights','time':time.time()})
        while not (root/'assets_ready.json').exists():
            ensure_alive('retrieval')
            if read(root/'asset_progress.json').get('errors'):raise RuntimeError('Asset download failed')
            time.sleep(10)
        # Bind complete downloaded model bytes before they enter any experiment reward.
        model_files={}
        for item in read(root/'asset_plan.json')['models']:
            for p in (workspace()/item['directory']).iterdir():
                if p.is_file() and p.suffix in ['.safetensors','.bin','.json','.txt','.model','.jinja']:model_files[str(p)]=sha(p)
        save(root/'downloaded_model_hashes.json',model_files)
        wait_gpu(root,plan['judge_gpus'])
        if health(plan['judge_url']+'/models'):raise RuntimeError('Judge port already occupied')
        judge_path=workspace()/'models/apex/base/Qwen3-32B'
        cmd=[py,'-m','vllm.entrypoints.openai.api_server','--model',str(judge_path),'--served-model-name','qwen',
             '--host','127.0.0.1','--port',str(plan['judge_port']),'--dtype','bfloat16','--tensor-parallel-size','2',
             '--max-model-len',str(plan['judge_context']),'--gpu-memory-utilization','.94','--max-num-seqs','2',
             '--kv-cache-dtype','fp8_e5m2','--enforce-eager','--disable-log-requests','--enable-chunked-prefill','--max-num-batched-tokens','1024']
        env=environment(root,plan['judge_gpus']);env.update(VLLM_USE_V1='0',VLLM_WORKER_MULTIPROC_METHOD='spawn',NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1')
        spawn('judge',cmd,env)
        save(root/'supervisor_status.json',{'phase':'waiting_for_retrieval_and_judge_services','time':time.time()})
        while not health(plan['judge_url']+'/models') or not health(plan['retrieval_url']+'/health'):
            ensure_alive('retrieval');ensure_alive('judge');time.sleep(10)
        from .judge import Judge
        judge=Judge(plan['judge_url'],root/'judge_preflight.jsonl')
        if not judge.correct('What is 1 + 1?','2','2'):raise RuntimeError('Judge synthetic preflight failed')
        save(root/'supervisor_status.json',{'phase':'offline_training','time':time.time()})
        worker=spawn('training',module('train'),environment(root))
        if wait_worker(worker):raise RuntimeError('Training failed; see status.json and logs/training.log')
        if not (root/'training_complete.json').exists():raise RuntimeError('Missing final training artifact')
        retrieval=owned.pop('retrieval');os.killpg(retrieval.pid,signal.SIGTERM);retrieval.wait(timeout=60)
        # Docker availability is a prerequisite, never a numeric benchmark result.
        try:
            docker=subprocess.run(['docker','info','--format','{{.ServerVersion}}'],capture_output=True,text=True,timeout=20)
            docker_ready=docker.returncode==0;docker_reason=docker.stderr.strip() if not docker_ready else ''
        except (FileNotFoundError,subprocess.TimeoutExpired) as exc:docker_ready=False;docker_reason=repr(exc)
        save(root/'docker_preflight.json',{'available':docker_ready,'reason':docker_reason})
        failed=[];blocked=[]
        for domain in plan['clbench']['tasks']:
            if domain in ['sales_prediction','codebase_adaptation'] and not docker_ready:
                blocked.append({'domain':domain,'reason':'Docker unavailable','detail':docker_reason});continue
            for repeat in plan['clbench']['repeats']:
                for arm in plan['clbench']['arms']:
                    name=f'cl_{domain}_{repeat}_{arm}'
                    save(root/'supervisor_status.json',{'phase':'clbench','domain':domain,'repeat':repeat,'arm':arm,'time':time.time()})
                    proc=spawn(name,module('evaluate')+['--domain',domain,'--repeat',str(repeat),'--arm',arm],environment(root))
                    rc=wait_worker(proc)
                    if rc:failed.append({'domain':domain,'repeat':repeat,'arm':arm,'returncode':rc})
                    summary(root)
        save(root/'clbench_completion.json',{'failed':failed,'blocked':blocked,'time':time.time()})
        status(root,'complete' if not failed and not blocked else 'finished_with_failures_or_blocked_domains',failed=failed,blocked=blocked)
        save(root/'supervisor_status.json',read(root/'status.json'))
    except BaseException as exc:
        save(root/'supervisor_status.json',{'phase':'failed','error':repr(exc),'traceback':traceback.format_exc(),'time':time.time()})
        raise
    finally:
        for name,proc in reversed(list(owned.items())):
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','verify','launch','supervise','status'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',type=int,default=2)
    p.add_argument('--judge-gpus',default='1,3');p.add_argument('--judge-port',type=int,default=18431);p.add_argument('--retrieval-port',type=int,default=18430)
    args=p.parse_args();root=args.root.resolve()
    if args.command=='prepare':prepare(root,args)
    elif args.command=='verify':verify(root);print('Frozen input hashes verified')
    elif args.command=='supervise':supervise(root)
    elif args.command=='status':
        print(json.dumps({k:read(root/k) if (root/k).exists() else None for k in
            ['supervisor_status.json','status.json','retrieval_progress.json','asset_progress.json']},ensure_ascii=False,indent=2))
    elif args.command=='launch':
        if (root/'supervisor_process.json').exists():raise FileExistsError('Supervisor was already launched; inspect its artifacts')
        plan=read(root/'plan.json');(root/'logs').mkdir(exist_ok=True)
        with (root/'logs/supervisor.log').open('x') as out:
            proc=subprocess.Popen([plan['python'],'-m','ttcl.apex_qwen4b.run','supervise','--root',str(root)],
                cwd=root/'source',env=environment(root),stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        save(root/'supervisor_process.json',{'pid':proc.pid,'time':time.time()});print(proc.pid)


if __name__=='__main__':main()
