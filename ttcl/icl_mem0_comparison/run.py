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
from ttcl.icl_mem0_comparison.protocol import (
    ARMS, TASKS, EMBEDDING_REPO, EMBEDDING_REVISION, read, save, sha,
)


def workspace():
    return Path(os.environ.get('TTCL_WORKSPACE', Path(__file__).resolve().parents[2])).resolve()


def embedding_download():
    from huggingface_hub import snapshot_download
    target = workspace()/'models/embedding/bge-m3'
    snapshot_download(EMBEDDING_REPO, revision=EMBEDDING_REVISION, local_dir=str(target),
        allow_patterns=['*.json', 'pytorch_model.bin', 'sentencepiece.bpe.model', '*.txt', '1_Pooling/*', 'README.md'],
        ignore_patterns=['onnx/*'])
    save(target/'download_provenance.json', dict(repo_id=EMBEDDING_REPO, revision=EMBEDDING_REVISION))
    bm25_revision = '22b8d2af71a76161e18dd432d2cee0eefa66e412'
    cache = workspace()/'models/embedding/fastembed'
    snapshot_download('Qdrant/bm25', revision=bm25_revision, local_dir=str(cache/'bm25'))
    save(cache/'download_provenance.json', dict(repo_id='Qdrant/bm25', revision=bm25_revision))
    # spaCy's loader requires a model package; install only that package under models/.
    nlp = workspace()/'models/nlp'
    if not (nlp/'en_core_web_sm').exists():
        subprocess.check_call([sys.executable,'-m','pip','install','--no-deps','--target',str(nlp),
            'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'])


def prepare(root, args):
    if root.exists():
        raise FileExistsError('Use a new directory; existing results and frozen inputs are preserved')
    w = workspace(); bench = w/'current_work/continual-learning-bench'
    model = w/'current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'
    embedding = w/'models/embedding/bge-m3'
    for p in (model/'config.json', embedding/'pytorch_model.bin'):
        if not p.exists():
            raise FileNotFoundError(p)
    root.mkdir(parents=True)
    plan = dict(schema_version=1, created_at=time.time(), model=str(model), embedding=str(embedding),
        embedding_repo=EMBEDDING_REPO, embedding_revision=EMBEDDING_REVISION,
        tasks=TASKS, historical_counts={**TASKS, 'database_exploration':20}, repeats=[303,404], arms=list(ARMS), expected_cells=sum(TASKS.values())*2*3,
        task_seed=42, context=262144, max_attempts=1, action_retries=2, max_turns_per_instance=64,
        actor_temperature=.7, actor_top_p=.9, actor_max_tokens=4096,
        writer_temperature=0., writer_max_tokens=4096, mem0_top_k=10,
        url=f'http://127.0.0.1:{args.port}', gpu=args.gpu, port=args.port,
        worker_python=str(Path(sys.executable).absolute()),
        server_python=str(Path(args.server_python).absolute()), workers=4,
        server_env={'CUDA_VISIBLE_DEVICES':str(args.gpu), 'VLLM_USE_V1':'0',
                    'VLLM_WORKER_MULTIPROC_METHOD':'spawn', 'OMP_NUM_THREADS':'4'},
        dtype='bfloat16', kv_cache_dtype='fp8_e5m2', gpu_memory_utilization=.80,
        policy=dict(training=False, adapters=False, scalar_reward_visible=False,
            feedback='official public observations only; identical visibility in all three arms',
            memory_reset='between domain/seed chains; preserved between tasks',
            icl='all public turns, raw outputs, schema, public feedback; no truncation or summarization',
            mem0='native mem0ai 2.0.0 extraction/additive storage/search; local Qwen transport',
            embedding='BGE-M3 dense CPU; native 8192 token limit, any embedding truncation audited',
            comparisons='new same-backend baseline is primary; old Delta/Reflexion/ExpeL are descriptive',
            missing='unscored failures preserved, never changed to zero; only common valid pairs averaged',
            selection='all local instances in canonical order; no checkpoint or prompt selection on scores',
            retries='two JSON formatting retries only; no task replay or best-of-N'))
    save(root/'plan.json', plan)
    source = root/'source'; (source/'ttcl').mkdir(parents=True)
    (source/'ttcl/__init__.py').write_text('')
    for name in ('icl_mem0_comparison', 'structured_memory', 'common'):
        shutil.copytree(w/'ttcl'/name, source/'ttcl'/name,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(bench/'src', source/'bench/src', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (source/'bench/data').symlink_to(bench/'data', target_is_directory=True)
    versions = {d.metadata['Name']:importlib.metadata.version(d.metadata['Name']) for d in importlib.metadata.distributions()}
    save(root/'worker_packages.json', versions)
    server_versions = subprocess.check_output([plan['server_python'],'-m','pip','freeze'], text=True)
    (root/'server_packages.txt').write_text(server_versions)
    req = w/'config/environments/requirements-icl-mem0.txt'
    shutil.copy2(req, root/req.name)
    paths = [p for p in source.rglob('*') if p.is_file()]
    paths += [p for p in (bench/'data').rglob('*') if p.is_file()]
    paths += [p for p in model.iterdir() if p.is_file() and p.suffix in ('.json','.safetensors','.jinja')]
    paths += [p for p in embedding.rglob('*') if p.is_file() and '.cache' not in p.parts]
    paths += [p for folder in (w/'models/nlp', w/'models/embedding/fastembed') for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.locks' not in p.parts]
    paths += [root/'plan.json',root/'worker_packages.json',root/'server_packages.txt',root/req.name]
    save(root/'input_hashes.json', {str(p):sha(p) for p in paths})
    save(root/'prepared.json', dict(manifest_sha256=sha(root/'input_hashes.json'), files=len(paths)))
    print(json.dumps(plan,indent=2))


def verify(root):
    if sha(root/'input_hashes.json') != read(root/'prepared.json')['manifest_sha256']:
        raise ValueError('Manifest changed')
    for path, expected in read(root/'input_hashes.json').items():
        if sha(path) != expected:
            raise ValueError(f'Frozen input changed: {path}')


def environment(root):
    w = workspace()
    return dict(os.environ, TTCL_WORKSPACE=str(w), TTCL_BENCH=str(root/'source/bench'),
        PYTHONPATH=os.pathsep.join(map(str,[root/'source',root/'source/bench',w/'ttcl/.runtime/structured_memory_deps', w/'models/nlp'])),
        OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', MEM0_TELEMETRY='false',
        HF_HUB_OFFLINE='1', FASTEMBED_CACHE_PATH=str(w/'models/embedding/fastembed'), PYTHONUNBUFFERED='1')


def server_command(plan):
    return [plan['server_python'], '-m','vllm.entrypoints.openai.api_server',
        '--model',plan['model'],'--served-model-name','frozen-actor','--host','127.0.0.1',
        '--port',str(plan['port']),'--dtype','bfloat16','--max-model-len',str(plan['context']),
        '--kv-cache-dtype','fp8_e5m2','--calculate-kv-scales','--gpu-memory-utilization','.80',
        '--max-num-seqs','2','--enable-prefix-caching','--enable-chunked-prefill',
        '--max-num-batched-tokens','4096','--enforce-eager','--disable-log-requests']


def summarize(root):
    plan = read(root/'plan.json')
    rows = [read(p) for p in (root/'runs').glob('*/*/*/episode_*/row.json')]
    domains = {}
    for task in plan['tasks']:
        groups = {arm:{(r['repeat'],r['canonical_index']):r for r in rows if r['task']==task
                     and r['arm']==arm and r['status']=='complete'} for arm in ARMS}
        common = set.intersection(*(set(g) for g in groups.values()))
        for key in common:
            if len({(g[key]['instance_id'],g[key]['initial_query_sha256']) for g in groups.values()}) != 1:
                raise ValueError('Mismatched paired task')
        table = {}
        for label, keys in [('all',common),('after_first',{k for k in common if k[1]>0}),
                            ('last_80_percent',{k for k in common if k[1]>=int(plan['tasks'][task]*.2)}),
                            ('old_delta_range_after_first',{k for k in common if 0<k[1]<plan.get('historical_counts',plan['tasks'])[task]}),
                            ('old_reflexion_expel_range',{k for k in common if int(plan.get('historical_counts',plan['tasks'])[task]*.2)<=k[1]<plan.get('historical_counts',plan['tasks'])[task]})]:
            means = {a:statistics.mean(groups[a][k]['reward'] for k in keys) if keys else None for a in ARMS}
            table[label] = dict(paired_count=len(keys),means=means,
                deltas_vs_none={a:means[a]-means['none'] if keys else None for a in ARMS if a!='none'})
        domains[task] = table
    summary = dict(updated_at=time.time(),expected_cells=plan['expected_cells'],recorded_cells=len(rows),
        scored_cells=sum(r['status']=='complete' for r in rows),
        failed_cells=sum(r['status']!='complete' for r in rows),
        memory_warnings=sum(r['memory_warnings'] for r in rows),domains=domains,
        costs={arm:{key:sum(r.get(key,0) for r in rows if r['arm']==arm) for key in
            ['actor_calls','actor_input_tokens','actor_output_tokens','writer_calls','writer_input_tokens','writer_output_tokens']}
            for arm in ARMS})
    save(root/'summary.json', summary)
    return summary


def supervise(root, adopt_server=None):
    import requests
    plan = read(root/'plan.json'); verify(root)
    env = environment(root); server = None
    if adopt_server:
        previous = read(Path(adopt_server))
        if previous['command'] != server_command(plan) or previous['gpu'] != plan['gpu']:
            raise ValueError('Preflight server configuration differs from frozen protocol')
        os.kill(previous['pid'],0)
        save(root/'server.json', dict(previous,adopted_from=str(adopt_server)))
    else:
        with (root/'server.log').open('w') as log:
            server = subprocess.Popen(server_command(plan),env=dict(env,**plan['server_env']),
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(root/'server.json',dict(pid=server.pid,command=server_command(plan),gpu=plan['gpu']))
    session = requests.Session(); session.trust_env=False
    for _ in range(180):
        if server and server.poll() is not None:
            raise RuntimeError('Model server exited')
        try:
            response = session.get(plan['url']+'/v1/models',timeout=3)
            if response.ok:
                break
        except requests.RequestException:
            pass
        time.sleep(5)
    else:
        raise RuntimeError('Server readiness timeout')
    from ttcl.icl_mem0_comparison.protocol import Client
    # Fixed synthetic warmup, also fixes the first real input for dynamic KV scales.
    warmup = [{'role':'user','content':('The quick brown fox jumps over the lazy dog. 0123456789. ' * 512)
               + '\nReturn exactly READY.'}]
    calibration = Client(plan,303).complete(warmup,12345,tokens=8,temperature=0.,top_p=1.)
    save(root/'server_warmup.json',dict(messages=warmup,**calibration))
    jobs = [(t,r) for r in plan['repeats'] for t in plan['tasks']]
    active, completed, failed = [], [], []
    (root/'logs').mkdir(exist_ok=True)
    try:
        while jobs or active:
            while jobs and len(active)<plan['workers']:
                task,repeat=jobs.pop(0)
                command=[plan['worker_python'],'-m','ttcl.icl_mem0_comparison.run','worker',
                         '--root',str(root),'--task',task,'--repeat',str(repeat)]
                with (root/'logs'/f'{task}_{repeat}.log').open('w') as log:
                    p=subprocess.Popen(command,env=env,cwd=root/'source',stdout=log,stderr=subprocess.STDOUT)
                active.append((p,task,repeat))
            still=[]
            for p,task,repeat in active:
                code=p.poll()
                if code is None:
                    still.append((p,task,repeat))
                else:
                    (completed if code==0 else failed).append(dict(task=task,repeat=repeat,exit_code=code))
            active=still
            stats=summarize(root)
            save(root/'status.json',dict(phase='running' if jobs or active else ('complete' if not failed else 'finished_with_failures'),
                pid=os.getpid(),queued=jobs,active=[dict(pid=p.pid,task=t,repeat=r) for p,t,r in active],
                completed=completed,failed=failed,recorded_cells=stats['recorded_cells'],updated_at=time.time()))
            if jobs or active:
                time.sleep(15)
    finally:
        if server and not active:
            server.terminate()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['download-embedding','prepare','supervise','worker','status','verify'])
    p.add_argument('--root',type=Path,default=workspace()/'results/icl_mem0_comparison/20260927')
    p.add_argument('--gpu',type=int,default=6);p.add_argument('--port',type=int,default=18327)
    p.add_argument('--server-python',default=os.environ.get('TTCL_PYTHON',str(workspace()/'ttcl/.runtime/alf_delta_env/bin/python')))
    p.add_argument('--adopt-server',type=Path);p.add_argument('--task',choices=list(TASKS));p.add_argument('--repeat',type=int)
    args=p.parse_args(); root=args.root.resolve()
    if args.command=='download-embedding': embedding_download()
    elif args.command=='prepare':prepare(root,args)
    elif args.command=='verify':verify(root)
    elif args.command=='supervise':supervise(root,args.adopt_server)
    elif args.command=='status':print(json.dumps(summarize(root),indent=2))
    elif args.command=='worker':
        from ttcl.icl_mem0_comparison.worker import work
        work(root,args.task,args.repeat)


if __name__=='__main__':
    main()
