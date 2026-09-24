from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
import shutil
import subprocess
import time

from ttcl.experience_evolution.core import (
    WRITER_SYSTEM, read, save, seed, workspace_root, python_executable,
)

WORKSPACE = workspace_root()
BENCH = WORKSPACE / 'current_work/continual-learning-bench'
PYTHON = python_executable(WORKSPACE)
OLD = WORKSPACE / 'ttcl/results/experience_evolution/alfworld_delta_20260922'
MODEL = WORKSPACE / 'current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'
ROOT = WORKSPACE / 'ttcl/results/experience_v2/20260923'


def select_locomo_questions(samples, per_conversation=30, random_seed=923):
    """Fixed proportional category sample; never inspect answers or outcomes."""
    rng = random.Random(random_seed)
    selected = {}
    for i, sample in enumerate(samples):
        groups = defaultdict(list)
        for j, question in enumerate(sample['qa']):
            groups[int(question['category'])].append(j)
        total = len(sample['qa'])
        if total < per_conversation:
            raise ValueError('Too few questions for balanced conversation sampling')
        exact = {c: per_conversation*len(xs)/total for c,xs in groups.items()}
        quota = {c:int(v) for c,v in exact.items()}
        for c in sorted(groups, key=lambda c:(-(exact[c]-quota[c]),c))[:per_conversation-sum(quota.values())]:
            quota[c] += 1
        selected[str(i)] = sorted(j for c,xs in groups.items() for j in rng.sample(xs,quota[c]))
    return selected


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def environment(root):
    return dict(os.environ, TTCL_WORKSPACE=str(WORKSPACE),
                TTCL_BENCH=str(BENCH), PYTHONUNBUFFERED='1',
                TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
                ALFWORLD_DATA=str(WORKSPACE / 'ttcl/data/alfworld_delta'),
                PYTHONPATH=os.pathsep.join(map(str, [root / 'source',
                    WORKSPACE / 'ttcl/.runtime/structured_memory_deps', BENCH])))


def freeze(root):
    target = root / 'source/ttcl'
    target.mkdir(parents=True, exist_ok=False)
    (target / '__init__.py').write_text('')
    shutil.copy2(WORKSPACE / 'ttcl/paths.py', target / 'paths.py')
    for name in ['experience_v2', 'experience_evolution', 'common', 'llm_memory', 'structured_memory']:
        shutil.copytree(WORKSPACE / 'ttcl' / name, target / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(WORKSPACE / 'current_work/delta-Mem/deltamem/eval/locomo_protocol.py',
                 target / 'experience_v2/locomo_protocol.py')
    paths=[p for directory in [root/'source',root/'adapters']
           for p in directory.rglob('*') if p.is_file()]
    paths += [root/n for n in ['plan.json','training_plan.json','locomo_selection.json',
                               'locomo10.json','data_hashes.json','PROTOCOL.md']]
    paths += list((BENCH/'src').rglob('*.py'))
    save(root/'input_hashes.json',{str(p):sha_file(p) for p in paths})


class Client:
    def __init__(self, url, context=65536, repeat=303):
        import requests
        from transformers import AutoTokenizer
        self.url, self.context, self.repeat = url, context, repeat
        self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
        self.session = requests.Session()
        self.session.trust_env = False

    def complete(self, messages, model='frozen-actor', random_seed=1, tokens=768,
                 temperature=1.0, top_p=1.0):
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        count = len(self.tokenizer.encode(rendered, add_special_tokens=False))
        if count + tokens > self.context:
            raise ValueError(f'Context overflow: {count}+{tokens}>{self.context}; no silent truncation')
        started = time.monotonic()
        response = self.session.post(self.url + '/v1/completions', timeout=900, json={
            'model': model, 'prompt': rendered, 'seed': random_seed % 2**32,
            'max_tokens': tokens, 'temperature': temperature, 'top_p': top_p,
            'top_k': -1, 'repetition_penalty': 1.0, 'add_special_tokens': False})
        response.raise_for_status()
        data = response.json()
        if count != data['usage']['prompt_tokens']:
            raise ValueError('Tokenizer mismatch')
        choice = data['choices'][0]
        return dict(raw_response=choice['text'].strip(), input_tokens=count,
                    output_tokens=data['usage']['completion_tokens'],
                    finish_reason=choice['finish_reason'], served_model=model,
                    writer_adapter_enabled=model != 'frozen-actor',
                    rendered_prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                    actual_generation_seed=random_seed % 2**32,
                    context_limit=self.context, seconds=time.monotonic()-started)

    def generate(self, messages, random_seed):
        return self.complete(messages, random_seed=seed(random_seed, self.repeat),
                             tokens=4096, temperature=.7, top_p=.9)


def start_server(root, gpu, port, adapters, context=65536):
    import requests
    env = environment(root)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), VLLM_WORKER_MULTIPROC_METHOD='spawn')
    cmd = [str(PYTHON), '-m', 'vllm.entrypoints.openai.api_server', '--model', str(MODEL),
           '--served-model-name', 'frozen-actor', '--host', '127.0.0.1', '--port', str(port),
           '--dtype', 'bfloat16', '--max-model-len', str(context),
           '--gpu-memory-utilization', '.88', '--max-num-seqs', '16',
           '--enable-prefix-caching', '--enforce-eager', '--disable-log-requests']
    if adapters:
        cmd += ['--enable-lora', '--max-lora-rank', '8', '--max-loras', str(len(adapters)),
                '--lora-modules', *[f'{k}={v}' for k,v in adapters.items()]]
    save(root / f'server_{port}_command.json', cmd)
    with (root / f'server_{port}.log').open('a') as log:
        process = subprocess.Popen(cmd, env=env, cwd=root / 'source', stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    save(root / f'server_{port}_pid.json', {'pid': process.pid, 'gpu': gpu, 'port': port})
    session = requests.Session(); session.trust_env = False
    for _ in range(180):
        if process.poll() is not None:
            raise RuntimeError(f'Model server {port} exited; see log')
        try:
            names = {v['id'] for v in session.get(f'http://127.0.0.1:{port}/v1/models', timeout=3).json()['data']}
            if {'frozen-actor', *adapters} <= names:
                return process
        except (requests.RequestException, ValueError, KeyError):
            pass
        time.sleep(2)
    process.terminate()
    raise TimeoutError('Server startup timeout')


def stop_server(process):
    import signal
    if process and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def public_messages(memory, episode):
    # Only task-public observations. Never inject hidden CLBench terminal scores.
    public = {k: episode[k] for k in ['public_task_brief', 'initial_public_query',
                                     'response_schemas', 'steps', 'completed', 'format_failures']}
    return [{'role': 'system', 'content': WRITER_SYSTEM},
            {'role': 'user', 'content': json.dumps({'previous_experience': memory,
             'completed_interaction': public}, ensure_ascii=False)}]
