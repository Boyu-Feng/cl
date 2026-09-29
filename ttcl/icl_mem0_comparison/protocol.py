from __future__ import annotations
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time

os.environ['MEM0_TELEMETRY'] = 'false'
TASKS = {'blind_spectrum_monitoring': 90, 'exploitable_poker': 120,
         'database_exploration': 30, 'cohort_studies': 20}
ARMS = ('none', 'full_history_icl', 'mem0')
EMBEDDING_REPO = 'BAAI/bge-m3'
EMBEDDING_REVISION = '5617a9f61b028005a4858fdac845db406aefb181'


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def seed(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16) % (2**31 - 1)


def normalize_prompt(prompt, brief, index, count):
    if brief:
        prompt = prompt.replace(brief + '\n\n', '')
    prompt = re.sub(r'\bQuestion \d+/\d+', f'Question {index+1}/{count}', prompt)
    prompt = re.sub(r'## Study\s+\d+/\d+', f'## Study {index+1}/{count}', prompt)
    return re.sub(r'\bHand #\d+', f'Hand #{index+1}', prompt)


def public_messages(prompt, action, feedback):
    # Deliberately no outcome, scalar reward, canonical ID, or hidden task state.
    messages = [{'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': json.dumps(action, ensure_ascii=False)}]
    if feedback:
        messages.append({'role': 'user', 'content': 'FEEDBACK: ' + feedback})
    return messages


def search_query(prompt, previous):
    if previous is None:
        return prompt
    q, action, feedback = previous
    parts = [f'PREVIOUS QUERY: {q}', f'PREVIOUS RESPONSE: {json.dumps(action, ensure_ascii=False)}']
    if feedback:
        parts.append('PREVIOUS FEEDBACK: ' + feedback)
    return '\n'.join(parts + ['CURRENT QUERY: ' + prompt])


class Client:
    def __init__(self, plan, repeat):
        import requests
        from transformers import AutoTokenizer
        self.plan, self.repeat = plan, repeat
        self.tokenizer = AutoTokenizer.from_pretrained(plan['model'], local_files_only=True)
        self.session = requests.Session(); self.session.trust_env = False

    def complete(self, messages, random_seed, *, tokens=4096, temperature=.7, top_p=.9):
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        count = len(self.tokenizer.encode(rendered, add_special_tokens=False))
        if count + tokens > self.plan['context']:
            raise ValueError(f'Context overflow: {count}+{tokens}>{self.plan["context"]}; no history truncation')
        started = time.monotonic()
        result = self.session.post(self.plan['url'] + '/v1/completions', timeout=1800, json={
            'model': 'frozen-actor', 'prompt': rendered, 'seed': random_seed % 2**32,
            'max_tokens': tokens, 'temperature': temperature, 'top_p': top_p,
            'top_k': -1, 'repetition_penalty': 1.0, 'add_special_tokens': False})
        result.raise_for_status(); data = result.json()
        if count != data['usage']['prompt_tokens']:
            raise ValueError('Server/client tokenizer mismatch')
        choice = data['choices'][0]
        return dict(raw_response=choice['text'].strip(), input_tokens=count,
                    output_tokens=data['usage']['completion_tokens'], finish_reason=choice['finish_reason'],
                    actual_generation_seed=random_seed % 2**32,
                    rendered_prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                    context_limit=self.plan['context'], seconds=time.monotonic()-started)

    def generate(self, messages, random_seed):
        return self.complete(messages, seed(random_seed, self.repeat))


class NativeMemory:
    """Native SDK; adapt only local transport and audit swallowed SDK errors."""
    def __init__(self, plan, client, directory):
        os.environ['MEM0_DIR'] = str(Path(directory)/'_sdk')
        from mem0 import Memory
        self.directory, self.client = Path(directory), client
        self.directory.mkdir(parents=True, exist_ok=True)
        self.calls, self.input_tokens, self.output_tokens, self.errors = 0, 0, 0, []
        self.embedding_calls = 0
        self.user_id = 'benchmark_user'
        self.memory = Memory.from_config({
            'llm': {'provider': 'openai', 'config': {'model': 'frozen-actor', 'api_key': 'local-only',
                'openai_base_url': plan['url'] + '/v1', 'temperature': 0., 'max_tokens': 4096}},
            'embedder': {'provider': 'huggingface', 'config': {'model': plan['embedding'],
                'embedding_dims': 1024, 'model_kwargs': {'device': 'cpu', 'local_files_only': True}}},
            'vector_store': {'provider': 'qdrant', 'config': {'collection_name': 'clbench_mem0',
                'embedding_model_dims': 1024, 'path': str(self.directory/'qdrant')}},
            'history_db_path': str(self.directory/'history.db'),
        })
        # LLM interface is transport-only: prompts and state transitions stay in SDK.
        self.memory.llm = self
        native_embed = self.memory.embedding_model.embed
        embedding_model = self.memory.embedding_model.model

        def audited_embed(text, memory_action=None):
            count = len(embedding_model.tokenizer.encode(text, add_special_tokens=True))
            limit = embedding_model.max_seq_length
            self.embedding_calls += 1
            append(self.directory/'embedding_calls.jsonl', dict(call=self.embedding_calls,
                action=memory_action, input_tokens=count, max_seq_length=limit,
                truncated_by_native_embedder=count > limit,
                text_sha256=hashlib.sha256(text.encode()).hexdigest()))
            return native_embed(text, memory_action)
        self.memory.embedding_model.embed = audited_embed
        owner = self

        class SDKLog(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.WARNING:
                    event = dict(level=record.levelname, logger=record.name, message=record.getMessage())
                    owner.errors.append(event)
                    append(owner.directory/'sdk_warnings.jsonl', event)
        self.handler = SDKLog()
        logging.getLogger('mem0').addHandler(self.handler)

    def generate_response(self, messages, tools=None, tool_choice='auto', **kwargs):
        if tools:
            raise ValueError('Unexpected native SDK tool-call interface')
        self.calls += 1
        event = dict(call=self.calls, messages=messages, response_format=kwargs.get('response_format'))
        try:
            response = self.client.complete(messages, seed(91327, self.client.repeat, self.calls),
                tokens=4096, temperature=0., top_p=1.)
            self.input_tokens += response['input_tokens']; self.output_tokens += response['output_tokens']
            event.update(response)
            # Validate the SDK's expected JSON; invalid outputs remain visible failures.
            from mem0.memory.utils import remove_code_blocks, extract_json
            raw = remove_code_blocks(response['raw_response'])
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError:
                raw = extract_json(raw); decoded = json.loads(raw)
            if not isinstance(decoded, dict) or not isinstance(decoded.get('memory'), list):
                raise ValueError('Mem0 extraction must contain a memory list')
            if any(not isinstance(m, dict) or not isinstance(m.get('text'), str) for m in decoded['memory']):
                raise ValueError('Malformed Mem0 memory entries')
            event['valid_native_json'] = True
            return response['raw_response']
        except Exception as exc:
            event['error'] = repr(exc)
            raise
        finally:
            append(self.directory/'writer_calls.jsonl', event)

    def search(self, query):
        result = self.memory.search(query, filters={'user_id': self.user_id}, top_k=10)
        append(self.directory/'retrievals.jsonl', dict(query=query, result=result))
        return result.get('results', [])

    def add(self, messages):
        before = len(self.errors)
        result = self.memory.add(messages, user_id=self.user_id)
        event = dict(messages=messages, result=result, sdk_warnings=self.errors[before:])
        append(self.directory/'updates.jsonl', event)
        return event

    def snapshot(self, path):
        save(path, self.memory.get_all(filters={'user_id': self.user_id}, top_k=100000))

    def close(self):
        logging.getLogger('mem0').removeHandler(self.handler)
        self.memory.vector_store.client.close()
        if self.memory._entity_store is not None:
            self.memory._entity_store.client.close()
