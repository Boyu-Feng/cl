"""Single-owner localhost inference; Delta state crosses episodes, KV never does.

Runs in the isolated upstream-compatible environment. Benchmark environments run
in the project's Python 3.12 environment and communicate using JSON over localhost.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import time
import traceback


class Engine:
    def __init__(self, model, adapter, output, context=65536):
        import torch
        from deltamem.runtime.session import DeltaMemChatSession, load_delta_mem_chat_model
        from deltamem.core.delta import iter_delta_mem_modules
        torch.cuda.set_per_process_memory_fraction(0.60, device=0)
        self.torch = torch
        self.output = Path(output).resolve()
        self.context = context
        self.model, self.tokenizer = load_delta_mem_chat_model(
            model_path=model, adapter_dir=adapter, device='cuda:0',
            dtype='bfloat16', attn_implementation='flash_attention_2')
        self.model.requires_grad_(False)
        self.wrappers = []
        for name, module in list(iter_delta_mem_modules(self.model)):
            parent_name, attr = name.rsplit('.', 1)
            self.wrappers.append((self.model.get_submodule(parent_name), attr, module))
        if not self.wrappers:
            raise RuntimeError('No Delta-Mem modules loaded')
        self.session = DeltaMemChatSession(self.model, self.tokenizer, device='cuda:0')
        self.active = False
        self.last_messages = []

    def state_path(self, value):
        path = Path(value).resolve()
        if not path.is_relative_to(self.output):
            raise ValueError('State path outside the experiment root')
        return path

    def begin(self, arm, previous=None):
        from deltamem.core.delta import load_delta_mem_online_state
        if arm not in ('base', 'deltamem_reset', 'deltamem_online'):
            raise ValueError(arm)
        # Restore wrappers before resetting, including modules inactive in base mode.
        for parent, attr, wrapped in self.wrappers:
            setattr(parent, attr, wrapped)
        self.session.reset()
        if arm == 'base':
            if previous is not None:
                raise ValueError('Base control cannot inherit a memory state')
            for parent, attr, wrapped in self.wrappers:
                setattr(parent, attr, wrapped.base)
        self.initial_state = {}
        previous_sha256 = None
        if previous:
            path = self.state_path(previous)
            previous_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            self.initial_state = self.torch.load(path, map_location='cpu', weights_only=True)
            load_delta_mem_online_state(self.model, self.initial_state)
        self.arm, self.active, self.rebuilds = arm, True, 0
        self.last_messages = []
        return {'arm': arm, 'inherited_state_sha256': previous_sha256,
                'inherited_state_bytes': sum(v.numel()*v.element_size() for v in self.initial_state.values()),
                'inherited_messages': 0, 'inherited_kv_tokens': 0}

    def ingest(self, messages, generation_prompt):
        from deltamem.core.delta import load_delta_mem_online_state
        if not self.active:
            raise RuntimeError('Call begin before inference')
        ids = self.session._tokenize_messages(messages, add_generation_prompt=generation_prompt)
        if ids.shape[1] > self.context:
            raise ValueError('Context overflow; no silent truncation')
        # Packaging repairs / whitespace changes can revise already decoded tokens.
        # Rebuild THIS episode from its incoming state, never reset earlier experience
        # and never write the same prefix twice into a carried state.
        previous = self.session.processed_input_ids
        if previous is not None and self.session._common_prefix_len(ids) < previous.shape[1]:
            self.session.reset()
            load_delta_mem_online_state(self.model, self.initial_state)
            self.rebuilds += 1
        self.session.messages = copy.deepcopy(messages)
        return ids, self.session._ingest_full_ids(ids)

    def generate(self, messages, random_seed, tokens, temperature, top_p):
        start = time.monotonic()
        ids = self.session._tokenize_messages(messages, add_generation_prompt=True)
        if ids.shape[1] + tokens > self.context:
            raise ValueError(f'Context overflow: {ids.shape[1]}+{tokens}>{self.context}')
        ids, logits = self.ingest(messages, True)
        if logits is None:
            raise RuntimeError('Empty generation suffix')
        torch = self.torch
        with torch.random.fork_rng(devices=[torch.device('cuda:0')]):
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            generated = self.session._decode_generate(logits, tokens,
                do_sample=temperature > 0, temperature=temperature, top_p=top_p, top_k=0)
        raw = self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        self.last_messages = copy.deepcopy(messages) + [{'role': 'assistant', 'content': raw}]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        rendered_ids = self.tokenizer.encode(rendered, add_special_tokens=False)
        if rendered_ids != ids[0].tolist():
            raise ValueError('Delta runtime and historical actor chat templates differ')
        return {'raw_response': raw, 'input_tokens': int(ids.shape[1]),
                'output_tokens': int(generated.shape[1]),
                'finish_reason': 'length' if generated.shape[1] == tokens else 'stop',
                'actual_generation_seed': random_seed, 'served_model': self.arm,
                'rendered_prompt_sha256': hashlib.sha256(rendered.encode()).hexdigest(),
                'seconds': time.monotonic()-start, 'context_limit': self.context,
                'writer_adapter_enabled': False, 'delta_mem_enabled': self.arm != 'base',
                'episode_prefix_rebuilds': self.rebuilds,
                'prefill': self.session.last_ingest_stats}

    def finish(self, destination, public_feedback='', messages=None):
        from deltamem.core.delta import get_delta_mem_online_state
        transcript = copy.deepcopy(self.last_messages if messages is None else messages)
        if public_feedback:
            transcript.append({'role': 'user', 'content': public_feedback})
        if transcript:
            self.ingest(transcript, False)
        state = get_delta_mem_online_state(self.model)
        observed = self.session.processed_input_ids is not None or bool(self.initial_state)
        if self.arm != 'base' and ((observed and not state) or not all(self.torch.isfinite(t).all() for t in state.values())):
            raise ValueError('Missing or non-finite online state')
        if self.arm == 'base' and state:
            raise ValueError('Base unexpectedly has Delta memory')
        path = self.state_path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        temporary = path.with_suffix('.tmp')
        self.torch.save(state, temporary)
        temporary.replace(path)
        result = {'state_path': str(path), 'state_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'state_bytes': sum(v.numel()*v.element_size() for v in state.values()),
                  'state_modules': len(state), 'prefix_rebuilds': self.rebuilds,
                  'public_feedback_sha256': hashlib.sha256(public_feedback.encode()).hexdigest()}
        self.active = False
        return result


def main():
    parser = argparse.ArgumentParser()
    for name in ('model', 'adapter', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    engine = Engine(args.model, args.adapter, args.output)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, value):
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(200, {'ready': True, 'delta_modules': len(engine.wrappers)})

        def do_POST(self):
            try:
                data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                functions = {'/begin': engine.begin, '/generate': engine.generate, '/finish': engine.finish}
                self.reply(200, functions[self.path](**data))
            except Exception as error:
                traceback.print_exc()
                self.reply(500, {'error': repr(error), 'traceback': traceback.format_exc()})

    print(json.dumps({'ready': True, 'port': args.port}), flush=True)
    HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
