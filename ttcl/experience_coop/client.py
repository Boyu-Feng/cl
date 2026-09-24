"""Exact sampled token IDs and old log probabilities from the installed vLLM API."""
from __future__ import annotations

import math
import time

from ttcl.experience_v2.common import Client as BaseClient
from ttcl.experience_evolution.core import seed
from .protocol import binding, validate_sample


def decode_sample(choice, prefix, messages, model, random_seed, temperature, top_p):
    logprobs = choice.get('logprobs')
    if not logprobs:
        raise ValueError('Server did not return sampled-token probabilities')
    tokens = logprobs['tokens']
    if any(not t.startswith('token_id:') for t in tokens):
        raise ValueError('Exact token IDs required; never retokenize generated text for PPO')
    response = [int(t.split(':', 1)[1]) for t in tokens]
    probabilities = logprobs['token_logprobs']
    if not response or len(response) != len(probabilities) or any(
            p is None or not math.isfinite(p) for p in probabilities):
        raise ValueError('Incomplete sampled token probabilities')
    sample = {'input_ids': prefix+response, 'prompt_length': len(prefix),
        'old_logp': probabilities, 'input_binding': binding(messages), 'model': model,
        'seed': random_seed, 'temperature': temperature, 'top_p': top_p,
        'finish_reason': choice['finish_reason']}
    sample['sample_binding'] = binding({k: sample[k] for k in
        ['input_ids', 'prompt_length', 'old_logp', 'input_binding', 'model', 'seed']})
    if temperature == 1. and top_p == 1.:
        validate_sample(sample)
    return sample


class Client(BaseClient):
    def __init__(self, plan, actor_model='frozen-actor', training=False, capture=False):
        super().__init__(plan['actor_url'], context=plan['context'])
        self.plan, self.actor_model = plan, actor_model
        self.training, self.capture, self.records = training, capture, []

    def complete(self, messages, model='frozen-actor', random_seed=1, tokens=768,
                 temperature=1., top_p=1., capture=False):
        prefix = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if len(prefix)+tokens > self.context:
            raise ValueError('Context overflow; no silent prompt truncation')
        random_seed %= 2**32
        payload = {'model': model, 'prompt': prefix, 'seed': random_seed,
            'max_tokens': tokens, 'temperature': temperature, 'top_p': top_p,
            'top_k': -1, 'repetition_penalty': 1., 'add_special_tokens': False}
        if capture:
            payload.update(logprobs=0, return_tokens_as_token_ids=True)
        started = time.monotonic()
        response = self.session.post(self.url+'/v1/completions', json=payload, timeout=900)
        response.raise_for_status()
        data = response.json(); choice = data['choices'][0]
        if data['usage']['prompt_tokens'] != len(prefix):
            raise ValueError('Tokenizer mismatch')
        result = {'raw_response': choice['text'].strip(), 'input_tokens': len(prefix),
            'output_tokens': data['usage']['completion_tokens'], 'finish_reason': choice['finish_reason'],
            'served_model': model, 'rendered_prompt_sha256': binding(prefix),
            'actual_generation_seed': random_seed, 'seconds': time.monotonic()-started}
        if capture:
            result['sample'] = decode_sample(choice, prefix, messages, model, random_seed, temperature, top_p)
            if len(result['sample']['old_logp']) != result['output_tokens']:
                raise ValueError('Generated token accounting mismatch, including EOS')
            decoded = self.tokenizer.decode(result['sample']['input_ids'][len(prefix):],
                                            skip_special_tokens=True).strip()
            if decoded != result['raw_response']:
                raise ValueError('Sampled token IDs do not decode to the executed text')
        return result

    def generate(self, messages, random_seed):
        result = self.complete(messages, self.actor_model, seed(random_seed, self.repeat), 4096,
                               1. if self.training else .7, 1. if self.training else .9,
                               capture=self.capture)
        if 'sample' in result:
            self.records.append(result.pop('sample'))
        return result
