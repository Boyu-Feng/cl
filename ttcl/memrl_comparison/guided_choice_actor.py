"""Use the environment's public admissible commands as a decoding constraint.

One model completion still produces one action per step. The exact command
list already appears in the actor prompt; guided_choice only prevents output
outside that list. No task family, object name, expert command, extra actor
call, or reward is supplied to the decoder.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from ttcl.alfworld_comparison.environment import Actor
from ttcl.experience_evolution.core import save


COMMAND_MARKER = '\nAvailable commands:\n'


def advertised_commands(messages: list[dict]) -> list[str]:
    if not messages or messages[-1].get('role') != 'user':
        raise ValueError('Missing public actor observation')
    content = messages[-1]['content']
    if COMMAND_MARKER not in content:
        raise ValueError('Actor observation has no advertised action set')
    commands = [line for line in content.rsplit(COMMAND_MARKER, 1)[1].splitlines()
                if line]
    if not commands or len(set(commands)) != len(commands):
        raise ValueError('Empty or duplicate public action set')
    return commands


class GuidedChoiceActor(Actor):
    def __init__(self, plan, client):
        super().__init__(plan)
        self.client = client

    def generate(self, messages, random_seed):
        commands = advertised_commands(messages)
        rendered = self.client.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        count = len(self.client.tokenizer.encode(rendered,
                                                  add_special_tokens=False))
        tokens = self.plan['actor_max_tokens']
        if count + tokens > self.client.plan['context']:
            raise ValueError('Guided actor context exceeds declared limit')
        started = time.monotonic()
        response = self.client.session.post(
            self.plan['actor_url'] + '/v1/completions', timeout=1800,
            json=dict(model='frozen-actor', prompt=rendered,
                      seed=random_seed % 2**32, max_tokens=tokens,
                      temperature=self.plan['actor_temperature'], top_p=1.,
                      top_k=-1, repetition_penalty=1.,
                      add_special_tokens=False, guided_choice=commands))
        response.raise_for_status()
        payload = response.json()
        if payload['usage']['prompt_tokens'] != count:
            raise ValueError('Guided actor server/client tokenizer mismatch')
        choice = payload['choices'][0]
        command = choice['text'].strip()
        if command not in commands:
            raise ValueError('Guided server returned an unadvertised command')
        return dict(text=command, finish_reason=choice['finish_reason'],
                    usage=dict(prompt_tokens=count,
                               completion_tokens=payload['usage']['completion_tokens']),
                    seed=random_seed, prompt_sha256=hashlib.sha256(json.dumps(
                        messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                    rendered_prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                    guided_choice_sha256=hashlib.sha256(json.dumps(
                        commands, ensure_ascii=False).encode()).hexdigest(),
                    guided_choice_count=len(commands),
                    seconds=time.monotonic()-started)

    def run_many(self, jobs):
        results = super().run_many(jobs)
        for job, episode in zip(jobs, results):
            episode['actor_adapter_enabled'] = True
            episode['guided_choice_decoder'] = True
            if any(not step['valid_command'] for step in episode['trajectory']):
                raise ValueError('Guided actor produced an invalid environment command')
            save(Path(job['output']) / 'episode.json', episode)
        return results
