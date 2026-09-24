"""Official tasks with explicit actor routing and reward-independent trace sampling."""
from __future__ import annotations

import os
from pathlib import Path
import random
from types import SimpleNamespace

from ttcl.alfworld_comparison.environment import Actor as SafeActor
from ttcl.experience_evolution.core import read, save, seed
from ttcl.experience_lab.collect import messages_for
from ttcl.experience_lab.prepare import TOTALS
from ttcl.experience_v2.common import BENCH
from .client import Client
from .protocol import binding


class Actor(SafeActor):
    def __init__(self, plan, client):
        super().__init__(plan)
        self.client = client

    def generate(self, messages, random_seed):
        result = self.client.complete(messages, self.client.actor_model, random_seed,
            self.plan['actor_max_tokens'], 1. if self.client.training else .7, 1.,
            capture=self.client.capture)
        if 'sample' in result:
            self.client.records.append(result.pop('sample'))
        return {'text': result['raw_response'], 'finish_reason': result['finish_reason'],
                'usage': {'prompt_tokens': result['input_tokens'], 'completion_tokens': result['output_tokens']},
                'seed': random_seed, 'seconds': result['seconds']}


class Environments:
    def __init__(self, plan, actor_model='frozen-actor', training=False, capture=False):
        from ttcl.experience_feedback.evaluate import configure_tasks
        configure_tasks(); os.chdir(BENCH)
        self.plan = plan
        self.client = Client(plan, actor_model, training, capture)
        self.alf = Actor(plan, self.client)

    def run(self, spec, memory, repeat, dest):
        dest = Path(dest)
        request = {'spec': spec, 'memory': memory, 'seed': repeat,
                   'actor_model': self.client.actor_model, 'training_sampling': self.client.training,
                   'capture': self.client.capture}
        if (dest/'completed.json').exists():
            if read(dest/'request.json') != request:
                raise ValueError('Cached environment request differs')
            return read(dest/'completed.json')
        if dest.exists():
            raise FileExistsError('Partial episodes require a new run or explicit recovery: '+str(dest))
        self.client.records = []
        if spec['domain'] == 'alfworld':
            ep = self.alf.run_many([{'game': spec['game'], 'memory': memory, 'seed': repeat,
                                     'output': str(dest)}])[0]
            ep['actor_adapter_enabled'] = self.client.actor_model != 'frozen-actor'
            ep['actor_model'] = self.client.actor_model
            save(dest/'episode.json', ep)
            row = {'status': 'complete', 'reward': ep['reward'], 'steps': ep['steps'],
                   'actor_calls': len(ep['generations']),
                   'actor_input_tokens': sum(x['usage']['prompt_tokens'] for x in ep['generations']),
                   'actor_output_tokens': sum(x['usage']['completion_tokens'] for x in ep['generations'])}
            identity = ['alfworld', ep['game']]; initial = ep['initial_observation']
        else:
            from ttcl.structured_memory.online_bank import run_episode
            args = SimpleNamespace(task=spec['domain'], seed=self.plan['environment_seed'],
                num_instances=TOTALS[spec['domain']], memory_chars=20000,
                max_turns_per_instance=64, action_retries=2, normalize_action=True,
                allow_initial_experience=True)
            self.client.repeat = repeat
            row, ep = run_episode(args, self.client, spec['index'], dest, memory)
            if row['status'] != 'complete' and not any(x in row.get('error','') for x in
                    ['no schema-valid JSON action', 'Safety cap exceeded']):
                raise RuntimeError('Infrastructure failure: '+row.get('error',''))
            identity = [spec['domain'], spec['index'], row['instance_id']]
            initial = ep['initial_public_query']
        # Uniform action subsampling is independent of actions, reward and length
        # eligibility. A long sampled action is excluded explicitly, never replaced
        # by a short successful one. This estimates a mean-over-actions objective.
        rng = random.Random(seed(repeat, spec['id'], 'reader_turns'))
        indices = sorted(rng.sample(range(len(self.client.records)),
                        min(self.plan['training']['reader_turns_per_episode'], len(self.client.records))))
        selected = [dict(self.client.records[i], action_index=i) for i in indices]
        value = {'reward': row['reward'], 'episode': ep, 'row': row,
                 'task_identity': identity, 'initial_query_hash': binding(initial),
                 'actor_model': self.client.actor_model, 'sampled_actions': selected,
                 'total_generated_actions': len(self.client.records)}
        save(dest/'request.json', request); save(dest/'completed.json', value)
        self.client.records = []
        return value

    def close(self):
        self.alf.pool.shutdown(wait=True); self.client.session.close()


def writer(client, messages, model, path, repeat, plan, sample=False):
    path = Path(path)
    if path.exists():
        result = read(path)
        if result['input_binding'] != binding(messages) or result['model'] != model or result['seed'] != repeat:
            raise ValueError('Cached writer input differs')
        return result
    result = client.complete(messages, model, repeat, plan['writer_tokens'],
                             1. if sample else 0., 1., capture=sample)
    result.update(input_binding=binding(messages), model=model, seed=repeat)
    result['usable'] = bool(result['raw_response']) and result['finish_reason'] == 'stop'
    result['usable'] &= len(client.tokenizer.encode(result['raw_response'], add_special_tokens=False)) <= plan['memory_tokens']
    save(path, result)
    return result
