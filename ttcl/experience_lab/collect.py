"""Real environment outcomes, counterfactual probes, and auditable preferences."""
from __future__ import annotations

import copy
from pathlib import Path
import os
from types import SimpleNamespace
import time

from ttcl.experience_evolution.core import read, save, seed, writer_messages
from ttcl.experience_evolution.clbench import writer_messages as cl_messages
from ttcl.experience_v2.common import BENCH, Client, sha_file
from ttcl.alfworld_comparison.environment import Actor
from .protocol import binding, EVIDENCE_SUFFIX, select_pair
from .prepare import TOTALS


class Environments:
    def __init__(self, plan):
        from ttcl.experience_feedback.evaluate import configure_tasks
        configure_tasks()
        os.chdir(BENCH)
        self.plan = plan
        self.alf = Actor(plan)
        self.client = Client(plan['actor_url'], context=plan['context'])

    def run_many(self, jobs):
        """Branch-independent seeds, fresh resets, explicit physical-call reuse."""
        results = []
        for offset in range(0, len(jobs), 16):
            batch = jobs[offset:offset+16]
            pending, identities = [], {}
            for spec, memory, repeat, dest in batch:
                dest = Path(dest)
                request = {'spec': spec, 'memory': memory, 'seed': repeat}
                if (dest/'completed.json').exists():
                    if read(dest/'request.json') != request:
                        raise ValueError('Resumed environment input changed')
                    continue
                if dest.exists():
                    dest.rename(dest.with_name(dest.name+f'.interrupted_{time.time_ns()}'))
                key = binding(request)
                if key in identities:
                    continue
                identities[key] = dest
                if spec['domain'] == 'alfworld':
                    pending.append({'game': spec['game'], 'memory': memory, 'seed': repeat,
                                    'output': str(dest)})
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
                        raise RuntimeError(f'Infrastructure failure: {row.get("error")}')
                    save(dest/'completed.json', {'reward': row['reward'], 'episode': ep, 'row': row,
                         'task_identity': [spec['domain'], spec['index'], row['instance_id']],
                         'initial_query_hash': binding(ep['initial_public_query'])})
                    save(dest/'request.json', request)
            if pending:
                episodes = self.alf.run_many(pending)
                for job, ep in zip(pending, episodes):
                    dest = Path(job['output'])
                    save(dest/'completed.json', {'reward': ep['reward'], 'episode': ep,
                        'row': {'status': 'complete', 'reward': ep['reward'], 'steps': ep['steps']},
                        'task_identity': ['alfworld', ep['game']],
                        'initial_query_hash': binding(ep['initial_observation'])})
            for spec, memory, repeat, dest in batch:
                dest = Path(dest)
                request = {'spec': spec, 'memory': memory, 'seed': repeat}
                if not (dest/'completed.json').exists():
                    original = identities[binding(request)]
                    value = dict(read(original/'completed.json'), reused_from=str(original))
                    save(dest/'completed.json', value)
                if not (dest/'request.json').exists():
                    save(dest/'request.json', request)
                results.append(read(dest/'completed.json'))
        return results

    def close(self):
        self.alf.pool.shutdown(wait=True)
        self.client.session.close()


def messages_for(memory, spec, result):
    return (writer_messages if spec['domain']=='alfworld' else cl_messages)(memory, result['episode'])


def generate(client, messages, model, path, random_seed, plan, temperature=0., evidence=False):
    path = Path(path)
    sampling_messages = copy.deepcopy(messages)
    if evidence:
        sampling_messages[0]['content'] += EVIDENCE_SUFFIX
    request = {'messages': sampling_messages, 'model': model, 'seed': random_seed,
               'temperature': temperature, 'tokens': plan['writer_tokens']}
    if path.exists():
        result = read(path)
        if result['request'] != request:
            raise ValueError('Cached writer input differs')
        return result
    result = client.complete(sampling_messages, model=model, random_seed=random_seed,
                             tokens=plan['writer_tokens'], temperature=temperature, top_p=1.)
    result.update(request=request, training_messages=messages, input_binding=binding(messages))
    result['usable'] = bool(result['raw_response']) and result['finish_reason']=='stop'
    if len(client.tokenizer.encode(result['raw_response'], add_special_tokens=False)) > plan['memory_tokens']:
        result['usable'] = False
    save(path, result)
    return result


def shard_for(history):
    return history.get('family', history['domain'])


def collect_shard(root, round_id, shard):
    root = Path(root)
    plan = read(root/'plan.json')
    rp = next(r for r in plan['rounds'] if r['id']==round_id)
    directory = root/round_id
    actor = Environments(plan)
    client = actor.client
    histories = [h for h in rp['histories'] if shard_for(h)==shard]
    progress = directory/'workers'/f'{shard}.json'
    try:
        for position, history in enumerate(histories):
            dest = directory/'histories'/history['id']
            if (dest/'selection.json').exists():
                continue
            save(progress, {'phase': 'collecting', 'history': history['id'],
                            'completed': position, 'expected': len(histories), 'time': time.time()})
            warmup = actor.run_many([(history['warmup'], '', history['warmup_seed'], dest/'warmup')])[0]
            previous = ''
            if warmup['reward'] is not None:
                generated = generate(client, messages_for('', history['warmup'], warmup), 'parent',
                    dest/'previous_writer.json', seed(history['id'], 'previous'), plan)
                if generated['usable']: previous = generated['raw_response']
            source = actor.run_many([(history['source'], previous, history['source_seed'], dest/'source')])[0]
            messages = messages_for(previous, history['source'], source)
            save(dest/'input.json', {'history': history, 'messages': messages, 'previous': previous,
                                    'input_binding': binding(messages)})
            texts = {'empty': '', 'keep': previous}
            for name, model, temp, evidence in [('parent_greedy','parent',0.,False),
                    ('parent_sample','parent',1.,False), ('base_evidence','frozen-actor',.7,True)]:
                result = generate(client, messages, model, dest/'candidates'/f'{name}.json',
                                  seed(history['id'], name), plan, temp, evidence)
                if result['usable']: texts[name] = result['raw_response']
            save(dest/'texts.json', texts)
            jobs, keys = [], []
            for repeat in rp['probe_seeds']:
                for ti, spec in enumerate(history['probes']):
                    for branch, text in texts.items():
                        jobs.append((spec, text, seed(repeat, history['id'], ti),
                                     dest/'probes'/str(repeat)/str(ti)/branch))
                        keys.append((str(repeat), ti, branch))
            results = actor.run_many(jobs)
            rewards = {branch: {str(s): [] for s in rp['probe_seeds']} for branch in texts}
            identities = {}
            for result, (repeat, ti, branch) in zip(results, keys):
                identity = [result['task_identity'], result['initial_query_hash']]
                if (repeat, ti) in identities and identities[(repeat, ti)] != identity:
                    raise ValueError('Counterfactual environments do not match')
                identities[(repeat, ti)] = identity
                rewards[branch][repeat].append(result['reward'])
            selection = select_pair(texts, rewards, *rp['probe_seeds'])
            selection.update(input_binding=binding(messages), history=history['id'],
                             domain=history['domain'], rewards=rewards,
                             supervision='automatic paired official environment outcomes; no human target or test feedback')
            save(dest/'selection.json', selection)
            save(progress, {'phase': 'collecting', 'completed': position+1,
                            'expected': len(histories), 'accepted': selection['accepted'], 'time': time.time()})
        save(progress, {'phase': 'complete', 'completed': len(histories), 'expected': len(histories)})
    finally:
        actor.close()


def build_dataset(root, round_id, tokenizer):
    """Review automatic label integrity and bind targets to exact public inputs."""
    from ttcl.experience_repair.run import encode_training
    root = Path(root)
    plan = read(root/'plan.json')
    rows, excluded, provenance = [], [], {}
    for rp in plan['rounds']:
        for history in rp['histories']:
            dest = root/rp['id']/'histories'/history['id']
            if not (dest/'selection.json').exists():
                raise ValueError(f'Collection incomplete: {dest}')
            selection = read(dest/'selection.json')
            if not selection['accepted']:
                excluded.append({'id': history['id'], 'reason': selection['reason']})
                continue
            inputs, texts = read(dest/'input.json'), read(dest/'texts.json')
            if inputs['input_binding'] != binding(inputs['messages']) or selection['input_binding'] != inputs['input_binding']:
                raise ValueError('Automatic target input binding failed')
            chosen, rejected = (texts[selection[k]] for k in ['chosen', 'rejected'])
            random_options = [texts[k] for k in plan['candidate_methods'] if k in texts]
            import random
            random_target = random.Random(seed(92471, history['id'], 'control')).choice(random_options)
            row = {'id': history['id'], 'domain': history['domain'], 'messages': inputs['messages'],
                   'chosen': chosen, 'rejected': rejected, 'random_target': random_target,
                   'input_binding': inputs['input_binding'], 'selection_path': str(dest/'selection.json')}
            try:
                for target in [chosen, rejected, random_target]:
                    encode_training(tokenizer, {'id': row['id'], 'messages': row['messages'], 'target': target}, plan['training']['max_length'])
            except ValueError as exc:
                excluded.append({'id': row['id'], 'reason': 'training_context_budget', 'detail': str(exc)})
                continue
            rows.append(row)
            for name in ['input.json', 'selection.json', 'texts.json']:
                p = dest/name; provenance[str(p)] = sha_file(p)
        if rp['id']==round_id: break
    save(root/round_id/'dataset.json', rows)
    save(root/round_id/'dataset_audit.json', {'accepted': len(rows), 'excluded': excluded,
        'automatic_labels_recomputed_each_round': True, 'human_annotations_reused': False,
        'provenance_hashes': provenance, 'dataset_sha256': sha_file(root/round_id/'dataset.json'),
        'by_domain': {d: sum(r['domain']==d for r in rows) for d in sorted({r['domain'] for r in rows})}})
    return rows
