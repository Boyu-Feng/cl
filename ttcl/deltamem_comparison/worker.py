"""Sequential benchmark ownership with explicit per-attempt state checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import requests

from ttcl.experience_evolution.core import read, save, seed

ARMS = ('base', 'deltamem_reset', 'deltamem_online')


class Client:
    def __init__(self, url, repeat=303):
        self.url, self.repeat = url, repeat
        self.http = requests.Session()
        self.http.trust_env = False

    def call(self, method, **kwargs):
        response = self.http.post(self.url+'/'+method, json=kwargs, timeout=3600)
        if response.status_code != 200:
            raise RuntimeError(response.text)
        return response.json()

    def generate(self, messages, random_seed):
        return self.call('generate', messages=messages, random_seed=seed(random_seed, self.repeat),
                         tokens=4096, temperature=.7, top_p=.9)


def previous_state(arm, attempt, incoming, previous_attempt):
    if arm == 'base':
        return None
    if attempt:
        return previous_attempt
    return incoming if arm == 'deltamem_online' else None


def archive_incomplete(path):
    if path.exists():
        path.rename(path.with_name(path.name+f'.interrupted_{time.time_ns()}'))


def check_request(path, request):
    if read(path/'request.json') != request:
        raise ValueError(f'Attempt lineage changed: {path}')
    audit = read(path/'memory_after.json')
    actual = hashlib.sha256((path/'state.pt').read_bytes()).hexdigest()
    if actual != audit['state_sha256']:
        raise ValueError(f'State hash mismatch: {path}')


def final_public_input(trial, row):
    """A poker hand may terminate before the actor is asked for any action."""
    responses_path = trial/'responses.jsonl'
    responses = [json.loads(line) for line in responses_path.read_text().splitlines()] if responses_path.exists() else []
    if not responses:
        if row.get('actor_calls') != 0 or row.get('status') != 'complete':
            raise ValueError('Missing actor log for an executed or incomplete episode')
        return '', []
    events_path = trial/'public_observations.jsonl'
    events = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
    final = responses[-1]
    action = json.dumps(final['action']) if final.get('packaging_repair') else final['raw_response']
    messages = final['messages'] + [{'role': 'assistant', 'content': action}]
    feedback = events[-1]['content'] if events and events[-1]['turn'] == final['turn'] else ''
    return feedback, messages


def clbench(root, client, plan):
    from ttcl.experience_feedback.evaluate import configure_tasks
    from ttcl.structured_memory import run_benchmark as base
    from ttcl.structured_memory.online_bank import run_episode
    configure_tasks()
    original = base.make_task

    def factory(name, random_seed, independent=False):
        if name == 'sales_prediction':
            from src.tasks.sales_prediction.task import SalesPredictionTask
            return SalesPredictionTask(seed=random_seed, schedule='default',
                                       clean_workspace_between_instances=independent)
        return original(name, random_seed, independent)

    base.make_task = factory
    os.chdir(base.BENCH)
    for domain, split in plan['clbench']['tasks'].items():
        if domain in ('sales_prediction', 'codebase_adaptation'):
            from .run import docker_status
            availability = docker_status()
            if not availability['available']:
                save(root/'workers'/f'{domain}.json', dict(phase='blocked', **availability))
                continue
        for repeat in plan['clbench']['repeats']:
            memories = dict.fromkeys(ARMS)
            args = SimpleNamespace(task=domain, seed=42, num_instances=split['total'],
                memory_chars=20000, max_turns_per_instance=64, action_retries=2,
                normalize_action=True, allow_initial_experience=True)
            for position, index in enumerate(split['test']):
                for arm in ARMS:
                    dest = root/'clbench'/domain/str(repeat)/arm/f'episode_{index+1:03}'
                    rows, last_state = [], None
                    for attempt in range(3):
                        inherited = previous_state(arm, attempt, memories[arm], last_state)
                        trial = dest/f'attempt_{attempt:02}'
                        request = dict(domain=domain, index=index, repeat=repeat,
                                       arm=arm, attempt=attempt, inherited_state=inherited)
                        save(root/'workers'/'clbench.json', dict(phase='running', **request,
                             updated_at=time.time()))
                        if (trial/'done.json').exists():
                            check_request(trial, request)
                            row = read(trial/'row.json')
                        else:
                            archive_incomplete(trial)
                            before = client.call('begin', arm=arm, previous=inherited)
                            client.repeat = repeat if attempt == 0 else seed(repeat, 'retry', attempt)
                            row, episode = run_episode(args, client, index, trial, '')
                            save(trial/'row.json', row)
                            save(trial/'request.json', request)
                            save(trial/'memory_before.json', before)
                            if row['status'] != 'complete' and not any(s in row.get('error', '') for s in
                                ['no schema-valid JSON action', 'Safety cap exceeded']):
                                raise RuntimeError(row.get('error', 'Unknown environment error'))
                            # Read ONLY the public observation stream. No hidden scalar,
                            # future sales labels or cohort population score enters memory.
                            feedback, final_messages = final_public_input(trial, row)
                            after = client.call('finish', destination=str(trial/'state.pt'),
                                                public_feedback=feedback, messages=final_messages)
                            save(trial/'memory_after.json', after)
                            save(trial/'done.json', {'status': 'recorded'})
                        rows.append(row)
                        last_state = str(trial/'state.pt') if arm != 'base' else None
                        if row.get('success') is True:
                            break
                    memories[arm] = last_state
                    record = dict(rows[-1], arm=arm, task=domain, repeat=repeat,
                                  canonical_index=index, position=position, attempts=len(rows),
                                  first_reward=rows[0]['reward'], first_success=rows[0].get('success'),
                                  attempt_rewards=[r['reward'] for r in rows],
                                  actor_calls=sum(r['actor_calls'] for r in rows),
                                  actor_input_tokens=sum(r['actor_input_tokens'] for r in rows),
                                  actor_output_tokens=sum(r['actor_output_tokens'] for r in rows))
                    save(dest/'result.json', record)
                    print(json.dumps(record), flush=True)
        save(root/'workers'/f'{domain}.json', {'phase': 'complete'})


def alfworld(root, client, plan):
    from ttcl.alfworld_comparison.environment import Actor
    settings = plan['alfworld']

    class LocalActor(Actor):
        def generate(self, messages, random_seed):
            value = client.call('generate', messages=messages, random_seed=random_seed,
                                tokens=64, temperature=.7, top_p=1.0)
            return dict(value, text=value['raw_response'], seed=random_seed,
                        usage={'prompt_tokens': value['input_tokens'],
                               'completion_tokens': value['output_tokens']})

    actor = LocalActor(settings)
    try:
        for sequence in settings['sequences']:
            repeat, family = sequence['repeat'], sequence['family']
            memories = dict.fromkeys(ARMS)
            for position, task in enumerate(sequence['tasks']):
                game = task['path']
                if hashlib.sha256((Path(settings['data_root'])/game).read_bytes()).hexdigest() != task['sha256']:
                    raise ValueError('ALFWorld task content changed')
                for arm in ARMS:
                    dest = root/'alfworld'/family/str(repeat)/f'task_{position:03}'/arm
                    episodes, last_state = [], None
                    for attempt in range(settings['max_attempts']):
                        inherited = previous_state(arm, attempt, memories[arm], last_state)
                        trial = dest/f'attempt_{attempt:02}'
                        request = dict(game=game, repeat=repeat, arm=arm, attempt=attempt,
                                       inherited_state=inherited)
                        save(root/'workers'/'alfworld.json', dict(phase='running', **request,
                             position=position, family=family, updated_at=time.time()))
                        if (trial/'done.json').exists():
                            check_request(trial, request)
                            episode = read(trial/'episode.json')
                        else:
                            archive_incomplete(trial)
                            before = client.call('begin', arm=arm, previous=inherited)
                            episode = actor.run_many([dict(game=game, memory='',
                                seed=seed(repeat, game, attempt, 'actor'), output=str(trial))])[0]
                            save(trial/'request.json', request)
                            save(trial/'memory_before.json', before)
                            # Final environment observation was produced after the last
                            # generated command; write it without any extra model call.
                            feedback = episode['trajectory'][-1]['observation']
                            feedback += '\nTask success: '+str(bool(episode['reward']))
                            after = client.call('finish', destination=str(trial/'state.pt'),
                                                public_feedback=feedback)
                            save(trial/'memory_after.json', after)
                            save(trial/'done.json', {'status': 'recorded'})
                        episodes.append(episode)
                        last_state = str(trial/'state.pt') if arm != 'base' else None
                        if episode['reward']:
                            break
                    memories[arm] = last_state
                    record = dict(status='complete', arm=arm, game=game, repeat=repeat,
                        family=family, position=position, first_success=int(episodes[0]['reward']),
                        success=int(episodes[-1]['reward']), attempts=len(episodes),
                        attempt_successes=[int(e['reward']) for e in episodes],
                        actor_calls=sum(e['steps'] for e in episodes),
                        actor_input_tokens=sum(g['input_tokens'] for e in episodes for g in e['generations']),
                        actor_output_tokens=sum(g['output_tokens'] for e in episodes for g in e['generations']),
                        invalid_commands=sum(not t['valid_command'] for e in episodes for t in e['trajectory']))
                    save(dest/'result.json', record)
                    print(json.dumps(record), flush=True)
    finally:
        actor.pool.shutdown(wait=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--suite', choices=['clbench', 'alfworld'], required=True)
    parser.add_argument('--url', required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        from .run import verify
        verify(root)
        globals()[args.suite](root, Client(args.url), read(root/'plan.json'))
        phase = 'complete'
        if args.suite == 'clbench' and any(read(p).get('phase') == 'blocked' for p in (root/'workers').glob('*.json')):
            phase = 'partial_blocked'
        save(root/'workers'/f'{args.suite}.json', {'phase': phase, 'updated_at': time.time()})
    except Exception as exc:
        save(root/'workers'/f'{args.suite}.json', {'phase': 'failed', 'error': repr(exc),
             'traceback': traceback.format_exc(), 'updated_at': time.time()})
        raise


if __name__ == '__main__':
    main()
