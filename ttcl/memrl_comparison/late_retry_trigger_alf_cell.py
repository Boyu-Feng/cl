"""Official ALFWorld cell with final-attempt invalid-rate guidance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import traceback

from ttcl.alfworld_comparison.environment import make_env
from ttcl.alfworld_comparison.methods import public_episode
from ttcl.experience_evolution.core import seed
from ttcl.icl_mem0_comparison.protocol import save, sha
from .retry_trigger_actor import RetryTriggeredActor
from .late_retry_trigger import should_guide_final_retry
from .memory import digest
from .worker import alf_query


def late_retry_trigger_alf_cell(plan, client, memory, item, repeat, arm, output):
    output.mkdir(parents=True, exist_ok=False)
    before = (memory.calls, memory.input_tokens, memory.output_tokens)
    limits_before = memory.limit_hits
    game = Path(plan['alf']['data_root']) / item['path']
    if sha(game) != item['sha256']:
        raise ValueError('Official ALFWorld input changed')
    environment = make_env(game)
    try:
        observation = str(environment.reset()['feedback'])
    finally:
        environment.close()
    query = alf_query(observation)
    actor = RetryTriggeredActor(plan['alf'], client)
    episodes = []
    row = dict(benchmark='alfworld', task=item['family'], repeat=repeat,
               arm=arm, game=item['path'], input_sha256=item['sha256'],
               status='failed', reward=None, retry_trigger_decoder=True)
    try:
        for attempt in range(plan['alf']['max_attempts']):
            actor.guide_retry = should_guide_final_retry(episodes)
            retrieval = memory.retrieve(query)
            save(output / f'retrieval_{attempt+1}.json', retrieval)
            episode = actor.run_many([dict(game=item['path'],
                memory=retrieval['context'],
                seed=seed(repeat, item['path'], attempt, 'actor'),
                output=output / f'attempt_{attempt+1}')])[0]
            if episode['initial_observation'] != observation:
                raise ValueError('Task reset observation changed')
            episode['retry_triggered'] = actor.guide_retry
            episode['prior_invalid_rate'] = (sum(not step['valid_command'] for step in
                episodes[-1]['trajectory']) / len(episodes[-1]['trajectory'])
                if episodes else None)
            save(output / f'attempt_{attempt+1}' / 'episode.json', episode)
            episodes.append(episode)
            public = public_episode(episode)
            trace = json.dumps(public, ensure_ascii=False)
            update = memory.update(query, trace,
                1. if episode['reward'] else -1., bool(episode['reward']),
                retrieval, dict(game_sha256=item['sha256'],
                                public_content_sha256=digest(public),
                                attempt=attempt, repeat=repeat))
            save(output / f'update_{attempt+1}.json', update)
            if episode['reward']:
                break
        row.update(status='complete',
                   reward=max(episode['reward'] for episode in episodes),
                   first_attempt=episodes[0]['reward'],
                   within_three=max(episode['reward'] for episode in episodes))
    except Exception as exc:
        row.update(error=repr(exc), traceback=traceback.format_exc())
    finally:
        actor.pool.shutdown(wait=True)
    after = (memory.calls, memory.input_tokens, memory.output_tokens)
    generations = [generation for episode in episodes
                   for generation in episode['generations']]
    row.update(attempts=len(episodes), actor_calls=len(generations),
               guided_attempts=sum(e['retry_triggered'] for e in episodes),
               guided_choice_calls=sum(g['guided_choice_used'] for g in generations),
               actor_input_tokens=sum(g['usage']['prompt_tokens'] for g in generations),
               actor_output_tokens=sum(g['usage']['completion_tokens'] for g in generations),
               writer_calls=after[0]-before[0],
               writer_input_tokens=after[1]-before[1],
               writer_output_tokens=after[2]-before[2],
               writer_token_limit_hits=memory.limit_hits-limits_before,
               initial_query_sha256=hashlib.sha256(observation.encode()).hexdigest(),
               first_prompt_sha256=(generations[0]['rendered_prompt_sha256']
                                    if generations else None))
    memory.snapshot(output / 'memory_after.json')
    row['memory_after_sha256'] = sha(output / 'memory_after.json')
    save(output / 'row.json', row)
    return row
