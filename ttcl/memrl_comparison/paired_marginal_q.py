"""Sampled paired-marginal Q update for native ALFWorld MemRL training.

Only the first retrieved memory on the first attempt receives a causal
counterfactual Q target. Other retrieved entries keep the native update.
The counterfactual actor run is read-only: it never writes memory or Q.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import traceback

from ttcl.icl_mem0_comparison.protocol import save, sha

from .memory import Memory, digest
from .worker import alf_query


COST_WEIGHT = 0.1
CALL_BUDGET = 50


def paired_advantage(full: dict, drop: dict) -> dict:
    if (full['status'] != 'complete' or drop['status'] != 'complete' or
            full['game'] != drop['game'] or full['seed'] != drop['seed']):
        raise ValueError('Counterfactual arms are not paired complete runs')
    full_calls, drop_calls = len(full['generations']), len(drop['generations'])
    if not 0 <= full_calls <= CALL_BUDGET or not 0 <= drop_calls <= CALL_BUDGET:
        raise ValueError('Actor call budget exceeded')
    raw = float(full['reward']) - float(drop['reward'])
    cost = COST_WEIGHT * (full_calls - drop_calls) / CALL_BUDGET
    value = max(-1.0, min(1.0, raw - cost))
    if not math.isfinite(value):
        raise ValueError('Nonfinite paired advantage')
    return dict(raw_reward_delta=raw, actor_call_delta=full_calls-drop_calls,
                cost_weight=COST_WEIGHT, call_budget=CALL_BUDGET,
                cost_adjustment=cost, q_advantage=value)


class PairedQMemory(Memory):
    def update_with_credit(self, query, public_trace, reward, success,
                           retrieval, binding, selected_id=None,
                           advantage=None):
        if (not isinstance(reward, (int, float)) or
                not math.isfinite(reward) or retrieval['query'] != query or
                (selected_id is None) != (advantage is None) or
                (selected_id is not None and selected_id not in retrieval['ids'])):
            raise ValueError('Invalid official or causal Q update')
        q_updates, q_rewards = {}, {}
        before = len(self.errors)
        for mid in retrieval['ids']:
            credit = float(advantage if mid == selected_id else reward)
            new = self.service.update_value(mid, credit)
            if new is None:
                raise RuntimeError('Upstream swallowed a Q update error')
            self.service._q_cache[mid] = new
            q_updates[mid], q_rewards[mid] = new, credit
        self.abstracts = []
        results = self.service.add_memories(
            [query], [public_trace],
            [success if success is not None else True],
            retrieved_memory_queries=[retrieval['similarities']],
            retrieved_memory_ids_list=[retrieval['ids']],
            metadatas=[{'success':success, 'official_reward':reward,
                        'input_binding':binding,
                        'public_trace_sha256':hashlib.sha256(
                            public_trace.encode()).hexdigest()}])
        if (len(self.errors) != before or not results or len(results) != 1 or
                not results[0][1] or len(self.abstracts) != 1):
            raise RuntimeError(f'MemRL memory write failed: {self.errors[before:]}')
        mid = results[0][1]
        item = self.store.get(mid).model_dump()
        item['metadata']['public_abstract'] = self.abstracts[0]
        item['metadata']['writer_token_limit_hit'] = self.last_limit_hit
        self.store.update(mid, item)
        if any(mid not in self.store.items for group in
               self.service.dict_memory.values() for mid in group):
            raise RuntimeError('Invalid upstream query index')
        return dict(new_memory_id=mid, q_updates=q_updates,
                    q_rewards=q_rewards, causal_memory_id=selected_id,
                    input_binding=binding)


def _drop_first(memory: Memory, retrieval: dict) -> tuple[str, str, str]:
    ids = retrieval['ids']
    if not ids:
        raise ValueError('No experience selected for paired Q')
    texts = {}
    for mid in ids:
        metadata = memory.store.get(mid).metadata.model_dump()
        texts[mid] = ('Task: ' + metadata['task_description'] +
                      '\nExperience: ' + metadata['public_abstract'])
    full = '\n\n'.join(texts[mid] for mid in ids)
    if full != retrieval['context']:
        raise ValueError('Native retrieval context changed')
    selected = ids[0]
    drop = '\n\n'.join(texts[mid] for mid in ids[1:])
    return selected, hashlib.sha256(texts[selected].encode()).hexdigest(), drop


def alf_cell_paired_q(plan: dict, client, memory: PairedQMemory, item: dict,
                      repeat: int, output: Path) -> dict:
    from ttcl.alfworld_comparison.environment import Actor, make_env
    from ttcl.alfworld_comparison.methods import public_episode
    from ttcl.experience_evolution.core import seed

    class LocalActor(Actor):
        def generate(self, messages, random_seed):
            result = client.complete(messages, random_seed,
                                     tokens=plan['alf']['actor_max_tokens'],
                                     temperature=plan['alf']['actor_temperature'],
                                     top_p=1.)
            return dict(text=result['raw_response'],
                        finish_reason=result['finish_reason'],
                        usage={'prompt_tokens':result['input_tokens'],
                               'completion_tokens':result['output_tokens']},
                        seed=random_seed, prompt_sha256=digest(messages),
                        seconds=result['seconds'],
                        rendered_prompt_sha256=result['rendered_prompt_sha256'])

    output.mkdir(parents=True, exist_ok=False)
    before = (memory.calls, memory.input_tokens, memory.output_tokens)
    limits_before = memory.limit_hits
    game = Path(plan['alf']['data_root']) / item['path']
    if sha(game) != item['sha256']:
        raise ValueError('ALFWorld input content changed')
    env = make_env(game)
    try:
        observation = str(env.reset()['feedback'])
    finally:
        env.close()
    query = alf_query(observation)
    actor = LocalActor(plan['alf'])
    episodes = []
    counterfactual_calls = 0
    row = dict(benchmark='alfworld', task=item['family'], repeat=repeat,
               arm='paired_q', game=item['path'],
               input_sha256=item['sha256'], status='failed', reward=None)
    try:
        for attempt in range(plan['alf']['max_attempts']):
            retrieval = memory.retrieve(query)
            save(output / f'retrieval_{attempt+1}.json', retrieval)
            actor_seed = seed(repeat, item['path'], attempt, 'actor')
            episode = actor.run_many([dict(
                game=item['path'], memory=retrieval['context'], seed=actor_seed,
                output=output / f'attempt_{attempt+1}')])[0]
            if episode['initial_observation'] != observation:
                raise ValueError('Task reset observation changed')
            episodes.append(episode)
            selected_id, advantage = None, None
            if attempt == 0 and retrieval['ids']:
                selected_id, text_hash, drop_context = _drop_first(
                    memory, retrieval)
                drop = actor.run_many([dict(
                    game=item['path'], memory=drop_context, seed=actor_seed,
                    output=output / 'counterfactual_1')])[0]
                if drop['initial_observation'] != observation:
                    raise ValueError('Counterfactual task reset changed')
                counterfactual_calls += len(drop['generations'])
                credit = paired_advantage(episode, drop)
                advantage = credit['q_advantage']
                save(output / 'counterfactual_1.json',
                     dict(selected_memory_id=selected_id,
                          selected_memory_text_sha256=text_hash,
                          full_context_sha256=hashlib.sha256(
                              retrieval['context'].encode()).hexdigest(),
                          drop_context_sha256=hashlib.sha256(
                              drop_context.encode()).hexdigest(),
                          full_episode_sha256=sha(output / 'attempt_1' /
                                                  'episode.json'),
                          drop_episode_sha256=sha(output / 'counterfactual_1' /
                                                  'episode.json'),
                          actor_seed=actor_seed, **credit))
            public = public_episode(episode)
            trace = json.dumps(public, ensure_ascii=False)
            reward = 1. if episode['reward'] else -1.
            update = memory.update_with_credit(
                query, trace, reward, bool(episode['reward']), retrieval,
                {'game_sha256':item['sha256'],
                 'public_content_sha256':digest(public),
                 'attempt':attempt, 'repeat':repeat},
                selected_id=selected_id, advantage=advantage)
            save(output / f'update_{attempt+1}.json', update)
            if episode['reward']:
                break
        row.update(status='complete', reward=max(ep['reward'] for ep in episodes),
                   first_attempt=episodes[0]['reward'],
                   within_three=max(ep['reward'] for ep in episodes))
    except Exception as exc:
        row.update(error=repr(exc), traceback=traceback.format_exc())
    finally:
        actor.pool.shutdown(wait=True)
    after = (memory.calls, memory.input_tokens, memory.output_tokens)
    generations = [g for ep in episodes for g in ep['generations']]
    row.update(attempts=len(episodes), actor_calls=len(generations),
               counterfactual_actor_calls=counterfactual_calls,
               total_actor_calls=len(generations)+counterfactual_calls,
               actor_input_tokens=sum(g['usage']['prompt_tokens']
                                      for g in generations),
               actor_output_tokens=sum(g['usage']['completion_tokens']
                                       for g in generations),
               writer_calls=after[0]-before[0],
               writer_input_tokens=after[1]-before[1],
               writer_output_tokens=after[2]-before[2],
               writer_token_limit_hits=memory.limit_hits-limits_before,
               initial_query_sha256=hashlib.sha256(
                   observation.encode()).hexdigest(),
               first_prompt_sha256=generations[0]['rendered_prompt_sha256']
               if generations else None)
    memory.snapshot(output / 'memory_after.json')
    row['memory_after_sha256'] = sha(output / 'memory_after.json')
    save(output / 'row.json', row)
    return row
