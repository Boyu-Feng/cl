"""Fork one reviewed public ALFWorld prefix with versus without memory.

The original no-memory prefix is replayed exactly, without actor calls. Each
continuation starts from identical environment state, chat history and action
budget. Prefixes were selected after earlier results, so this is a mechanism
test, not an unbiased benchmark or online learning experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.alfworld_comparison.environment import make_env
from ttcl.experience_evolution.core import digest, seed
from ttcl.experience_evolution.environment import ACTOR_SYSTEM, clean_command
from ttcl.icl_mem0_comparison.protocol import Client, read, save, sha


REPEATS = (94101, 94102, 94103)
ARMS = ('with_memory', 'without_memory', 'without_repeat')
SELECTORS = ((0, 94001), (1, 94002))


def design_for(stage: Path, review_path: Path, output: Path, url: str) -> dict:
    source = read(stage / 'design.json')
    review = read(review_path)
    if (source['schema'] != 'alf_delayed_memory_activation_v1' or
            review['schema'] != 'alf_stage_prefix_reviews_v1' or
            len(review['targets']) != len(SELECTORS)):
        raise ValueError('Wrong source stage or prefix review')
    cases = []
    for case_id, original_repeat in SELECTORS:
        case = source['cases'][case_id]
        origin = Path(case['origin'])
        prior_path = (stage / f'case_{case_id}' /
                      f'repeat_{original_repeat}' / 'delayed' / 'episode.json')
        prior = read(prior_path)
        prefix = prior['activation_turn'] - 1
        if (prefix <= 0 or prefix >= 50 or
                any(step['memory_active'] for step in prior['trajectory'][:prefix]) or
                not prior['trajectory'][prefix-1]['trigger_after_action']):
            raise ValueError('Prefix does not end at first public object evidence')
        actions = [step['action'] for step in prior['trajectory'][:prefix]]
        last = prior['trajectory'][prefix-1]['observation']
        matched = [row for row in review['targets']
                   if row['case'] == case['case'] and row['reviewed'] and
                   row['public_task'] == case['public_task'] and
                   row['source_input_sha256'] == case['game_sha256'] and
                   row['goal_object'] == case['goal_object'] and
                   row['source_episode_sha256'] == sha(prior_path) and
                   row['prefix_actions_sha256'] == digest(json.dumps(
                       actions, ensure_ascii=False)) and
                   row['prefix_last_observation_sha256'] == hashlib.sha256(
                       last.encode()).hexdigest() and
                   row['prefix_steps'] == prefix]
        if len(matched) != 1:
            raise ValueError('Missing reviewed, source-bound public prefix')
        plan = read(origin / 'plan.json')
        if (sha(origin / 'plan.json') != case['source_plan_sha256'] or
                sha(Path(plan['alf']['data_root']) / case['game']) !=
                case['game_sha256'] or plan['alf']['max_steps'] != 50):
            raise ValueError('Original game or action budget changed')
        cases.append(dict(index=case_id, origin=str(origin),
                          game=case['game'], game_sha256=case['game_sha256'],
                          public_task=case['public_task'],
                          goal_object=case['goal_object'],
                          memory_context=case['memory_context'],
                          memory_text_sha256=case['memory_text_sha256'],
                          source_plan_sha256=case['source_plan_sha256'],
                          source_stage_episode=str(prior_path),
                          source_stage_episode_sha256=sha(prior_path),
                          source_prefix_repeat=original_repeat,
                          prefix_length=prefix,
                          prefix_actions_sha256=digest(json.dumps(
                              actions, ensure_ascii=False)),
                          last_prefix_observation_sha256=hashlib.sha256(
                              last.encode()).hexdigest()))
    return dict(schema='alf_memory_at_evidence_prefix_v1',
                stage_design_sha256=sha(stage / 'design.json'),
                reviews_sha256=sha(review_path),
                runner_sha256=sha(Path(__file__)),
                environment_adapter_sha256=sha(Path(__file__).parents[1] /
                                                 'alfworld_comparison' /
                                                 'environment.py'),
                cases=cases, repeats=list(REPEATS), arms=list(ARMS),
                url=url,
                budget='Two previously reviewed official train game prefixes; three new continuation seeds; three arms; original 50-step budget including prefix; no actor calls during prefix replay',
                caveat='Posthoc selected prefixes; fixed public history isolates suffix memory intervention but does not estimate a general online policy')


def replay_prefix(plan: dict, case: dict):
    original = read(Path(case['source_stage_episode']))
    if sha(Path(case['source_stage_episode'])) != case['source_stage_episode_sha256']:
        raise ValueError('Source stage episode changed')
    game = Path(plan['alf']['data_root']) / case['game']
    if sha(game) != case['game_sha256']:
        raise ValueError('Official game changed')
    env = make_env(game)
    try:
        state = env.reset()
        initial = (str(state['feedback']),
                   digest(json.dumps(list(state['admissible_commands']))))
        if initial != (original['initial_observation'],
                       original['initial_commands_sha256']):
            raise ValueError('Public reset differs from frozen prefix')
        messages = [{'role':'system', 'content':ACTOR_SYSTEM}]
        for turn in range(case['prefix_length']):
            step = original['trajectory'][turn]
            generation = original['generations'][turn]
            messages.append({'role':'user', 'content':str(state['feedback']) +
                             '\nAvailable commands:\n' +
                             '\n'.join(state['admissible_commands'])})
            if (digest(messages) != generation['prompt_sha256'] or
                    step['memory_active'] or
                    clean_command(generation['text'],
                                  state['admissible_commands']) != step['action']):
                raise ValueError('Prefix actor prompt or command changed')
            state, _, done = env.step(step['action'])
            if done or state['won'] or str(state['feedback']) != step['observation']:
                raise ValueError('Frozen prefix feedback changed or finished early')
            messages.append({'role':'assistant', 'content':generation['text']})
        if hashlib.sha256(str(state['feedback']).encode()).hexdigest() != \
                case['last_prefix_observation_sha256']:
            raise ValueError('Frozen public prefix endpoint changed')
        return env, state, messages, initial
    except Exception:
        env.close()
        raise


def run_suffix(plan: dict, client: Client, case: dict, repeat: int,
               arm: str, output: Path) -> dict:
    if arm not in ARMS or output.exists():
        raise ValueError('Invalid or existing suffix run')
    output.mkdir(parents=True)
    env, state, messages, initial = replay_prefix(plan, case)
    actor_seed = seed(repeat, case['game'], 0, 'actor')
    memory = case['memory_context'] if arm == 'with_memory' else ''
    if memory:
        messages[0]['content'] += '\n\nPast experience:\n' + memory
    episode = dict(game=case['game'], game_sha256=case['game_sha256'],
                   arm=arm, repeat=repeat, seed=actor_seed,
                   prefix_length=case['prefix_length'],
                   source_stage_episode_sha256=case['source_stage_episode_sha256'],
                   initial_observation=initial[0],
                   initial_commands_sha256=initial[1],
                   suffix_start_observation=str(state['feedback']),
                   suffix_start_observation_sha256=hashlib.sha256(
                       str(state['feedback']).encode()).hexdigest(),
                   memory=memory, memory_sha256=hashlib.sha256(
                       memory.encode()).hexdigest(),
                   trajectory=[], generations=[], reward=0.0,
                   total_steps=case['prefix_length'], status='failed')
    try:
        for turn in range(case['prefix_length'], plan['alf']['max_steps']):
            available = list(state['admissible_commands'])
            messages.append({'role':'user', 'content':str(state['feedback']) +
                             '\nAvailable commands:\n' + '\n'.join(available)})
            result = client.complete(messages, seed(actor_seed, turn),
                                     tokens=plan['alf']['actor_max_tokens'],
                                     temperature=plan['alf']['actor_temperature'],
                                     top_p=1.)
            generation = dict(text=result['raw_response'],
                              finish_reason=result['finish_reason'],
                              usage={'prompt_tokens':result['input_tokens'],
                                     'completion_tokens':result['output_tokens']},
                              seed=seed(actor_seed, turn),
                              prompt_sha256=digest(messages),
                              rendered_prompt_sha256=result['rendered_prompt_sha256'],
                              seconds=result['seconds'])
            command = clean_command(generation['text'], available)
            was_valid = command in available
            state, _, done = env.step(command)
            messages.append({'role':'assistant', 'content':generation['text']})
            episode['generations'].append(generation)
            episode['trajectory'].append(dict(action=command,
                                              observation=str(state['feedback']),
                                              valid_command=was_valid))
            episode.update(reward=float(bool(state['won'])),
                           total_steps=turn + 1)
            if done or state['won'] or turn + 1 == plan['alf']['max_steps']:
                episode['status'] = 'complete'
                episode['termination'] = ('success' if state['won'] else
                                          'budget_or_environment_done')
                break
        save(output / 'episode.json', episode)
        return episode
    finally:
        env.close()


def run(stage: Path, reviews: Path, output: Path,
        url: str, prepare_only: bool) -> None:
    design = design_for(stage, reviews, output, url)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'design.json'
    if path.exists():
        if read(path) != design:
            raise ValueError('Frozen evidence-prefix design changed')
    else:
        save(path, design)
    print(json.dumps(dict(design_sha256=sha(path),
                          prefixes=[case['prefix_length']
                                    for case in design['cases']])), flush=True)
    if prepare_only:
        return
    for case in design['cases']:
        plan = read(Path(case['origin']) / 'plan.json')
        plan['url'] = url
        plan['alf']['actor_url'] = url
        for repeat in REPEATS:
            for arm in ARMS:
                target = output / f"case_{case['index']}" / f'repeat_{repeat}' / arm
                if (target / 'episode.json').exists():
                    continue
                episode = run_suffix(plan, Client(plan, repeat), case,
                                     repeat, arm, target)
                print(json.dumps(dict(case=case['index'], repeat=repeat,
                                      arm=arm, reward=episode['reward'],
                                      total_steps=episode['total_steps'])),
                      flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18559')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run(args.stage.resolve(), args.reviews.resolve(), args.output.resolve(),
        args.url, args.prepare_only)


if __name__ == '__main__':
    main()
