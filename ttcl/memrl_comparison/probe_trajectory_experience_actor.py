"""Stage-gated actor use of cross-task claims learned from completed traces."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

from ttcl.alfworld_comparison.environment import Actor, make_env
from ttcl.experience_evolution.core import digest, seed
from ttcl.experience_evolution.environment import clean_command
from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .feedback_skill_actor import public_state
from .probe_state_progress_actor import FAMILIES
from .trajectory_writer_learning import ClaimMemory, teacher_operations


REPEATS = (94901, 94902)
ARMS = ('native', 'online_claims', 'pretrained_frozen', 'pretrained_online')
QUOTAS = dict(look_at_obj_in_light=1, pick_and_place_simple=1,
              pick_clean_then_place_in_recep=2, pick_cool_then_place_in_recep=2,
              pick_heat_then_place_in_recep=2, pick_two_obj_and_place=1)


def learned_memory(dataset: dict) -> tuple[ClaimMemory, list[str]]:
    memory = ClaimMemory()
    sources = []
    seen = set()
    for row in dataset['rows']['train']:
        input_sha = row['input_sha256']
        if input_sha in seen:
            continue
        seen.add(input_sha)
        episode_path = Path(row['source_episode'])
        if sha(episode_path) != row['source_episode_sha256']:
            raise ValueError('Writer training source changed')
        episode = read(episode_path)
        if episode['game'] != row['game']:
            raise ValueError('Writer training game binding changed')
        result = memory.apply(teacher_operations(memory, episode), episode, input_sha)
        if result['rejected']:
            raise ValueError('Training claim evidence failed')
        sources.append(input_sha)
    return memory, sources


def relevant_experience(task: str, messages: list[dict], memory: ClaimMemory) -> tuple[str, dict]:
    try:
        state = public_state(task, messages)
    except ValueError:
        return '', dict(reason='unsupported_goal')
    selected = []
    kind, obj = state['kind'], state['target']
    if kind in ('clean', 'cool', 'heat') and any(
            held.startswith(obj + ' ') and held not in state['prepared'] for held in state['held']):
        selected.extend((key, value) for key, value in memory.entries.items()
                        if key.startswith(f'verified_preparation_tool:{kind}:'))
    if len(messages) >= 3 and messages[-2]['role'] == 'assistant':
        previous = messages[-3]['content'].split('\nAvailable commands:\n', 1)[1].splitlines()
        action = clean_command(messages[-2]['content'], previous).lower()
        feedback = messages[-1]['content'].split('\nAvailable commands:\n', 1)[0]
        if re.fullmatch(r'move \w+ \d+ to \w+ \d+', action) and 'Nothing happens.' in feedback:
            key = 'move_retry_after_navigation'
            if key in memory.entries:
                selected.append((key, memory.entries[key]))
    lines = [f"[{key}; {len(value['sources'])} independent earlier games] "
             f"When {value['claim']['when']}, {value['claim']['advice']}"
             for key, value in selected]
    return '\n'.join(lines), dict(reason='matched' if lines else 'no_supported_stage',
        state=state, claim_keys=[key for key, _ in selected])


class TrajectoryExperienceActor(Actor):
    def __init__(self, plan: dict, task: str, memory: ClaimMemory):
        super().__init__(plan)
        self.task, self.memory = task, memory

    def generate(self, messages: list[dict], random_seed: int) -> dict:
        text, decision = relevant_experience(self.task, messages, self.memory)
        prompt = [dict(message) for message in messages]
        if text:
            prompt[-1]['content'] += ('\n\nRelevant experience from completed earlier tasks '
                '(apply only if its stated condition matches this task):\n' + text)
        result = super().generate(prompt, random_seed)
        result['experience_decision'] = decision
        result['experience_text_sha256'] = digest(text)
        return result


def select_cases(data_root: Path, excluded: set[str]) -> list[dict]:
    root = data_root / 'json_2.1.1/train'
    cases = []
    for family in FAMILIES:
        paths = sorted(root.glob(f'{family}-*/trial_*/game.tw-pddl'),
            key=lambda path: hashlib.sha256(str(path.relative_to(root)).encode()).hexdigest())
        selected = 0
        for path in paths:
            input_sha = sha(path)
            if input_sha in excluded:
                continue
            environment = make_env(path)
            try:
                feedback = str(environment.reset()['feedback'])
            finally:
                environment.close()
            task = feedback.split('Your task is to: ', 1)[1].splitlines()[0]
            cases.append(dict(family=family, game=str(path.relative_to(data_root)),
                input_sha256=input_sha, public_task=task,
                initial_observation_sha256=hashlib.sha256(feedback.encode()).hexdigest(),
                reviewed=True))
            selected += 1
            if selected == QUOTAS[family]:
                break
        if selected != QUOTAS[family]:
            raise ValueError(f'Missing new official train games: {family}')
    return cases


def snapshot_memory(memory: ClaimMemory) -> dict:
    return {key: dict(claim=value['claim'], sources=sorted(value['sources']))
            for key, value in sorted(memory.entries.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--exclude-design', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18559')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    plan_path, dataset_path = args.plan.resolve(), args.dataset.resolve()
    excluded_paths = [path.resolve() for path in args.exclude_design]
    output = args.output.resolve()
    plan_source, dataset = read(plan_path), read(dataset_path)
    data_root = Path(plan_source['alf']['data_root']).resolve()
    memory, sources = learned_memory(dataset)
    excluded = set(sources)
    excluded.update(row['input_sha256'] for row in dataset['rows']['validation'])
    excluded.update(dataset['frozen_pilot_excluded_inputs'])
    for path in excluded_paths:
        excluded.update(case['input_sha256'] for case in read(path)['cases'])
    cases = select_cases(data_root, excluded)
    design = dict(schema='alf_trajectory_experience_online_pilot_v1', cases=cases,
        selection='SHA-ranked official train games, two each for clean/cool/heat and one each for look/simple/two, excluding full writer corpus and previous train design inputs, before rewards; family-blocked order',
        repeats=list(REPEATS), arms=list(ARMS),
        plan_path=str(plan_path), plan_sha256=sha(plan_path),
        dataset_path=str(dataset_path), dataset_sha256=sha(dataset_path),
        excluded_designs=[dict(path=str(path), sha256=sha(path)) for path in excluded_paths],
        source_game_sha256=sources, claim_memory=snapshot_memory(memory),
        claim_memory_text_sha256=digest(memory.experience_text()),
        runner_sha256=sha(Path(__file__)), url=args.url,
        budget='Nine source-disjoint official train games, two independent seed streams, four arms, max 50 environment actions and actor calls per episode; frozen actor, stage-conditioned experience text, update only after full episode',
        caveat='Online memory pilot with verified trajectory updates on official train, not writer weight training or valid_unseen scoring')
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / 'design.json'
    if frozen.exists():
        if read(frozen) != design:
            raise ValueError('Frozen trajectory experience design changed')
    else:
        save(frozen, design)
    print(json.dumps({'design_sha256': sha(frozen), 'cases': [(c['family'], c['public_task']) for c in cases],
        'claim_memory': memory.public_state()}), flush=True)
    if args.prepare_only:
        return
    plan = dict(plan_source['alf'], actor_url=args.url)
    for repeat in REPEATS:
        stores = {'online_claims': ClaimMemory(),
                  'pretrained_frozen': copy.deepcopy(memory),
                  'pretrained_online': copy.deepcopy(memory)}
        for index, case in enumerate(cases):
            for arm in ARMS:
                target = output / str(repeat) / f'{index:02d}_{case["input_sha256"][:12]}' / arm
                target.mkdir(parents=True, exist_ok=True)
                if arm != 'native':
                    before = snapshot_memory(stores[arm])
                    before_path = target / 'memory_before.json'
                    if before_path.exists():
                        if read(before_path) != before:
                            raise ValueError('Online trajectory memory chain changed before episode')
                    else:
                        save(before_path, before)
                if (target / 'episode.json').exists():
                    episode = read(target / 'episode.json')
                    if episode['game'] != case['game']:
                        raise ValueError('Existing episode belongs to a different source game')
                else:
                    actor = (Actor(plan) if arm == 'native' else
                             TrajectoryExperienceActor(plan, case['public_task'], stores[arm]))
                    try:
                        episode = actor.run_many([dict(game=case['game'], memory='',
                            seed=seed(repeat, case['game'], 0, 'actor'), output=target)])[0]
                    finally:
                        actor.pool.shutdown(wait=True)
                update = None
                if arm in ('online_claims', 'pretrained_online'):
                    operations = teacher_operations(stores[arm], episode)
                    result = stores[arm].apply(operations, episode, case['input_sha256'])
                    if result['rejected']:
                        raise ValueError('New trajectory claim failed exact feedback review')
                    update = dict(schema='alf_trajectory_claim_update_v1',
                        game=case['game'], input_sha256=case['input_sha256'],
                        source_episode_sha256=sha(target / 'episode.json'),
                        operations=operations, applied=result['applied'], reviewed=True)
                    update_path = target / 'reviewed_update.json'
                    if update_path.exists():
                        if read(update_path) != update:
                            raise ValueError('Online claim annotation changed')
                    else:
                        save(update_path, update)
                if arm != 'native':
                    after = snapshot_memory(stores[arm])
                    after_path = target / 'memory_after.json'
                    if after_path.exists():
                        if read(after_path) != after:
                            raise ValueError('Online trajectory memory changed after episode')
                    else:
                        save(after_path, after)
                print(json.dumps(dict(family=case['family'], input_sha256=case['input_sha256'],
                    index=index, repeat=repeat, arm=arm, reward=episode['reward'], steps=episode['steps'],
                    new_claims=len(update['operations']) if update else 0,
                    memory_entries=len(stores[arm].entries) if arm != 'native' else 0,
                    activated_turns=sum(bool(g.get('experience_decision', {}).get('claim_keys'))
                                        for g in episode['generations']))), flush=True)


if __name__ == '__main__':
    main()
