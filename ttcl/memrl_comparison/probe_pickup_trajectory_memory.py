"""Online trajectory memory atop a shared public-goal pickup controller."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from ttcl.alfworld_comparison.environment import Actor, make_env
from ttcl.experience_evolution.core import digest, seed
from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .feedback_skill_actor import public_state
from .probe_trajectory_experience_actor import relevant_experience, snapshot_memory
from .trajectory_writer_learning import ClaimMemory, teacher_operations


FAMILIES = ('pick_clean_then_place_in_recep', 'pick_cool_then_place_in_recep',
            'pick_heat_then_place_in_recep')
REPEATS = (94911, 94912)
ARMS = ('pickup_frozen', 'pickup_online')


class PickupExperienceActor(Actor):
    def __init__(self, plan: dict, task: str, memory: ClaimMemory, use_memory: bool):
        super().__init__(plan)
        self.task, self.memory, self.use_memory = task, memory, use_memory

    def generate(self, messages: list[dict], random_seed: int) -> dict:
        user = messages[-1]['content']
        available = user.split('\nAvailable commands:\n', 1)[1].splitlines()
        state = public_state(self.task, messages)
        obj = state['target']
        target = rf'{re.escape(obj)} \d+'
        for command in available:
            match = re.fullmatch(rf'take ({target}) from \w+ \d+', command)
            if match and match.group(1) not in state['held'] and match.group(1) not in state['moved']:
                return dict(text=command, finish_reason='public_goal_pickup',
                    usage={'prompt_tokens': 0, 'completion_tokens': 0}, seed=random_seed,
                    prompt_sha256=digest(json.dumps(messages, ensure_ascii=False)), seconds=0.0,
                    pickup_state=state, experience_decision={'reason': 'pickup_first'},
                    experience_text_sha256=digest(''))
        prompt = [dict(message) for message in messages]
        prompt[-1]['content'] += ('\n\nCurrent goal: ' + self.task +
            '\nVisited locations: ' + (', '.join(state['visited'][-20:]) or 'none') +
            '\nChoose a listed command that advances the unfinished goal. '
            'Search a new location when possible; revisit one when carrying the goal object.')
        text, decision = (relevant_experience(self.task, messages, self.memory)
                          if self.use_memory else ('', dict(reason='frozen_empty')))
        if text:
            prompt[-1]['content'] += ('\nRelevant experience from completed earlier tasks '
                '(apply only if its condition matches):\n' + text)
        result = super().generate(prompt, random_seed)
        result['pickup_state'], result['experience_decision'] = state, decision
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
            public_state(task, [])
            cases.append(dict(family=family, game=str(path.relative_to(data_root)),
                input_sha256=input_sha, public_task=task,
                initial_observation_sha256=hashlib.sha256(feedback.encode()).hexdigest(),
                reviewed=True))
            selected += 1
            if selected == 3:
                break
        if selected != 3:
            raise ValueError(f'Missing three unseen train inputs: {family}')
    return cases


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
    excluded = {row['input_sha256'] for split in dataset['rows'] for row in dataset['rows'][split]}
    excluded.update(dataset['frozen_pilot_excluded_inputs'])
    for path in excluded_paths:
        excluded.update(case['input_sha256'] for case in read(path)['cases'])
    cases = select_cases(data_root, excluded)
    design = dict(schema='alf_pickup_trajectory_memory_online_v1', cases=cases,
        selection='First three SHA-ranked official train games per preparation family excluding writer corpus and all listed previous designs, before rewards; family-blocked order',
        repeats=list(REPEATS), arms=list(ARMS),
        plan_path=str(plan_path), plan_sha256=sha(plan_path),
        dataset_path=str(dataset_path), dataset_sha256=sha(dataset_path),
        excluded_designs=[dict(path=str(path), sha256=sha(path)) for path in excluded_paths],
        runner_sha256=sha(Path(__file__)), url=args.url,
        budget='Nine official train games per seed, two independent seeds, two arms; shared direct target pickup and search summary, max 50 steps; online experience only after each episode',
        caveat='Development train diagnostic; no writer weight update or standard unseen score')
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / 'design.json'
    if frozen.exists():
        if read(frozen) != design:
            raise ValueError('Frozen pickup-memory design changed')
    else:
        save(frozen, design)
    print(json.dumps({'design_sha256': sha(frozen),
        'cases': [(c['family'], c['public_task']) for c in cases]}), flush=True)
    if args.prepare_only:
        return
    plan = dict(plan_source['alf'], actor_url=args.url)
    for repeat in REPEATS:
        online = ClaimMemory()
        for index, case in enumerate(cases):
            for arm in ARMS:
                target = output / str(repeat) / f'{index:02d}_{case["input_sha256"][:12]}' / arm
                target.mkdir(parents=True, exist_ok=True)
                before = snapshot_memory(online) if arm == 'pickup_online' else {}
                if (target / 'memory_before.json').exists():
                    if read(target / 'memory_before.json') != before:
                        raise ValueError('Online memory changed before task')
                else:
                    save(target / 'memory_before.json', before)
                if (target / 'episode.json').exists():
                    episode = read(target / 'episode.json')
                    if episode['game'] != case['game']:
                        raise ValueError('Existing episode has wrong input')
                else:
                    actor = PickupExperienceActor(plan, case['public_task'], online,
                                                  arm == 'pickup_online')
                    try:
                        episode = actor.run_many([dict(game=case['game'], memory='',
                            seed=seed(repeat, case['game'], 0, 'actor'), output=target)])[0]
                    finally:
                        actor.pool.shutdown(wait=True)
                update = None
                if arm == 'pickup_online':
                    operations = teacher_operations(online, episode)
                    result = online.apply(operations, episode, case['input_sha256'])
                    if result['rejected']:
                        raise ValueError('Experience annotation failed exact feedback review')
                    update = dict(schema='alf_trajectory_claim_update_v1', game=case['game'],
                        input_sha256=case['input_sha256'],
                        source_episode_sha256=sha(target / 'episode.json'),
                        operations=operations, applied=result['applied'], reviewed=True)
                    if (target / 'reviewed_update.json').exists():
                        if read(target / 'reviewed_update.json') != update:
                            raise ValueError('Online update changed')
                    else:
                        save(target / 'reviewed_update.json', update)
                after = snapshot_memory(online) if arm == 'pickup_online' else {}
                if (target / 'memory_after.json').exists():
                    if read(target / 'memory_after.json') != after:
                        raise ValueError('Online memory changed after task')
                else:
                    save(target / 'memory_after.json', after)
                print(json.dumps(dict(index=index, repeat=repeat,
                    family=case['family'], input_sha256=case['input_sha256'], arm=arm,
                    reward=episode['reward'], steps=episode['steps'],
                    new_claims=len(update['operations']) if update else 0,
                    memory_entries=len(online.entries) if arm == 'pickup_online' else 0,
                    activated_turns=sum(bool(g.get('experience_decision', {}).get('claim_keys'))
                        for g in episode['generations']))), flush=True)


if __name__ == '__main__':
    main()
