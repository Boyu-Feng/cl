"""Explore a state-progress actor independent of MemRL on six ALFWorld train games.

One hash-selected game per family, no learned memory or extra model calls.
Public goal/feedback/available commands drive a small state tracker. This is
an exploratory first-attempt pilot, not valid_unseen or a full online chain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from ttcl.alfworld_comparison.environment import Actor, make_env
from ttcl.experience_evolution.core import digest, seed
from ttcl.icl_mem0_comparison.protocol import read, save, sha


FAMILIES = ('look_at_obj_in_light', 'pick_and_place_simple',
            'pick_clean_then_place_in_recep', 'pick_cool_then_place_in_recep',
            'pick_heat_then_place_in_recep', 'pick_two_obj_and_place')
REPEATS = (94401,)
ARMS = ('native', 'state_progress')


def _goal(task: str) -> tuple[str, str, str | None]:
    t = task.lower().rstrip('.')
    patterns = (
        (r'examine the (\w+) with the (\w+)', 'look'),
        (r'put a (\w+) in (\w+)', 'simple'),
        (r'put a clean (\w+) in (\w+)', 'clean'),
        (r'cool some (\w+) and put it in (\w+)', 'cool'),
        (r'put a hot (\w+) in (\w+)', 'heat'),
        (r'find two (\w+) and put them in (\w+)', 'two'),
    )
    for pattern, kind in patterns:
        match = re.fullmatch(pattern, t)
        if match:
            return kind, match.group(1), match.group(2)
    raise ValueError(f'Unreviewed goal syntax: {task}')


def select_public_action(task: str, messages: list[dict], available: list[str]) -> tuple[str | None, dict]:
    kind, obj, destination = _goal(task)
    previous = [m['content'].strip().lower() for m in messages if m['role'] == 'assistant']
    visited = list(dict.fromkeys(a.removeprefix('go to ') for a in previous if a.startswith('go to ')))
    moved_ids = set()
    for action in previous:
        m = re.fullmatch(rf'move ({re.escape(obj)} \d+) to \S+ \d+', action)
        if m:
            moved_ids.add(m.group(1))
    target = rf'{re.escape(obj)} \d+'
    dest = rf'{re.escape(destination)} \d+'
    matching_take = [a for a in available if re.fullmatch(rf'take ({target}) from \S+ \d+', a)]
    matching_take = [a for a in matching_take if re.match(r'take (\S+ \d+) from ', a).group(1) not in moved_ids]
    matching_move = [a for a in available if re.fullmatch(rf'move ({target}) to {dest}', a)]
    prepared = kind in ('simple', 'two') or any(
        a.startswith({'clean': 'clean ', 'cool': 'cool ', 'heat': 'heat '}.get(kind, '\0')) and
        re.search(rf'\b{target}\b', a) for a in previous)
    plan = dict(kind=kind, target=obj, destination=destination,
                visited=visited, moved=sorted(moved_ids), prepared=prepared)
    if kind == 'look':
        lamps = [a for a in available if re.fullmatch(rf'use {dest}', a)]
        if lamps and not any(a in previous for a in lamps):
            return lamps[0], dict(plan, reason='turn_on_goal_light')
    if matching_move and prepared:
        return matching_move[0], dict(plan, reason='finish_or_advance_goal')
    if kind in ('clean', 'cool', 'heat') and not prepared:
        prep = [a for a in available if re.fullmatch(
            rf'{kind} {target} with \S+ \d+', a)]
        if prep:
            return prep[0], dict(plan, reason='perform_required_preparation')
    if matching_take:
        return matching_take[0], dict(plan, reason='take_goal_object')
    return None, dict(plan, reason='actor_search')


class StateProgressActor(Actor):
    def __init__(self, plan: dict, task: str):
        super().__init__(plan)
        self.task = task

    def generate(self, messages: list[dict], random_seed: int) -> dict:
        user = messages[-1]['content']
        available = user.split('\nAvailable commands:\n', 1)[1].splitlines()
        command, state = select_public_action(self.task, messages, available)
        if command is not None:
            return dict(text=command, finish_reason='public_state_rule',
                        usage={'prompt_tokens': 0, 'completion_tokens': 0},
                        seed=random_seed, prompt_sha256=digest(json.dumps(messages, ensure_ascii=False)),
                        seconds=0.0, public_state=state)
        augmented = [dict(m) for m in messages]
        augmented[-1]['content'] += ('\n\nCurrent goal: ' + self.task +
            '\nVisited locations: ' + (', '.join(state['visited'][-20:]) or 'none') +
            '\nChoose a listed command that advances the unfinished goal. '
            'Search a new location when possible; revisit one when carrying the goal object.')
        result = super().generate(augmented, random_seed)
        result['public_state'] = state
        return result


def design_for(plan_path: Path, review_path: Path, url: str) -> dict:
    plan = read(plan_path)
    review = read(review_path)
    if review['schema'] != 'alf_state_progress_reviews_v1' or len(review['targets']) != 6:
        raise ValueError('Six reviewed goals are required')
    root = Path(plan['alf']['data_root']) / 'json_2.1.1/train'
    cases = []
    for fam in FAMILIES:
        paths = sorted(root.glob(f'{fam}-*/trial_*/game.tw-pddl'),
                       key=lambda p: hashlib.sha256(str(p.relative_to(root)).encode()).hexdigest())
        if not paths:
            raise ValueError(f'Missing official train family {fam}')
        game = paths[0]
        rel = 'json_2.1.1/train/' + str(game.relative_to(root))
        rows = [x for x in review['targets'] if x['family'] == fam]
        if len(rows) != 1 or rows[0]['game'] != rel or rows[0]['input_sha256'] != sha(game) or not rows[0]['reviewed']:
            raise ValueError(f'Missing content-bound review for {fam}')
        env = make_env(game)
        try:
            state = env.reset()
            task = re.search(r'Your task is to: (.+)', str(state['feedback'])).group(1)
        finally:
            env.close()
        if rows[0]['public_task'] != task:
            raise ValueError(f'Reviewed public goal changed for {fam}')
        _goal(task)
        cases.append(dict(family=fam, game=rel, input_sha256=sha(game), public_task=task,
                          initial_observation_sha256=hashlib.sha256(str(state['feedback']).encode()).hexdigest()))
    return dict(schema='alf_state_progress_pilot_v1', cases=cases, repeats=list(REPEATS),
                arms=list(ARMS), plan_path=str(plan_path), plan_sha256=sha(plan_path),
                review_path=str(review_path), review_sha256=sha(review_path),
                runner_sha256=sha(Path(__file__)), url=url,
                budget='Six result-blind hash-selected official train games, one seed, native versus state-progress, no memory, max 50 environment actions and 50 actor calls per arm',
                caveat='Exploratory one-seed first-attempt pilot, not online memory learning or a general benchmark result')


def run(design: dict, output: Path, prepare_only: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / 'design.json'
    if frozen.exists():
        if read(frozen) != design:
            raise ValueError('Frozen design changed')
    else:
        save(frozen, design)
    print(json.dumps({'design_sha256': sha(frozen)}), flush=True)
    if prepare_only:
        return
    source_plan = read(Path(design['plan_path']))
    plan = dict(source_plan['alf'], actor_url=design['url'])
    for case in design['cases']:
        for repeat in REPEATS:
            for arm in ARMS:
                target = output / case['family'] / str(repeat) / arm
                if (target / 'episode.json').exists():
                    continue
                target.mkdir(parents=True, exist_ok=True)
                actor = Actor(plan) if arm == 'native' else StateProgressActor(plan, case['public_task'])
                try:
                    episode = actor.run_many([dict(game=case['game'], memory='',
                                  seed=seed(repeat, case['game'], 0, 'actor'), output=target)])[0]
                finally:
                    actor.pool.shutdown(wait=True)
                print(json.dumps({'family':case['family'], 'repeat':repeat, 'arm':arm,
                                  'reward':episode['reward'], 'steps':episode['steps'],
                                  'invalid':sum(not x['valid_command'] for x in episode['trajectory'])}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18559')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    design = design_for(args.plan.resolve(), args.reviews.resolve(), args.url)
    run(design, args.output.resolve(), args.prepare_only)


if __name__ == '__main__':
    main()
