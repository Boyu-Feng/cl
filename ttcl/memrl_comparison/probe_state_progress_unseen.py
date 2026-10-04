"""Frozen small valid_unseen evaluation of the state-progress ALFWorld actor.

Two path-hash-selected games per family, two actor seeds, no memory. Unknown
public goal syntax falls back to the original actor and remains in the score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from ttcl.alfworld_comparison.environment import Actor, make_env
from ttcl.experience_evolution.core import seed
from ttcl.icl_mem0_comparison.protocol import read, save, sha
from . import probe_state_progress_actor as pilot
from .probe_state_progress_holdout import extended_goal


REPEATS = (94601, 94602)
ARMS = ('native', 'state_progress')


def design_for(plan_path: Path, review_path: Path, url: str) -> dict:
    plan = read(plan_path)
    review = read(review_path)
    if review['schema'] != 'alf_state_progress_unseen_reviews_v1' or len(review['targets']) != 12:
        raise ValueError('Twelve content-bound public test reviews required')
    root = Path(plan['alf']['data_root']) / 'json_2.1.1/valid_unseen'
    cases = []
    for fam in pilot.FAMILIES:
        paths = sorted(root.glob(f'{fam}-*/trial_*/game.tw-pddl'),
                       key=lambda p: hashlib.sha256(str(p.relative_to(root)).encode()).hexdigest())[:2]
        if len(paths) != 2:
            raise ValueError(f'Missing valid_unseen games: {fam}')
        for rank, game in enumerate(paths):
            rel = 'json_2.1.1/valid_unseen/' + str(game.relative_to(root))
            rows = [x for x in review['targets'] if x['game'] == rel]
            if (len(rows) != 1 or not rows[0]['reviewed'] or
                    rows[0]['family'] != fam or rows[0]['input_sha256'] != sha(game)):
                raise ValueError(f'Public input review changed: {rel}')
            env = make_env(game)
            try:
                state = env.reset()
                task = re.search(r'Your task is to: (.+)', str(state['feedback'])).group(1)
            finally:
                env.close()
            if task != rows[0]['public_task']:
                raise ValueError(f'Public instruction changed: {rel}')
            try:
                extended_goal(task)
                supported = True
            except ValueError:
                supported = False
            cases.append(dict(family=fam, rank=rank, game=rel, input_sha256=sha(game),
                              public_task=task, supported=supported,
                              initial_observation_sha256=hashlib.sha256(
                                  str(state['feedback']).encode()).hexdigest()))
    return dict(schema='alf_state_progress_unseen_v1', split='valid_unseen',
                cases=cases, repeats=list(REPEATS), arms=list(ARMS),
                plan_path=str(plan_path), plan_sha256=sha(plan_path),
                review_path=str(review_path), review_sha256=sha(review_path),
                policy_sha256=sha(Path(pilot.__file__)),
                parser_sha256=sha(Path(__file__).with_name('probe_state_progress_holdout.py')),
                runner_sha256=sha(Path(__file__)), url=url,
                budget='12 hash-selected official valid_unseen games, two actor seeds, two arms, no memory, max 50 official environment actions and 50 actor calls per arm',
                caveat='Small frozen test subset; unsupported public goal syntax uses native actor and remains in paired score; no post-test tuning')


def run(design: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / 'design.json'
    if frozen.exists():
        if read(frozen) != design:
            raise ValueError('Frozen unseen design changed')
    else:
        save(frozen, design)
    print(json.dumps({'design_sha256':sha(frozen),
                      'supported':sum(x['supported'] for x in design['cases'])}), flush=True)
    source = read(Path(design['plan_path']))
    plan = dict(source['alf'], actor_url=design['url'])
    pilot._goal = extended_goal
    for case in design['cases']:
        for repeat in REPEATS:
            for arm in ARMS:
                target = output / case['family'] / str(case['rank']) / str(repeat) / arm
                if (target / 'episode.json').exists():
                    continue
                target.mkdir(parents=True, exist_ok=True)
                actor = (pilot.StateProgressActor(plan, case['public_task'])
                         if arm == 'state_progress' and case['supported'] else Actor(plan))
                try:
                    episode = actor.run_many([dict(game=case['game'], memory='',
                                  seed=seed(repeat, case['game'], 0, 'actor'), output=target)])[0]
                finally:
                    actor.pool.shutdown(wait=True)
                print(json.dumps({'family':case['family'], 'rank':case['rank'],
                                  'repeat':repeat, 'arm':arm, 'supported':case['supported'],
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
    if args.prepare_only:
        args.output.mkdir(parents=True, exist_ok=True)
        frozen = args.output / 'design.json'
        if frozen.exists() and read(frozen) != design:
            raise ValueError('Frozen unseen design changed')
        if not frozen.exists():
            save(frozen, design)
        print(json.dumps({'design_sha256':sha(frozen),
                          'supported':sum(x['supported'] for x in design['cases'])}), flush=True)
        return
    run(design, args.output.resolve())


if __name__ == '__main__':
    main()
