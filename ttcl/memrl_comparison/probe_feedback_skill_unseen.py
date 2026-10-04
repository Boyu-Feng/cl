"""Frozen feedback-skill transfer check on new official valid_unseen games."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.alfworld_comparison.environment import make_env
from ttcl.experience_evolution.core import seed
from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .feedback_skill_actor import FeedbackSkillActor, learn


FAMILIES = ('pick_clean_then_place_in_recep', 'pick_cool_then_place_in_recep',
            'pick_heat_then_place_in_recep')
REPEATS = (94701, 94702)


def select_cases(data_root: Path, excluded: set[str]) -> list[dict]:
    cases = []
    root = data_root / 'json_2.1.1/valid_unseen'
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
            if selected == 2:
                break
        if selected != 2:
            raise ValueError(f'Missing two unseen games: {family}')
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--previous-unseen-design', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18559')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    plan_path, reviews_path = args.plan.resolve(), args.reviews.resolve()
    previous_path, output = args.previous_unseen_design.resolve(), args.output.resolve()
    plan_source, review, previous = read(plan_path), read(reviews_path), read(previous_path)
    data_root = Path(plan_source['alf']['data_root']).resolve()
    if any(not case['game'].startswith('json_2.1.1/valid_unseen/') for case in previous['cases']):
        raise ValueError('Previous exclusion design must be valid_unseen')
    excluded = {case['input_sha256'] for case in previous['cases']}
    excluded.update(target['input_sha256'] for target in review['targets'])
    cases = select_cases(data_root, excluded)
    table = learn(review, data_root)
    design = dict(schema='alf_feedback_skill_unseen_v1', cases=cases,
        selection='First two SHA-ranked official valid_unseen games per preparation family, excluding prior state-progress unseen games and all skill-source games, before outcomes',
        repeats=list(REPEATS), arms=['empty_skill', 'learned_skill'],
        plan_path=str(plan_path), plan_sha256=sha(plan_path),
        review_path=str(reviews_path), review_sha256=sha(reviews_path),
        previous_unseen_design_path=str(previous_path), previous_unseen_design_sha256=sha(previous_path),
        skill_table=table, runner_sha256=sha(Path(__file__)),
        core_runner_sha256=sha(Path(__file__).with_name('feedback_skill_actor.py')),
        url=args.url, budget='Six valid_unseen games, two seeds, two arms, max 50 environment actions and actor calls; frozen train-only skill table')
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / 'design.json'
    if frozen.exists():
        if read(frozen) != design:
            raise ValueError('Frozen unseen design changed')
    else:
        save(frozen, design)
    print(json.dumps({'design_sha256': sha(frozen),
                      'cases': [(case['family'], case['input_sha256'][:12], case['public_task']) for case in cases]}), flush=True)
    if args.prepare_only:
        return
    plan = dict(plan_source['alf'], actor_url=args.url)
    for case in cases:
        for repeat in REPEATS:
            for arm in design['arms']:
                target = output / case['family'] / case['input_sha256'][:12] / str(repeat) / arm
                if (target / 'episode.json').exists():
                    if read(target / 'episode.json')['game'] != case['game']:
                        raise ValueError('Existing unseen episode has wrong input')
                    continue
                target.mkdir(parents=True, exist_ok=True)
                actor = FeedbackSkillActor(plan, case['public_task'], table if arm == 'learned_skill' else {'skills': {}})
                try:
                    episode = actor.run_many([dict(game=case['game'], memory='',
                        seed=seed(repeat, case['game'], 0, 'actor'), output=target)])[0]
                finally:
                    actor.pool.shutdown(wait=True)
                print(json.dumps({'family': case['family'], 'input_sha256': case['input_sha256'],
                    'repeat': repeat, 'arm': arm, 'reward': episode['reward'], 'steps': episode['steps'],
                    'invalid': sum(not row['valid_command'] for row in episode['trajectory']),
                    'learned_actions': sum(g.get('skill_reason', '').startswith('learned_') for g in episode['generations'])}), flush=True)


if __name__ == '__main__':
    main()
