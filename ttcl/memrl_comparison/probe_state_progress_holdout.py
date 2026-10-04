"""Apply the unchanged state-progress action policy to six new train games.

Only public-goal syntax parsing is extended before examining any holdout
rewards. Games are the second SHA-ranked path in each official train family.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

from ttcl.alfworld_comparison.environment import make_env
from ttcl.icl_mem0_comparison.protocol import read, sha
from . import probe_state_progress_actor as pilot


def extended_goal(task: str) -> tuple[str, str, str]:
    text = task.lower().rstrip('.')
    extra = (
        (r'look at (\w+) under the (\w+)', 'look'),
        (r'heat some (\w+) and put it in (\w+)', 'heat'),
        (r'put two (\w+) in (\w+)', 'two'),
    )
    for pattern, kind in extra:
        match = re.fullmatch(pattern, text)
        if match:
            return kind, match.group(1), match.group(2)
    return _original_goal(task)


_original_goal = pilot._goal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--pilot-reviews', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18559')
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    review_path = args.reviews.resolve()
    pilot._goal = extended_goal
    pilot.REPEATS = (94501,)
    design = pilot.design_for(plan_path, args.pilot_reviews.resolve(), args.url)
    original_review_sha256 = design['review_sha256']
    review = read(review_path)
    if review['schema'] != 'alf_state_progress_holdout_reviews_v1' or len(review['targets']) != 6:
        raise ValueError('Six new reviewed tasks required')
    source_plan = read(plan_path)
    root = Path(source_plan['alf']['data_root']) / 'json_2.1.1/train'
    cases = []
    for family in pilot.FAMILIES:
        paths = sorted(root.glob(f'{family}-*/trial_*/game.tw-pddl'),
                       key=lambda p: hashlib.sha256(str(p.relative_to(root)).encode()).hexdigest())
        if len(paths) < 2:
            raise ValueError(f'Missing second hash-ranked game: {family}')
        game = paths[1]
        rel = 'json_2.1.1/train/' + str(game.relative_to(root))
        matching = [x for x in review['targets'] if x['family'] == family]
        if (len(matching) != 1 or matching[0]['game'] != rel or
                matching[0]['input_sha256'] != sha(game) or not matching[0]['reviewed']):
            raise ValueError(f'New input binding failed: {family}')
        env = make_env(game)
        try:
            state = env.reset()
            task = re.search(r'Your task is to: (.+)', str(state['feedback'])).group(1)
        finally:
            env.close()
        if matching[0]['public_task'] != task:
            raise ValueError(f'Public task changed: {family}')
        extended_goal(task)
        cases.append(dict(family=family, game=rel, input_sha256=sha(game),
                          public_task=task, initial_observation_sha256=hashlib.sha256(
                              str(state['feedback']).encode()).hexdigest()))
    design.update(cases=cases, review_path=str(review_path), review_sha256=sha(review_path),
                  pilot_review_sha256=original_review_sha256,
                  base_runner_sha256=design['runner_sha256'],
                  runner_sha256=sha(Path(__file__)),
                  budget='Six new content-bound hash-ranked official train games, one actor seed, two arms, max 50 actions and actor calls each',
                  caveat='Development holdout after the six-game pilot; goal syntax extended using public instructions before outcomes, action selection unchanged; not valid_unseen or full online CLBench')
    pilot.run(design, args.output.resolve(), False)


if __name__ == '__main__':
    main()
