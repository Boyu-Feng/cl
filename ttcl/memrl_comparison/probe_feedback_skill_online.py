"""Transductive ALFWorld valid_unseen diagnostic with per-game skill updates.

This is deliberately separate from frozen held-out scoring: each online arm
may use earlier test-game feedback, never the current game's future feedback.
Two independent seed streams prevent experience sharing between repeats.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from ttcl.alfworld_comparison.environment import make_env
from ttcl.experience_evolution.core import digest, seed
from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .feedback_skill_actor import FeedbackSkillActor, PREP, effect, learn, public_state


FAMILIES = ('pick_clean_then_place_in_recep', 'pick_cool_then_place_in_recep',
            'pick_heat_then_place_in_recep')
REPEATS = (94801, 94802)
ARMS = ('empty_frozen', 'empty_online', 'trained_frozen', 'trained_online')


class SkillStore:
    """One distinct input contributes at most one positive/negative vote."""
    def __init__(self, review: dict | None = None):
        self.evidence: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
            lambda: {'positive': set(), 'negative': set()})
        if review:
            for target in review['targets']:
                for row in target['events']:
                    key = (row['verb'], row['tool'].split()[0])
                    self.evidence[key]['positive' if row['positive'] else 'negative'].add(
                        target['input_sha256'])

    def snapshot(self) -> dict:
        skills = {}
        for (verb, tool), evidence in sorted(self.evidence.items()):
            pos, neg = evidence['positive'], evidence['negative']
            if len(pos) < 2:
                continue
            skills.setdefault(verb, []).append(dict(tool=tool, positive_games=len(pos),
                negative_games=len(neg), source_input_sha256=sorted(pos)))
        for rows in skills.values():
            rows.sort(key=lambda row: (-row['positive_games'], row['negative_games'], row['tool']))
        return {'skills': skills}

    def observe(self, input_sha: str, episode: dict) -> dict:
        """Review this episode's exact local effects, then update the next task."""
        events = []
        for index, row in enumerate(episode['trajectory']):
            match = PREP.fullmatch(row['action'].lower().strip())
            if not match:
                continue
            verb, obj, tool = match.groups()
            positive = effect(row['action'], row['observation']) is not None
            if not positive and 'Nothing happens.' not in row['observation']:
                continue
            event = dict(trajectory_index=index, verb=verb, object=obj, tool=tool,
                positive=positive, action=row['action'], observation=row['observation'],
                row_sha256=digest(json.dumps(row, sort_keys=True, ensure_ascii=False)))
            events.append(event)
            self.evidence[(verb, tool.split()[0])]['positive' if positive else 'negative'].add(input_sha)
        return dict(schema='alf_online_effect_review_v1', game=episode['game'],
            input_sha256=input_sha, episode_sha256=digest(json.dumps(episode, sort_keys=True, ensure_ascii=False)),
            evidence_rule='Exact action, object ID and tool ID in positive environment feedback; or exact Nothing happens negative feedback',
            events=events, reviewed=True)


def select_cases(data_root: Path, excluded: set[str]) -> list[dict]:
    root = data_root / 'json_2.1.1/valid_unseen'
    by_family = {}
    for family in FAMILIES:
        paths = sorted(root.glob(f'{family}-*/trial_*/game.tw-pddl'),
            key=lambda path: hashlib.sha256(str(path.relative_to(root)).encode()).hexdigest())
        selected = []
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
            try:
                public_state(task, [])
            except ValueError:
                raise ValueError(f'Unreviewed public goal syntax: {task}')
            selected.append(dict(family=family, game=str(path.relative_to(data_root)),
                input_sha256=input_sha, public_task=task,
                initial_observation_sha256=hashlib.sha256(feedback.encode()).hexdigest(),
                reviewed=True))
            if len(selected) == 3:
                break
        if len(selected) != 3:
            raise ValueError(f'Missing three independent unseen games for {family}')
        by_family[family] = selected
    return [by_family[family][index] for family in FAMILIES for index in range(3)]


def run(args):
    plan_path = args.plan.resolve()
    review_path = args.reviews.resolve()
    previous_paths = [path.resolve() for path in args.exclude_design]
    output = args.output.resolve()
    plan_source, review = read(plan_path), read(review_path)
    data_root = Path(plan_source['alf']['data_root']).resolve()
    trained = learn(review, data_root)
    initialized = SkillStore(review).snapshot()
    if initialized['skills'] != trained['skills']:
        raise ValueError('Incremental train initializer differs from frozen learner')
    excluded = {target['input_sha256'] for target in review['targets']}
    for path in previous_paths:
        excluded.update(case['input_sha256'] for case in read(path)['cases'])
    cases = select_cases(data_root, excluded)
    design = dict(schema='alf_feedback_skill_online_v1', cases=cases,
        selection='First three SHA-ranked valid_unseen games per preparation family, excluding prior unseen designs and all train skill source inputs, before outcomes; family-blocked order',
        repeats=list(REPEATS), arms=list(ARMS),
        plan_path=str(plan_path), plan_sha256=sha(plan_path),
        review_path=str(review_path), review_sha256=sha(review_path),
        exclude_designs=[dict(path=str(path), sha256=sha(path)) for path in previous_paths],
        initial_trained_skill_table=trained, runner_sha256=sha(Path(__file__)),
        core_runner_sha256=sha(Path(__file__).with_name('feedback_skill_actor.py')),
        url=args.url, budget='Nine different valid_unseen games per seed stream, two seed streams, four arms, max 50 environment actions and actor calls per episode; update only after full episode',
        caveat='Test-time cross-game adaptation is transductive diagnostic, not standard frozen valid_unseen scoring')
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / 'design.json'
    if frozen.exists():
        if read(frozen) != design:
            raise ValueError('Frozen online design changed')
    else:
        save(frozen, design)
    print(json.dumps({'design_sha256': sha(frozen),
        'cases': [(case['family'], case['input_sha256'][:12], case['public_task']) for case in cases]}), flush=True)
    if args.prepare_only:
        return
    plan = dict(plan_source['alf'], actor_url=args.url)
    for repeat in REPEATS:
        stores = {arm: SkillStore(review if arm.startswith('trained_') else None) for arm in ARMS}
        for index, case in enumerate(cases):
            for arm in ARMS:
                target = output / str(repeat) / f'{index:02d}_{case["input_sha256"][:12]}' / arm
                target.mkdir(parents=True, exist_ok=True)
                before = stores[arm].snapshot()
                before_path = target / 'skill_before.json'
                if before_path.exists():
                    if read(before_path) != before:
                        raise ValueError('Online chain changed before episode')
                else:
                    save(before_path, before)
                episode_path = target / 'episode.json'
                if episode_path.exists():
                    episode = read(episode_path)
                    if episode['game'] != case['game']:
                        raise ValueError('Existing episode input changed')
                else:
                    actor = FeedbackSkillActor(plan, case['public_task'], before)
                    try:
                        episode = actor.run_many([dict(game=case['game'], memory='',
                            seed=seed(repeat, case['game'], 0, 'actor'), output=target)])[0]
                    finally:
                        actor.pool.shutdown(wait=True)
                if arm.endswith('_online'):
                    annotation = stores[arm].observe(case['input_sha256'], episode)
                    update_path = target / 'reviewed_update.json'
                    if update_path.exists():
                        if read(update_path) != annotation:
                            raise ValueError('Online update annotation changed')
                    else:
                        save(update_path, annotation)
                after = stores[arm].snapshot()
                after_path = target / 'skill_after.json'
                if after_path.exists():
                    if read(after_path) != after:
                        raise ValueError('Online skill chain changed after episode')
                else:
                    save(after_path, after)
                print(json.dumps(dict(index=index, family=case['family'],
                    input_sha256=case['input_sha256'], repeat=repeat, arm=arm,
                    reward=episode['reward'], steps=episode['steps'],
                    effect_events=len(annotation['events']) if arm.endswith('_online') else 0,
                    skill_types_before={k: [row['tool'] for row in v] for k, v in before['skills'].items()},
                    skill_types_after={k: [row['tool'] for row in v] for k, v in after['skills'].items()})), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--exclude-design', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18559')
    parser.add_argument('--prepare-only', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
