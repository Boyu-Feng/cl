"""Posthoc same-seed stability control for seven positive unseen pairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.alfworld_comparison.environment import Actor
from ttcl.experience_evolution.core import seed
from ttcl.icl_mem0_comparison.protocol import read, save, sha
from . import probe_state_progress_actor as pilot
from .probe_state_progress_holdout import extended_goal


def design_for(source: Path, audit_path: Path, reviews: Path, url: str) -> dict:
    source_design = read(source / 'design.json')
    audit = read(audit_path)
    review = read(reviews)
    if (source_design['schema'] != 'alf_state_progress_unseen_v1' or
            audit['schema'] != 'alf_state_progress_unseen_audit_v1' or
            audit['design_sha256'] != sha(source / 'design.json') or
            review['schema'] != 'alf_state_progress_unseen_repeat_reviews_v1'):
        raise ValueError('Frozen unseen source or repeat review changed')
    pairs = {}
    for row in audit['rows']:
        pairs.setdefault((row['game'],row['repeat']),{})[row['arm']] = row
    selected = []
    for (game,repeat), arms in pairs.items():
        if arms['state_progress']['reward'] <= arms['native']['reward']:
            continue
        case = next(x for x in source_design['cases'] if x['game'] == game)
        matching = [x for x in review['targets'] if x['game'] == game and x['repeat'] == repeat]
        if (len(matching) != 1 or not matching[0]['reviewed'] or
                matching[0]['input_sha256'] != case['input_sha256'] or
                matching[0]['public_task'] != case['public_task']):
            raise ValueError(f'Repeat input review mismatch: {game}:{repeat}')
        selected.append(dict(case, repeat=repeat,
                             original_native_episode_sha256=arms['native']['episode_sha256'],
                             original_candidate_episode_sha256=arms['state_progress']['episode_sha256']))
    if len(selected) != 7 or len(review['targets']) != 7:
        raise ValueError('Expected seven reviewed positive pairs')
    return dict(schema='alf_state_progress_unseen_repeat_v1', cases=selected,
                source_design_sha256=sha(source / 'design.json'),
                source_audit_sha256=sha(audit_path), source_root=str(source),
                source_plan=str(source_design['plan_path']),
                source_plan_sha256=source_design['plan_sha256'],
                review_sha256=sha(reviews), review_path=str(reviews),
                runner_sha256=sha(Path(__file__)),
                policy_sha256=sha(Path(pilot.__file__)), url=url,
                arms=['state_progress_repeat','native_repeat'],
                budget='Seven posthoc positive official valid_unseen pairs; same game and actor seed, candidate then native, original 50-action and 50-call limits',
                caveat='Same-input repeat control only; selected after seeing positive differences, not additional independent test evidence')


def run(design: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / 'design.json'
    if frozen.exists():
        if read(frozen) != design:
            raise ValueError('Frozen repeat design changed')
    else:
        save(frozen, design)
    print(json.dumps({'design_sha256':sha(frozen)}), flush=True)
    plan = dict(read(Path(design['source_plan']))['alf'], actor_url=design['url'])
    pilot._goal = extended_goal
    for case in design['cases']:
        for arm in design['arms']:
            target = output / case['family'] / str(case['rank']) / str(case['repeat']) / arm
            if (target / 'episode.json').exists():
                continue
            target.mkdir(parents=True, exist_ok=True)
            actor = (pilot.StateProgressActor(plan, case['public_task'])
                     if arm == 'state_progress_repeat' else Actor(plan))
            try:
                episode = actor.run_many([dict(game=case['game'], memory='',
                                    seed=seed(case['repeat'], case['game'], 0, 'actor'),
                                    output=target)])[0]
            finally:
                actor.pool.shutdown(wait=True)
            print(json.dumps({'family':case['family'], 'rank':case['rank'],
                              'repeat':case['repeat'], 'arm':arm,
                              'reward':episode['reward'], 'steps':episode['steps'],
                              'invalid':sum(not x['valid_command'] for x in episode['trajectory'])}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18559')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    design = design_for(args.source.resolve(), args.audit.resolve(), args.reviews.resolve(), args.url)
    if args.prepare_only:
        args.output.mkdir(parents=True, exist_ok=True)
        frozen = args.output / 'design.json'
        if frozen.exists() and read(frozen) != design:
            raise ValueError('Frozen repeat design changed')
        if not frozen.exists():
            save(frozen, design)
        print(json.dumps({'design_sha256':sha(frozen)}), flush=True)
        return
    run(design, args.output.resolve())


if __name__ == '__main__':
    main()
