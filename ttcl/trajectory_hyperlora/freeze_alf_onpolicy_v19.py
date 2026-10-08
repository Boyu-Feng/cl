"""Freeze new content-bound targets for method-on-policy ALFWorld chains."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash


METHODS = ('base', 'continuous_success_mean',
           'fixed_first_success', 'own_success_raw_text')


def expected(args):
    plan = json.loads(args.plan.read_text())
    if (plan['orders'] != 3 or plan['per_family_per_order'] != 5 or
            len(plan['order_plans']) != 3 or
            plan['previously_exposed_game_count'] != 134):
        raise ValueError('Changed precommitted official target orders')
    plan_hash = file_hash(args.plan)
    checkpoint_hash = file_hash(args.checkpoint)
    residual_hash = file_hash(args.residual)
    targets = []
    seen_games = set()
    for sequence in plan['order_plans']:
        order_id = sequence['order_id']
        if order_id != len({row['order_id'] for row in targets}):
            raise ValueError('Changed order identifiers')
        if len(sequence['targets']) != 30:
            raise ValueError('Incomplete official order')
        for position, target in enumerate(sequence['targets']):
            game = target['game']
            if (target['index'] != position or game in seen_games or
                    '/valid_unseen/' not in '/' + game or
                    file_hash(args.data_root / game) != target['game_sha256']):
                raise ValueError('Changed, repeated, or unbound official game')
            seen_games.add(game)
            for method in METHODS:
                content = {'protocol': 'alf_onpolicy_v19',
                    'plan_sha256': plan_hash,
                    'checkpoint_sha256': checkpoint_hash,
                    'residual_sha256': residual_hash,
                    'order_id': order_id, 'position': position,
                    'method': method, 'family': target['family'],
                    'game': game,
                    'game_sha256': target['game_sha256'],
                    'max_steps': 50, 'max_new_tokens': 64,
                    'actor_history_turns': 2,
                    'loop_guard_max': 2, 'source_token_limit': 2048,
                    'memory_eligibility': 'own official-won episodes only'}
                targets.append({**content,
                    'input_content_sha256': digest(content),
                    'reviewed_target': True})
    if len(targets) != 360 or len(seen_games) != 90:
        raise ValueError('Incomplete distinct official method-on-policy plan')
    return {'protocol': 'Fresh v19 per-method target and input-content bindings for three precommitted official valid_unseen orders; each method starts empty and collects only its own successes; all games named in older reports, so exploratory not unseen-game blind',
        'plan_sha256': plan_hash,
        'checkpoint_sha256': checkpoint_hash,
        'residual_sha256': residual_hash,
        'methods': list(METHODS),
        'orders': 3, 'targets_per_order': 30,
        'targets': targets}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'check'))
    parser.add_argument('--plan', type=Path, default=Path(
        'ttcl/trajectory_hyperlora/alf_valid_unseen_multiorder_v10_plan.json'))
    parser.add_argument('--data-root', type=Path, default=Path(
        'ttcl/data/alfworld_delta'))
    parser.add_argument('--checkpoint', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--residual', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_20261007.pt'))
    parser.add_argument('--review', type=Path, default=Path(
        'data/annotations/alf_onpolicy_multiorder_v19_reviewed_20261008.json'))
    args = parser.parse_args()
    value = expected(args)
    if args.command == 'prepare':
        if args.review.exists():
            raise FileExistsError(args.review)
        args.review.parent.mkdir(parents=True, exist_ok=True)
        args.review.write_text(json.dumps(value, ensure_ascii=False,
                                          indent=2) + '\n')
    elif json.loads(args.review.read_text()) != value:
        raise ValueError('Changed reviewed official order or actor')
    print(json.dumps({'bindings': len(value['targets']),
        'methods': len(METHODS), 'orders': value['orders'],
        'review_sha256': file_hash(args.review)}), flush=True)


if __name__ == '__main__':
    main()
