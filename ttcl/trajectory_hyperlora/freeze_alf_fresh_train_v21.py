"""Freeze locally unexposed official-train games for final confirmation.

This is a path-blind held-out evaluation set inside the official train split,
not the official valid_unseen benchmark. Look-at-object has no completely
unexposed task directories left, so that family can only be path-disjoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v2 import FAMILIES


SALT = 'alf_fresh_train90_v21_20261008'
ROOTS = ('results', 'data/annotations', 'ttcl/results',
         'ttcl/trajectory_hyperlora', 'current_work')
GAME_PATTERN = r'json_2\.1\.1/train/[^"[:space:]]+/game\.tw-pddl'


def scan_exposure():
    if any(not Path(root).exists() for root in ROOTS):
        raise FileNotFoundError('Missing declared exposure-scan root')
    command = ['rg', '--no-ignore', '-o', '--no-filename',
               '-g', '*.json', GAME_PATTERN, *ROOTS]
    process = subprocess.run(command, capture_output=True,
                             text=True, check=False)
    if process.returncode not in (0, 1) or process.stderr:
        raise RuntimeError(f'Local JSON exposure scan failed: {process.stderr}')
    names = sorted(set(process.stdout.splitlines()))
    if len(names) < 2000:
        raise ValueError('Unexpectedly small prior exposure snapshot')
    return names


def expected(args, snapshot):
    if (snapshot['salt'] != SALT or
            snapshot['scan_roots'] != list(ROOTS) or
            snapshot['games'] != sorted(set(snapshot['games'])) or
            len(snapshot['games']) < 2000):
        raise ValueError('Changed frozen exposure snapshot')
    excluded = set(snapshot['games'])
    inventory = sorted(str(path.relative_to(args.data_root)) for path in
        (args.data_root / 'json_2.1.1/train').rglob('game.tw-pddl'))
    if len(inventory) != 3553 or len(set(inventory)) != 3553:
        raise ValueError('Changed official ALFWorld train inventory')
    exposed_dirs = {Path(game).parts[2] for game in excluded}
    chosen_by_family = {}
    remaining_by_family = {}
    for family in FAMILIES:
        remaining = [game for game in inventory
                     if game not in excluded and
                     Path(game).parts[2].startswith(family+'-')]
        remaining_by_family[family] = len(remaining)
        fresh_dir = [game for game in remaining
                     if Path(game).parts[2] not in exposed_dirs]
        trial_only = [game for game in remaining
                      if Path(game).parts[2] in exposed_dirs]
        key = lambda game: (hashlib.sha256(
            (SALT+'|'+game).encode()).hexdigest(), game)
        ordered = sorted(fresh_dir, key=key) + sorted(trial_only, key=key)
        selected = []
        seen_dirs = set()
        for game in ordered:
            directory = Path(game).parts[2]
            if directory in seen_dirs:
                continue
            selected.append(game)
            seen_dirs.add(directory)
            if len(selected) == 15:
                break
        if len(selected) != 15:
            raise ValueError(f'Insufficient distinct held-out games: {family}')
        chosen_by_family[family] = selected
    orders = []
    selected_games = set()
    for order_id in range(3):
        targets = []
        for round_id in range(5):
            for family in FAMILIES:
                game = chosen_by_family[family][order_id*5+round_id]
                if game in selected_games:
                    raise ValueError('Repeated held-out official game')
                selected_games.add(game)
                targets.append({'index': len(targets), 'round': round_id,
                    'family': family, 'game': game,
                    'game_sha256': file_hash(args.data_root / game),
                    'task_directory_previously_exposed':
                        Path(game).parts[2] in exposed_dirs})
        orders.append({'order_id': order_id, 'targets': targets})
    if (len(selected_games) != 90 or selected_games & excluded or
            len({Path(game).parts[2] for game in selected_games}) != 90):
        raise ValueError('Changed path-blind held-out coverage')
    directory_overlap = dict(Counter(
        row['family'] for order in orders for row in order['targets']
        if row['task_directory_previously_exposed']))
    return {'protocol': 'Precommitted 90 path-unexposed official ALFWorld train games, three disjoint 30-game on-policy confirmation orders, six families times five per order; five families prefer unseen task directories, look-at-object only path-blind because all its directories were already locally exposed; not official valid_unseen',
        'salt': SALT,
        'exposure_snapshot_sha256': file_hash(args.snapshot),
        'exposure_count': len(excluded),
        'inventory_count': len(inventory),
        'unexposed_count_by_family': remaining_by_family,
        'selected_previously_exposed_directory_by_family': directory_overlap,
        'orders': 3,
        'per_family_per_order': 5,
        'order_plans': orders}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'check'))
    parser.add_argument('--data-root', type=Path, default=Path(
        'ttcl/data/alfworld_delta'))
    parser.add_argument('--snapshot', type=Path, default=Path(
        'data/annotations/alf_fresh_train90_v21_exposure_snapshot_20261008.json'))
    parser.add_argument('--plan', type=Path, default=Path(
        'ttcl/trajectory_hyperlora/alf_fresh_train90_v21_plan.json'))
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.snapshot.exists() or args.plan.exists():
            raise FileExistsError('Use fresh exposure snapshot and target plan')
        snapshot = {'salt': SALT, 'scan_roots': list(ROOTS),
                    'games': scan_exposure()}
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot.write_text(json.dumps(snapshot, ensure_ascii=False,
                                            indent=2)+'\n')
        plan = expected(args, snapshot)
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        args.plan.write_text(json.dumps(plan, ensure_ascii=False,
                                        indent=2)+'\n')
    else:
        snapshot = json.loads(args.snapshot.read_text())
        plan = json.loads(args.plan.read_text())
        if plan != expected(args, snapshot):
            raise ValueError('Changed held-out target or exposure snapshot')
    print(json.dumps({'orders': 3, 'targets': 90,
        'exposure_snapshot_sha256': file_hash(args.snapshot),
        'plan_sha256': file_hash(args.plan),
        'directory_overlap': plan['selected_previously_exposed_directory_by_family']}),
        flush=True)


if __name__ == '__main__':
    main()
