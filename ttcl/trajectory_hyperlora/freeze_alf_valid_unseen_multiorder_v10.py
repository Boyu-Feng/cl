"""Precommit three disjoint official ALFWorld valid_unseen online orders.

All 134 games have appeared in earlier local reports, so these are disjoint
evaluation orders, not previously unseen games. The manifest records that
exposure explicitly; no reward or model output enters selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.freeze_alf_fresh_seen_v6 import FAMILIES


PATTERN = re.compile(r'json_2\.1\.1/valid_unseen/[^"\s]+?game\.tw-pddl')
INVENTORY = 134
ORDERS = 3
PER_FAMILY = 5
SALT = 'alf_valid_unseen_multiorder_v10_20261007'


def family(path):
    value = path.split('/')[2].split('-')[0]
    if value not in FAMILIES:
        raise ValueError(f'Unexpected official family {value}')
    return value


def inventory(data_root):
    paths = sorted(str(path.relative_to(data_root)) for path in
        (data_root / 'json_2.1.1' / 'valid_unseen').rglob('game.tw-pddl'))
    if len(paths) != INVENTORY or len(set(paths)) != INVENTORY:
        raise ValueError('Changed official valid_unseen inventory')
    return paths


def selected(data_root):
    paths = inventory(data_root)
    by_family = {name: [] for name in FAMILIES}
    for path in paths:
        by_family[family(path)].append(path)
    for name in FAMILIES:
        if len(by_family[name]) < ORDERS * PER_FAMILY:
            raise ValueError(f'Insufficient {name} tasks for disjoint orders')
        by_family[name].sort(key=lambda path:
            (hashlib.sha256(f'{SALT}:{path}'.encode()).hexdigest(), path))
    orders = []
    for order in range(ORDERS):
        groups = {}
        for name in FAMILIES:
            start = order * PER_FAMILY
            group = by_family[name][start:start + PER_FAMILY]
            groups[name] = sorted(group, key=lambda path:
                (hashlib.sha256(f'{SALT}:{order}:{path}'.encode()).hexdigest(),
                 path))
        rows = []
        for round_id in range(PER_FAMILY):
            for family_offset in range(len(FAMILIES)):
                name = FAMILIES[(family_offset + order + round_id)
                                % len(FAMILIES)]
                path = groups[name][round_id]
                rows.append({'index': len(rows), 'round': round_id,
                    'family': name, 'game': path,
                    'game_sha256': file_hash(data_root / path)})
        orders.append({'order_id': order, 'seed': 20261007 + order,
            'targets': rows})
    flat = [row['game'] for order in orders for row in order['targets']]
    if len(flat) != 90 or len(set(flat)) != 90:
        raise ValueError('Disjoint multiorder selection failed')
    return orders


def exposure(archive):
    search = subprocess.run(['rg', '-l', 'json_2.1.1/valid_unseen/',
        str(archive), '-g', '*.json'], capture_output=True, text=True,
        check=False)
    if search.returncode != 0:
        raise RuntimeError(f'Cannot audit past reports: {search.stderr}')
    paths = sorted(Path(path) for path in search.stdout.splitlines())
    seen = set()
    for path in paths:
        seen.update(PATTERN.findall(path.read_text()))
    return paths, seen


def freeze(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    orders = selected(args.data_root)
    reports, seen = exposure(args.archive)
    if seen != set(inventory(args.data_root)):
        raise ValueError('Exposure audit no longer covers all official games')
    value = {'protocol': 'Precommitted three disjoint official valid_unseen 30-game online orders, five games per six families per order; path-hash selection only, no outcome selection; all inventory games previously named in old reports, so not new-game blind testing',
        'salt': SALT, 'inventory_count': INVENTORY,
        'orders': ORDERS, 'per_family_per_order': PER_FAMILY,
        'previously_exposed_game_count': len(seen),
        'previously_exposed_family_counts': dict(sorted(
            Counter(family(path) for path in seen).items())),
        'exposure_reports': [{'path': str(path), 'sha256': file_hash(path)}
                             for path in reports],
        'order_plans': orders}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, ensure_ascii=False,
                                      indent=2) + '\n')
    print(json.dumps({'orders': len(orders),
                      'targets_per_order': len(orders[0]['targets']),
                      'previously_exposed': len(seen),
                      'manifest_sha256': file_hash(args.output)}), flush=True)


def check(args):
    value = json.loads(args.output.read_text())
    if (value['salt'] != SALT or
            value['inventory_count'] != INVENTORY or
            value['orders'] != ORDERS or
            value['per_family_per_order'] != PER_FAMILY or
            value['previously_exposed_game_count'] != INVENTORY or
            value['order_plans'] != selected(args.data_root)):
        raise ValueError('Changed frozen multiorder manifest or official games')
    print(json.dumps({'verified_orders': ORDERS,
                      'total_disjoint_targets': 90,
                      'manifest_sha256': file_hash(args.output)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze', 'check'))
    parser.add_argument('--data-root', type=Path, default=Path(
        'ttcl/data/alfworld_delta'))
    parser.add_argument('--archive', type=Path, default=Path(
        'results/trajectory_hyperlora'))
    parser.add_argument('--output', type=Path, default=Path(
        'ttcl/trajectory_hyperlora/alf_valid_unseen_multiorder_v10_plan.json'))
    args = parser.parse_args()
    (freeze if args.command == 'freeze' else check)(args)
