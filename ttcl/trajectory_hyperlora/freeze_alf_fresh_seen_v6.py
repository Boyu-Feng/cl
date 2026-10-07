"""Freeze a fresh official valid_seen task order before later LoRA evaluation.

The selection excludes games named in the existing trajectory_hyperlora JSON
archive. This manifest is a public experiment plan, not an outcome report.
New trajectory collection must additionally create content-bound annotations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash


PATTERN = re.compile(r'json_2\.1\.1/valid_seen/[^"\s]+?game\.tw-pddl')
FAMILIES = (
    'look_at_obj_in_light',
    'pick_and_place_simple',
    'pick_clean_then_place_in_recep',
    'pick_cool_then_place_in_recep',
    'pick_heat_then_place_in_recep',
    'pick_two_obj_and_place',
)


def family(path: str) -> str:
    value = path.split('/')[2].split('-')[0]
    if value not in FAMILIES:
        raise ValueError(f'Unknown official task family: {value}')
    return value


def select(data_root: Path, excluded: set[str], per_family: int) -> list[dict]:
    root = data_root / 'json_2.1.1' / 'valid_seen'
    candidates = [str(path.relative_to(data_root))
                  for path in root.rglob('game.tw-pddl')]
    if len(candidates) != 140 or len(set(candidates)) != 140:
        raise ValueError('Changed official valid_seen inventory')
    if not excluded.issubset(set(candidates)):
        raise ValueError('Previously exposed task missing from inventory')
    by_family = {}
    for name in FAMILIES:
        remaining = (path for path in candidates
                     if path not in excluded and family(path) == name)
        by_family[name] = sorted(
            remaining,
            key=lambda path: (hashlib.sha256(path.encode()).hexdigest(), path))
        if len(by_family[name]) < per_family:
            raise ValueError(f'Too few unused tasks in {name}')
    rows = []
    for round_id in range(per_family):
        for name in FAMILIES:
            path = by_family[name][round_id]
            rows.append({'index': len(rows), 'round': round_id,
                         'family': name, 'game': path,
                         'game_sha256': file_hash(data_root / path)})
    return rows


def freeze(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    search = subprocess.run(
        ['rg', '-l', 'json_2.1.1/valid_seen/', str(args.archive),
         '-g', '*.json'], capture_output=True, text=True, check=False)
    if search.returncode != 0:
        raise RuntimeError(f'Cannot inspect prior JSON reports: {search.stderr}')
    reports = [Path(line) for line in search.stdout.splitlines()]
    excluded = set()
    for path in reports:
        excluded.update(PATTERN.findall(path.read_text()))
    rows = select(args.data_root, excluded, args.per_family)
    manifest = {
        'protocol': 'Precommitted official valid_seen targets for later online trajectory-to-LoRA comparison; exclude paths previously present in trajectory_hyperlora JSON reports, sort remaining paths by SHA-256 per family, interleave six families; no new model outcome read',
        'data_root_layout': 'json_2.1.1/valid_seen',
        'per_family': args.per_family,
        'inventory_count': 140,
        'exposed_count': len(excluded),
        'exposed_family_counts': dict(sorted(Counter(map(family, excluded)).items())),
        'exposure_reports': [{'path': str(path), 'sha256': file_hash(path)}
                             for path in sorted(reports)],
        'excluded_games': sorted(excluded),
        'targets': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False,
                                      indent=2) + '\n')
    print(json.dumps({'selected': len(rows), 'exposed': len(excluded),
                      'reports': len(reports),
                      'families': dict(Counter(row['family'] for row in rows))}),
          flush=True)


def check(args):
    manifest = json.loads(args.output.read_text())
    excluded = set(manifest['excluded_games'])
    rows = select(args.data_root, excluded, manifest['per_family'])
    if (manifest['inventory_count'] != 140 or
            manifest['exposed_count'] != len(excluded) or
            manifest['targets'] != rows or
            len(manifest['exposure_reports']) == 0 or
            len(rows) != 6 * manifest['per_family'] or
            len({row['game'] for row in rows}) != len(rows)):
        raise ValueError('Frozen target manifest or official task content changed')
    print(json.dumps({'manifest_sha256': file_hash(args.output),
                      'verified_targets': len(rows),
                      'excluded_games': len(excluded)}), flush=True)


def expected_review(args):
    manifest = json.loads(args.output.read_text())
    rows = select(args.data_root, set(manifest['excluded_games']),
                  manifest['per_family'])
    if rows != manifest['targets'] or len(rows) != 36:
        raise ValueError('Changed frozen official target plan')
    common = {'plan_sha256': file_hash(args.output),
              'checkpoint_sha256': file_hash(args.checkpoint),
              'max_steps': 50, 'max_new_tokens': 64,
              'actor_history_turns': 2, 'loop_guard_max': 2,
              'history_policy': 'source-free start, only own completed episodes eligible for future memory; no future target trajectory in current actor input'}
    return {'protocol': 'New content-bound annotation targets for frozen fresh official valid_seen online trajectory-to-LoRA comparison; no outcome-derived target selection',
            **common,
            'targets': [{**row, 'input_content_sha256': digest({**row, **common}),
                         'reviewed_target': True}
                        for row in rows]}


def prepare_review(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    review = expected_review(args)
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(review, ensure_ascii=False,
                                      indent=2) + '\n')
    print(json.dumps({'reviewed_targets': len(review['targets']),
                      'review_sha256': file_hash(args.review)}), flush=True)


def check_review(args):
    review = json.loads(args.review.read_text())
    if review != expected_review(args):
        raise ValueError('Changed target annotation or input-content binding')
    print(json.dumps({'reviewed_targets': len(review['targets']),
                      'review_sha256': file_hash(args.review)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze', 'check',
                                            'prepare_review', 'check_review'))
    parser.add_argument('--data-root', type=Path,
                        default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--archive', type=Path,
                        default=Path('results/trajectory_hyperlora'))
    parser.add_argument('--output', type=Path,
                        default=Path('ttcl/trajectory_hyperlora/alf_fresh_seen36_v6_plan.json'))
    parser.add_argument('--per-family', type=int, default=6)
    parser.add_argument('--checkpoint', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--review', type=Path, default=Path(
        'data/annotations/alf_fresh_seen36_online_v6_reviewed_20261007.json'))
    args = parser.parse_args()
    if args.per_family < 1:
        parser.error('per-family must be positive')
    {'freeze': freeze, 'check': check,
     'prepare_review': prepare_review,
     'check_review': check_review}[args.command](args)
