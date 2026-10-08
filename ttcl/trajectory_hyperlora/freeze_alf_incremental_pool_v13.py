"""Freeze fresh official-train targets for incremental LoRA reward learning.

This only chooses games; it never reads outcome labels. The local exposure
snapshot is retained with the ignored annotation so later results cannot
silently change which targets were considered fresh at selection time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v2 import FAMILIES
from ttcl.trajectory_hyperlora.collect_alf_incremental_update_v9 import TRANSITIONS


GAME_PATTERN = r'json_2\.1\.1/train/[^"[:space:]]+/game\.tw-pddl'


def exposed_games(roots: list[Path]) -> list[str]:
    command = ['rg', '--no-ignore', '-o', '--no-filename',
               '--glob', '*.json', GAME_PATTERN, *(str(root) for root in roots)]
    scan = subprocess.run(command, capture_output=True, text=True,
                          check=False)
    if scan.returncode not in (0, 1) or scan.stderr:
        raise RuntimeError(f'Exposure scan failed: {scan.stderr}')
    values = sorted(set(scan.stdout.splitlines()))
    if len(values) < 1900:
        raise ValueError('Exposure scan unexpectedly small')
    return values


def family_and_directory(game: str):
    parts = Path(game).parts
    if len(parts) != 5 or parts[:2] != ('json_2.1.1', 'train') or \
            parts[-1] != 'game.tw-pddl':
        raise ValueError(f'Unexpected official train path: {game}')
    family = next((name for name in FAMILIES if parts[2].startswith(name + '-')),
                  None)
    return family, parts[2]


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    roots = [Path('results'), Path('data/annotations'), Path('ttcl/results')]
    if any(not root.exists() for root in roots):
        raise FileNotFoundError('Exposure scan root missing')
    excluded = exposed_games(roots)
    excluded_set = set(excluded)
    inventory = sorted(str(path.relative_to(args.data_root)) for path in
        (args.data_root / 'json_2.1.1/train').rglob('game.tw-pddl'))
    if len(inventory) != 3553:
        raise ValueError('Changed public ALFWorld train inventory')
    parent = json.loads(args.parent_review.read_text())
    old = json.loads(args.old_review.read_text())
    if (len(parent['sources']) != 8 or len(old['targets']) != 108 or
            file_hash(args.old_report) !=
                json.loads(args.old_audit.read_text())['raw_report_sha256']):
        raise ValueError('Changed old own-success source lineage')
    old_targets = {row['game'] for row in old['targets']}
    source_games = {row['game'] for row in parent['sources']}
    if old_targets - excluded_set or source_games - excluded_set:
        raise ValueError('Exposure scan missed previous source or target')
    family_candidates = defaultdict(list)
    for game in inventory:
        if game in excluded_set:
            continue
        family, directory = family_and_directory(game)
        if family is None:
            continue
        score = hashlib.sha256(('incremental-v13|' + game).encode()).hexdigest()
        family_candidates[family].append((score, game, directory))
    targets = []
    for family in FAMILIES:
        used_directories = set()
        chosen = []
        for _, game, directory in sorted(family_candidates[family]):
            if directory not in used_directories:
                chosen.append(game)
                used_directories.add(directory)
            if len(chosen) == 13:
                break
        if len(chosen) != 13:
            raise ValueError(f'Insufficient unexposed directories: {family}')
        for family_position, game in enumerate(chosen):
            split = 'train' if family_position < 10 else 'dev'
            targets.append({'target_id': len(targets), 'split': split,
                'family': family, 'family_position': family_position,
                'game': game, 'game_sha256': file_hash(args.data_root / game)})
    source_meta = [{key: source[key] for key in
        ('source_id', 'sequence', 'source_index', 'game', 'game_sha256',
         'records_sha256', 'report_sha256')}
        for source in parent['sources']]
    core = {'protocol': 'Frozen official train incremental-update target pool; deterministic path hash, one game per task directory, 10 train and 3 dev per family; excludes all locally exposed JSON game paths at selection time',
        'parent_review_sha256': file_hash(args.parent_review),
        'old_review_sha256': file_hash(args.old_review),
        'old_report_sha256': file_hash(args.old_report),
        'old_audit_sha256': file_hash(args.old_audit),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'residual_sha256': file_hash(args.residual),
        'inventory_count': len(inventory),
        'excluded_game_paths': excluded,
        'exposure_scan_roots': [str(root) for root in roots],
        'selection_seed': 'incremental-v13',
        'sources': source_meta,
        'transitions': [list(pair) for pair in TRANSITIONS],
        'targets': targets,
        'max_steps': 50, 'max_new_tokens': 64,
        'actor_history_turns': 2, 'loop_guard_max': 2}
    pairs = []
    for target in targets:
        for transition_index, (old_id, new_id) in enumerate(TRANSITIONS):
            content = {'method': 'incremental-v13',
                'target_id': target['target_id'], 'split': target['split'],
                'target_game_sha256': target['game_sha256'],
                'transition_index': transition_index,
                'old_records_sha256': source_meta[old_id]['records_sha256'],
                'new_records_sha256': source_meta[new_id]['records_sha256'],
                'checkpoint_sha256': core['checkpoint_sha256'],
                'residual_sha256': core['residual_sha256'],
                'alpha': [0., .5], 'max_steps': 50}
            pairs.append({**content, 'input_content_sha256': digest(content),
                          'reviewed_target': True})
    if (len(targets) != 78 or len(pairs) != 468 or
            len({row['game'] for row in targets}) != 78 or
            len({row['input_content_sha256'] for row in pairs}) != 468):
        raise ValueError('Incomplete target or pair schedule')
    core['pairs'] = pairs
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(core, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'targets': 78, 'train': 60, 'dev': 18,
        'pairs': 468, 'excluded': len(excluded),
        'review_sha256': file_hash(args.review)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path,
                        default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--parent-review', type=Path, default=Path(
        'data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--old-review', type=Path, default=Path(
        'data/annotations/alf_incremental_update108_v9_reviewed_20261007.json'))
    parser.add_argument('--old-report', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_incremental_update108_v9_20261007.json'))
    parser.add_argument('--old-audit', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_incremental_update108_v9_audited_20261007.json'))
    parser.add_argument('--checkpoint', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--residual', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_20261007.pt'))
    parser.add_argument('--review', type=Path, default=Path(
        'data/annotations/alf_incremental_pool78_v13_reviewed_20261008.json'))
    prepare(parser.parse_args())
