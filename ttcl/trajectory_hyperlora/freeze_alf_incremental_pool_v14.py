"""Freeze train/dev games with a directory-disjoint dev split.

Uses the v13 pre-outcome exposure snapshot. The v13 candidate was rejected
before any rollout because some dev games shared task directories with old
local results. Keep that candidate intact and create a new reviewed manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v2 import FAMILIES
from ttcl.trajectory_hyperlora.freeze_alf_incremental_pool_v13 import family_and_directory


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    candidate = json.loads(args.candidate_review.read_text())
    excluded = candidate['excluded_game_paths']
    if (len(excluded) != 1969 or len(candidate['targets']) != 78 or
            candidate['selection_seed'] != 'incremental-v13' or
            candidate['checkpoint_sha256'] != file_hash(args.checkpoint) or
            candidate['residual_sha256'] != file_hash(args.residual)):
        raise ValueError('Changed pre-outcome exposure snapshot or source model')
    excluded_set = set(excluded)
    exposed_directories = {family_and_directory(game)[1] for game in excluded}
    inventory = sorted(str(path.relative_to(args.data_root)) for path in
        (args.data_root / 'json_2.1.1/train').rglob('game.tw-pddl'))
    if len(inventory) != 3553 or candidate['inventory_count'] != 3553:
        raise ValueError('Changed official train inventory')
    by_family = defaultdict(list)
    for game in inventory:
        if game in excluded_set:
            continue
        family, directory = family_and_directory(game)
        if family is None:
            continue
        priority = hashlib.sha256(('incremental-v14|' + game).encode()).hexdigest()
        by_family[family].append((priority, game, directory))
    targets = []
    for family in FAMILIES:
        distinct = {}
        for score, game, directory in sorted(by_family[family]):
            distinct.setdefault(directory, (score, game, directory))
        fully_new = sorted(row for row in distinct.values()
                           if row[2] not in exposed_directories)
        partly_seen = sorted(row for row in distinct.values()
                             if row[2] in exposed_directories)
        if len(fully_new) < 3 or len(distinct) < 13:
            raise ValueError(f'Insufficient clean dev or total directories: {family}')
        dev_rows = fully_new[:3]
        dev_directories = {row[2] for row in dev_rows}
        train_rows = ([row for row in fully_new if row[2] not in dev_directories] +
                      partly_seen)[:10]
        if len(train_rows) != 10:
            raise ValueError(f'Insufficient train directories: {family}')
        for split, rows in (('train', train_rows), ('dev', dev_rows)):
            for family_position, (_, game, directory) in enumerate(rows):
                targets.append({'target_id': len(targets), 'split': split,
                    'family': family, 'family_position': family_position,
                    'game': game, 'game_sha256': file_hash(args.data_root / game),
                    'directory_previously_exposed':
                        directory in exposed_directories})
    source_meta = candidate['sources']
    core = {'protocol': 'Pre-outcome frozen official train increment reward targets; dev games from completely unexposed task directories; train games prefer them; 10 train and 3 dev per family, 6 chronological source transitions',
        'candidate_review_sha256': file_hash(args.candidate_review),
        'excluded_game_paths': excluded,
        'checkpoint_sha256': file_hash(args.checkpoint),
        'residual_sha256': file_hash(args.residual),
        'parent_review_sha256': candidate['parent_review_sha256'],
        'old_review_sha256': candidate['old_review_sha256'],
        'old_report_sha256': candidate['old_report_sha256'],
        'old_audit_sha256': candidate['old_audit_sha256'],
        'selection_seed': 'incremental-v14',
        'sources': source_meta, 'transitions': candidate['transitions'],
        'targets': targets,
        'max_steps': 50, 'max_new_tokens': 64,
        'actor_history_turns': 2, 'loop_guard_max': 2}
    pairs = []
    for target in targets:
        for transition_index, (old_id, new_id) in enumerate(core['transitions']):
            content = {'method': 'incremental-v14',
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
            len({Path(row['game']).parts[2] for row in targets}) != 78 or
            any(row['directory_previously_exposed']
                for row in targets if row['split'] == 'dev')):
        raise ValueError('Incomplete or leaky frozen schedule')
    core['pairs'] = pairs
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(core, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'targets': 78, 'train': 60, 'dev': 18,
        'pairs': 468,
        'train_previous_directories': sum(row['directory_previously_exposed']
            for row in targets if row['split'] == 'train'),
        'dev_previous_directories': 0,
        'review_sha256': file_hash(args.review)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path,
                        default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--candidate-review', type=Path, default=Path(
        'data/annotations/alf_incremental_pool78_v13_reviewed_20261008.json'))
    parser.add_argument('--checkpoint', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--residual', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_20261007.pt'))
    parser.add_argument('--review', type=Path, default=Path(
        'data/annotations/alf_incremental_pool78_v14_reviewed_20261008.json'))
    prepare(parser.parse_args())
