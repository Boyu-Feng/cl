"""Freeze full-prefix LoRA write counterfactuals beyond the second success.

The v14 pool already covers each sequence's 1->2 prefix transition. This
manifest freezes the four later transitions without selecting on outcomes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash


PREFIX_TRANSITIONS = (
    {'sequence': 0, 'old_source_ids': [0, 1], 'new_source_id': 2},
    {'sequence': 0, 'old_source_ids': [0, 1, 2], 'new_source_id': 3},
    {'sequence': 0, 'old_source_ids': [0, 1, 2, 3], 'new_source_id': 7},
    {'sequence': 6, 'old_source_ids': [4, 5], 'new_source_id': 6},
)


def expected(args):
    pool = json.loads(args.pool_review.read_text())
    if (pool['selection_seed'] != 'incremental-v14' or
            len(pool['sources']) != 8 or len(pool['targets']) != 78 or
            pool['checkpoint_sha256'] != file_hash(args.checkpoint) or
            pool['residual_sha256'] != file_hash(args.residual)):
        raise ValueError('Changed frozen v14 source/target pool')
    sources = pool['sources']
    for transition in PREFIX_TRANSITIONS:
        ids = transition['old_source_ids'] + [transition['new_source_id']]
        rows = [sources[source_id] for source_id in ids]
        if (any(row['source_id'] != source_id
                for row, source_id in zip(rows, ids, strict=True)) or
                any(row['sequence'] != transition['sequence'] for row in rows) or
                [row['source_index'] for row in rows] != sorted(
                    row['source_index'] for row in rows) or
                len(set(ids)) != len(ids)):
            raise ValueError('Invalid chronological own-success prefix')
    targets = []
    for target in pool['targets']:
        if file_hash(args.data_root / target['game']) != target['game_sha256']:
            raise ValueError('Changed official target game')
        targets.append({key: target[key] for key in
            ('target_id', 'split', 'family', 'game', 'game_sha256')})
    pairs = []
    for target in targets:
        for transition_index, transition in enumerate(PREFIX_TRANSITIONS):
            old_ids = transition['old_source_ids']
            new_id = transition['new_source_id']
            old_hashes = [sources[i]['records_sha256'] for i in old_ids]
            content = {
                'method': 'cumulative_own_success_prefix_v18',
                'v14_pool_review_sha256': file_hash(args.pool_review),
                'checkpoint_sha256': file_hash(args.checkpoint),
                'residual_sha256': file_hash(args.residual),
                'target_id': target['target_id'],
                'split': target['split'],
                'target_game_sha256': target['game_sha256'],
                'transition_index': transition_index,
                'sequence': transition['sequence'],
                'old_source_ids': old_ids,
                'new_source_id': new_id,
                'old_records_sha256': old_hashes,
                'new_records_sha256': sources[new_id]['records_sha256'],
                'old_prefix_sha256': digest(old_hashes),
                'new_prefix_sha256': digest(old_hashes + [
                    sources[new_id]['records_sha256']]),
                'old_prefix_count': len(old_ids),
                'updated_prefix_count': len(old_ids) + 1,
                'new_source_weight': 1. / (len(old_ids) + 1),
                'max_steps': 50,
            }
            pairs.append({**content,
                'input_content_sha256': digest(content),
                'reviewed_target': True})
    if (len(targets) != 78 or len(pairs) != 312 or
            len({row['input_content_sha256'] for row in pairs}) != 312 or
            sum(row['split'] == 'train' for row in pairs) != 240 or
            sum(row['split'] == 'dev' for row in pairs) != 72):
        raise ValueError('Incomplete cumulative prefix schedule')
    return {
        'protocol': 'Frozen full-prefix own-success LoRA update versus freeze, four later chronological writes over v14 official train target pool; v14 already covers the two 1-to-2 prefixes; no outcome-based selection',
        'v14_pool_review_sha256': file_hash(args.pool_review),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'residual_sha256': file_hash(args.residual),
        'source_reports_sha256': sorted(set(row['report_sha256']
            for row in sources)),
        'source_records_sha256': [row['records_sha256'] for row in sources],
        'transitions': list(PREFIX_TRANSITIONS),
        'targets': targets,
        'pairs': pairs,
        'max_steps': 50,
        'max_new_tokens': 64,
        'actor_history_turns': 2,
        'loop_guard_max': 2,
    }


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    value = expected(args)
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'targets': len(value['targets']),
        'later_prefix_transitions': len(value['transitions']),
        'pairs': len(value['pairs']),
        'train_pairs': 240, 'dev_pairs': 72,
        'review_sha256': file_hash(args.review)}), flush=True)


def check(args):
    value = json.loads(args.review.read_text())
    if value != expected(args):
        raise ValueError('Changed reviewed prefix target/input binding')
    print(json.dumps({'checked_pairs': len(value['pairs']),
        'review_sha256': file_hash(args.review)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'check'))
    parser.add_argument('--pool-review', type=Path, default=Path(
        'data/annotations/alf_incremental_pool78_v14_reviewed_20261008.json'))
    parser.add_argument('--data-root', type=Path,
                        default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--checkpoint', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--residual', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_20261007.pt'))
    parser.add_argument('--review', type=Path, default=Path(
        'data/annotations/alf_incremental_prefix78_v18_reviewed_20261008.json'))
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else check)(args)
