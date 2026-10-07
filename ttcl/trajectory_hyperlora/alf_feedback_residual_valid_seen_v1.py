"""Freeze an untouched official valid_seen target set for residual evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import copy
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v2 import (
    FAMILIES, checked as checked_training,
)


def expected(args):
    training_args = copy(args)
    training_args.review = args.training_review
    training_args.target_policy = 'train18'
    sources, _, _ = checked_training(training_args)
    source = sources[args.source_id]
    root = args.data_root / 'json_2.1.1' / 'valid_seen'
    targets = []
    for family in FAMILIES:
        candidates = sorted(root.glob(f'{family}-*/*/game.tw-pddl'))
        if len(candidates) < 2:
            raise ValueError(f'Missing official valid_seen targets: {family}')
        ordered = sorted(candidates, key=lambda path: hashlib.sha256(
            f'{args.seed}|{path.relative_to(args.data_root)}'.encode()
            ).hexdigest())
        for path in ordered[:2]:
            game = str(path.relative_to(args.data_root))
            content = {'method': 'feedback_residual_valid_seen12_v1',
                'game': game, 'game_sha256': file_hash(path),
                'family': family,
                'source_id': args.source_id,
                'source_records_sha256': source['records_sha256'],
                'training_review_sha256': file_hash(args.training_review),
                'checkpoint_sha256': file_hash(args.checkpoint),
                'selection_seed': args.seed,
                'max_steps': 50, 'max_new_tokens': 64,
                'actor_history_turns': 2, 'loop_guard_max': 2}
            targets.append({**content,
                'input_content_sha256': digest(content),
                'reviewed_target': True})
    core = {'protocol': 'Frozen official valid_seen 12-task set: two seed-hash-selected games per family; no environment reward used in selection; one reviewed own-success training source',
        'selection_seed': args.seed,
        'source_id': args.source_id,
        'source_game': source['game'],
        'source_records_sha256': source['records_sha256'],
        'checkpoint_sha256': file_hash(args.checkpoint),
        'training_review_sha256': file_hash(args.training_review),
        'targets': targets}
    return core, source


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    core, _ = expected(args)
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(core, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'targets': len(core['targets']),
        'review_sha256': file_hash(args.review)}), flush=True)


def checked(args):
    core, source = expected(args)
    if json.loads(args.review.read_text()) != core:
        raise ValueError('Changed or unreviewed valid_seen target content')
    return core['targets'], source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'check'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--source-report-seq0', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq0_6_20261007.json'))
    parser.add_argument('--source-report-seq6', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq6_6_20261007.json'))
    parser.add_argument('--plan', type=Path, default=Path('ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json'))
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--checkpoint-training-candidates', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json'))
    parser.add_argument('--warmstart-candidates', type=Path, default=Path('results/trajectory_hyperlora/alf_next_task_candidates_20261005.json'))
    parser.add_argument('--prior-train-review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--training-review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--review', type=Path, default=Path('data/annotations/alf_feedback_residual_valid_seen12_reviewed_20261007.json'))
    parser.add_argument('--source-id', type=int, choices=range(8), default=3)
    parser.add_argument('--seed', type=int, default=20261007)
    args = parser.parse_args()
    result = prepare(args) if args.command == 'prepare' else checked(args)
    if args.command == 'check':
        print(json.dumps({'targets': len(result[0]),
            'review_sha256': file_hash(args.review)}), flush=True)


if __name__ == '__main__':
    main()
