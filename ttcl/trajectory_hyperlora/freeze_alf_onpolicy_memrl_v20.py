"""Freeze independent native MemRL on-policy targets on v19 official orders."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.evaluate_xland_memrl import plan as native_plan


def expected(args):
    parent = json.loads(args.parent_review.read_text())
    if (parent['plan_sha256'] != file_hash(args.plan) or
            parent['orders'] != 3 or parent['targets_per_order'] != 30 or
            len(parent['targets']) != 360 or
            args.memory_tokens != 2048 or args.writer_tokens != 256 or
            args.retrieval_threshold != .5 or args.seed != 42):
        raise ValueError('Changed frozen on-policy target/memory budget')
    memory_plan = native_plan(args)
    common = {'protocol': 'alf_onpolicy_memrl_v20',
        'parent_review_sha256': file_hash(args.parent_review),
        'plan_sha256': file_hash(args.plan),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'memory_plan_sha256': digest(memory_plan),
        'memory_adapter_sha256': file_hash(Path(
            'ttcl/memrl_comparison/memory.py')),
        'memory_tokens': 2048, 'writer_tokens': 256,
        'retrieval_threshold': .5, 'seed': 42,
        'max_steps': 50, 'max_new_tokens': 64,
        'actor_history_turns': 2, 'loop_guard_max': 2,
        'memory_eligibility': 'own official-won episodes only'}
    targets = []
    for order in range(3):
        base = [row for row in parent['targets']
                if row['order_id'] == order and row['method'] == 'base']
        if len(base) != 30:
            raise ValueError('Missing v19 parent targets')
        for position, row in enumerate(base):
            if (row['position'] != position or
                    file_hash(args.data_root / row['game']) !=
                    row['game_sha256']):
                raise ValueError('Changed official game content/order')
            content = {**common, 'order_id': order, 'position': position,
                'family': row['family'], 'game': row['game'],
                'game_sha256': row['game_sha256'],
                'parent_target_binding': row['input_content_sha256']}
            targets.append({**content,
                'input_content_sha256': digest(content),
                'reviewed_target': True})
    if len(targets) != 90:
        raise ValueError('Incomplete native MemRL on-policy target binding')
    return {'protocol': 'Fresh content-bound native MemRL method-on-policy review: same three official v19 orders, independent empty store and own successful trajectories per order; all games previously exposed in local reports',
        'parent_review_sha256': file_hash(args.parent_review),
        'plan_sha256': file_hash(args.plan),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'memory_plan': memory_plan,
        'memory_plan_sha256': digest(memory_plan),
        'targets': targets}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'check'))
    parser.add_argument('--model', type=Path, default=Path(
        'current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--plan', type=Path, default=Path(
        'ttcl/trajectory_hyperlora/alf_valid_unseen_multiorder_v10_plan.json'))
    parser.add_argument('--parent-review', type=Path, default=Path(
        'data/annotations/alf_onpolicy_multiorder_v19_reviewed_20261008.json'))
    parser.add_argument('--data-root', type=Path, default=Path(
        'ttcl/data/alfworld_delta'))
    parser.add_argument('--upstream', type=Path, default=Path(
        'current_work/MemRL'))
    parser.add_argument('--embedding', type=Path, default=Path(
        'models/embedding/bge-m3'))
    parser.add_argument('--memory-tokens', type=int, default=2048)
    parser.add_argument('--writer-tokens', type=int, default=256)
    parser.add_argument('--retrieval-threshold', type=float, default=.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--review', type=Path, default=Path(
        'data/annotations/alf_onpolicy_memrl_v20_reviewed_20261008.json'))
    args = parser.parse_args()
    value = expected(args)
    if args.command == 'prepare':
        if args.review.exists():
            raise FileExistsError(args.review)
        args.review.parent.mkdir(parents=True, exist_ok=True)
        args.review.write_text(json.dumps(value, ensure_ascii=False,
                                          indent=2) + '\n')
    elif json.loads(args.review.read_text()) != value:
        raise ValueError('Changed native MemRL on-policy annotation')
    print(json.dumps({'bindings': len(value['targets']),
        'orders': 3, 'review_sha256': file_hash(args.review)}), flush=True)


if __name__ == '__main__':
    main()
