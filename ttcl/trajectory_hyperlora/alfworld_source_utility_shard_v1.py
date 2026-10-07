"""Evaluate independent source-target shards, then merge by reviewed pair order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v2 import checked, save
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import bounded_source_text
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields


def collect(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    sources, targets, pairs = checked(args)
    parent = json.loads(args.parent_report.read_text())
    if (args.target_policy != 'independent12' or
            not 0 <= args.start < args.stop <= len(pairs) or
            args.start % len(targets) or args.stop % len(targets) or
            parent['review_sha256'] != file_hash(args.review) or
            parent['checkpoint_sha256'] != file_hash(args.checkpoint) or
            parent['failures'] or len(parent['base']) != len(targets)):
        raise ValueError('Changed or incomplete frozen shard input')
    base_sha = digest(parent['base'])
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != 'contextual' or
            agent.task_context_scope != 'current' or
            agent.task_pair_pooling != 'mean'):
        raise ValueError('Expected frozen task-conditioned hypernetwork')
    result = {'protocol': 'Disjoint official future-reward source-target shard; same frozen actor and budget as full matrix',
        'review_sha256': file_hash(args.review),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'base_sha256': base_sha, 'start': args.start, 'stop': args.stop,
        'max_steps': 50, 'max_new_tokens': 64,
        'actor_history_turns': 2, 'loop_guard_max': 2,
        'pairs': [], 'failures': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_cache = {}
    for index in range(args.start, args.stop):
        binding = pairs[index]
        source = sources[binding['source_id']]
        target = targets[binding['target_id']]
        if source['source_id'] not in source_cache:
            bounded, original, retained = bounded_source_text(
                tokenizer, source['records'], 2048)
            fields = contextual_text_fields(agent, tokenizer, bounded,
                args.device, 2048, pooling='both')
            source_cache[source['source_id']] = fields, original, retained
        fields, original, retained = source_cache[source['source_id']]
        episode = run_episode(agent, tokenizer,
            args.data_root / target['game'], fields, adapter=True,
            device=args.device, max_steps=50, max_new_tokens=64,
            constrain_actions=True, actor_history_turns=2, loop_guard_max=2)
        if episode['initial_observation'] != parent['base'][binding['target_id']]['episode']['initial_observation']:
            raise ValueError('Shard target reset changed')
        result['pairs'].append({'source_id': source['source_id'],
            'target_id': target['target_id'],
            'input_content_sha256': binding['input_content_sha256'],
            'source_tokens_original': original,
            'source_tokens_retained': retained,
            'episode': episode})
        if episode['status'] != 'complete':
            result['failures'].append({'pair_index': index})
        save(args.output, result)
        print(json.dumps({'start': args.start, 'done': len(result['pairs']),
            'stop': args.stop, 'failures': len(result['failures'])}), flush=True)
        if result['failures']:
            raise RuntimeError('Shard failure recorded')


def merge(args):
    sources, targets, pairs = checked(args)
    parent = json.loads(args.parent_report.read_text())
    if (args.target_policy != 'independent12' or
            parent['review_sha256'] != file_hash(args.review) or
            parent['checkpoint_sha256'] != file_hash(args.checkpoint) or
            parent['failures'] or len(parent['base']) != len(targets) or
            len(parent['pairs']) != args.start):
        raise ValueError('Primary report has unexpected prefix')
    base_sha = digest(parent['base'])
    expected_start = args.start
    for shard_path in args.shards:
        shard = json.loads(shard_path.read_text())
        if (shard['review_sha256'] != file_hash(args.review) or
                shard['checkpoint_sha256'] != file_hash(args.checkpoint) or
                shard['base_sha256'] != base_sha or shard['failures'] or
                shard['start'] != expected_start or
                len(shard['pairs']) != shard['stop'] - shard['start'] or
                shard['max_steps'] != 50 or shard['max_new_tokens'] != 64 or
                shard['actor_history_turns'] != 2 or
                shard['loop_guard_max'] != 2):
            raise ValueError(f'Changed or incomplete shard: {shard_path}')
        for index, actual in enumerate(shard['pairs'], shard['start']):
            binding = pairs[index]
            if (actual['source_id'] != binding['source_id'] or
                    actual['target_id'] != binding['target_id'] or
                    actual['input_content_sha256'] != binding['input_content_sha256'] or
                    actual['episode']['status'] != 'complete'):
                raise ValueError(f'Unreviewed shard pair {index}')
        parent['pairs'].extend(shard['pairs'])
        expected_start = shard['stop']
    if expected_start != len(pairs) or len(parent['pairs']) != len(pairs):
        raise ValueError('Shard gap or duplicate')
    save(args.merged_report, parent)
    print(json.dumps({'merged_pairs': len(parent['pairs']),
        'sources': len(sources), 'targets': len(targets),
        'output_sha256': file_hash(args.merged_report)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('collect', 'merge'))
    parser.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--source-report-seq0', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq0_6_20261007.json'))
    parser.add_argument('--source-report-seq6', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq6_6_20261007.json'))
    parser.add_argument('--plan', type=Path, default=Path('ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json'))
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--target-policy', choices=('independent12',), default='independent12')
    parser.add_argument('--checkpoint-training-candidates', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json'))
    parser.add_argument('--warmstart-candidates', type=Path, default=Path('results/trajectory_hyperlora/alf_next_task_candidates_20261005.json'))
    parser.add_argument('--prior-train-review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_holdout_8x12_v3c_reviewed_20261007.json'))
    parser.add_argument('--parent-report', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_holdout_8x12_v3c_20261007.json'))
    parser.add_argument('--merged-report', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_holdout_8x12_v3c_merged_20261007.json'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--start', type=int, default=24)
    parser.add_argument('--stop', type=int, default=48)
    parser.add_argument('--shards', nargs='*', type=Path, default=[])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.65)
    args = parser.parse_args()
    if args.command == 'collect':
        if args.output is None:
            parser.error('--output required for collect')
        collect(args)
    else:
        if not args.shards:
            parser.error('--shards required for merge')
        merge(args)


if __name__ == '__main__':
    main()
