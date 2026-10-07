"""Run disjoint reviewed source-target shards and merge with a saved prefix.

This parallelizes only this experiment's remaining pairs after a fully saved
prefix. Each shard binds the same parent base episodes, prefix, checkpoint,
reviewed pair order and actor budget. Raw shard reports remain ignored by Git.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import bounded_source_text
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v2 import save
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v3 import checked
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields


def parent_state(args, sources, targets, pairs):
    parent = json.loads(args.parent_report.read_text())
    if (parent['review_sha256'] != file_hash(args.review) or
            parent['checkpoint_sha256'] != file_hash(args.checkpoint) or
            parent['failures'] or len(parent['base']) != len(targets) or
            len(parent['pairs']) != args.prefix_pairs or
            parent['max_steps'] != 50 or
            parent['max_new_tokens'] != 64 or
            parent['actor_history_turns'] != 2 or
            parent['loop_guard_max'] != 2 or
            parent['context_tokens'] != 2048):
        raise ValueError('Changed or incomplete saved prefix')
    for target, row in zip(targets, parent['base'], strict=True):
        if (row['target_id'] != target['target_id'] or
                row['game'] != target['game'] or
                row['game_sha256'] != target['game_sha256'] or
                row['episode']['status'] != 'complete'):
            raise ValueError('Changed target base prefix')
    for binding, row in zip(pairs[:args.prefix_pairs], parent['pairs'],
                            strict=True):
        if (row['input_content_sha256'] != binding['input_content_sha256'] or
                row['episode']['status'] != 'complete' or
                row['source_id'] != binding['source_id'] or
                row['target_id'] != binding['target_id']):
            raise ValueError('Changed reviewed source prefix')
    return parent


def collect(args):
    if args.shard_output.exists():
        raise FileExistsError(args.shard_output)
    sources, targets, pairs = checked(args)
    parent = parent_state(args, sources, targets, pairs)
    if not args.prefix_pairs <= args.start < args.stop <= len(pairs):
        raise ValueError('Invalid disjoint shard interval')
    result = {'protocol': 'Disjoint continuation shard of reviewed additional official train 8x18 source-target matrix; exact old frozen actor protocol',
        'review_sha256': file_hash(args.review),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'parent_report_sha256': file_hash(args.parent_report),
        'parent_base_sha256': digest(parent['base']),
        'parent_prefix_sha256': digest(parent['pairs']),
        'prefix_pairs': args.prefix_pairs,
        'start': args.start, 'stop': args.stop,
        'max_steps': 50, 'max_new_tokens': 64,
        'actor_history_turns': 2, 'loop_guard_max': 2,
        'context_tokens': 2048, 'pairs': [], 'failures': []}
    args.shard_output.parent.mkdir(parents=True, exist_ok=True)
    save(args.shard_output, result)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != 'contextual' or
            agent.task_context_scope != 'current' or
            agent.task_pair_pooling != 'mean'):
        raise ValueError('Changed frozen actor family')
    source_cache = {}
    for index in range(args.start, args.stop):
        binding = pairs[index]
        source = sources[binding['source_id']]
        target = targets[binding['target_id']]
        if source['source_id'] not in source_cache:
            bounded, original, retained = bounded_source_text(
                tokenizer, source['records'], 2048)
            fields = contextual_text_fields(agent, tokenizer,
                bounded, args.device, 2048, pooling='both')
            source_cache[source['source_id']] = fields, original, retained
        fields, original, retained = source_cache[source['source_id']]
        episode = run_episode(agent, tokenizer,
            args.data_root / target['game'], fields, adapter=True,
            device=args.device, max_steps=50, max_new_tokens=64,
            constrain_actions=True, actor_history_turns=2,
            loop_guard_max=2)
        result['pairs'].append({'source_id': source['source_id'],
            'target_id': target['target_id'],
            'input_content_sha256': binding['input_content_sha256'],
            'source_tokens_original': original,
            'source_tokens_retained': retained,
            'episode': episode})
        if (episode['status'] != 'complete' or
                episode['initial_observation'] !=
                    parent['base'][target['target_id']]['episode']['initial_observation']):
            result['failures'].append({'pair_index': index,
                'reason': episode.get('error')})
        save(args.shard_output, result)
        print(json.dumps({'start': args.start,
            'done': len(result['pairs']), 'stop': args.stop,
            'reward': episode.get('reward'),
            'failures': len(result['failures'])}), flush=True)
        if result['failures']:
            raise RuntimeError('Shard failure retained for diagnosis')


def merge(args):
    if args.merged_output.exists():
        raise FileExistsError(args.merged_output)
    sources, targets, pairs = checked(args)
    parent = parent_state(args, sources, targets, pairs)
    expected_start = args.prefix_pairs
    assembled = list(parent['pairs'])
    for shard_path in args.shards:
        shard = json.loads(shard_path.read_text())
        if (shard['review_sha256'] != file_hash(args.review) or
                shard['checkpoint_sha256'] != file_hash(args.checkpoint) or
                shard['parent_report_sha256'] !=
                    file_hash(args.parent_report) or
                shard['parent_base_sha256'] != digest(parent['base']) or
                shard['parent_prefix_sha256'] != digest(parent['pairs']) or
                shard['prefix_pairs'] != args.prefix_pairs or
                shard['start'] != expected_start or
                shard['stop'] > len(pairs) or shard['failures'] or
                len(shard['pairs']) != shard['stop']-shard['start'] or
                shard['max_steps'] != 50 or shard['max_new_tokens'] != 64 or
                shard['actor_history_turns'] != 2 or
                shard['loop_guard_max'] != 2 or
                shard['context_tokens'] != 2048):
            raise ValueError(f'Changed, incomplete or overlapping shard: {shard_path}')
        for index, row in enumerate(shard['pairs'], shard['start']):
            binding = pairs[index]
            if (row['input_content_sha256'] !=
                    binding['input_content_sha256'] or
                    row['source_id'] != binding['source_id'] or
                    row['target_id'] != binding['target_id'] or
                    row['episode']['status'] != 'complete' or
                    row['episode']['initial_observation'] !=
                        parent['base'][binding['target_id']]['episode']['initial_observation']):
                raise ValueError(f'Unreviewed shard pair {index}')
        assembled.extend(shard['pairs'])
        expected_start = shard['stop']
    if expected_start != len(pairs) or len(assembled) != len(pairs):
        raise ValueError('Shard gap or missing pair')
    parent['pairs'] = assembled
    save(args.merged_output, parent)
    print(json.dumps({'merged_pairs': len(assembled),
        'sources': len(sources), 'targets': len(targets),
        'report_sha256': file_hash(args.merged_output)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('collect', 'merge'))
    parser.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--source-report-seq0', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq0_6_20261007.json'))
    parser.add_argument('--source-report-seq6', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq6_6_20261007.json'))
    parser.add_argument('--plan', type=Path, default=Path('ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json'))
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--checkpoint-candidates', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json'))
    parser.add_argument('--warmstart-candidates', type=Path, default=Path('results/trajectory_hyperlora/alf_next_task_candidates_20261005.json'))
    parser.add_argument('--first-review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--second-review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_holdout_8x12_v3c_reviewed_20261007.json'))
    parser.add_argument('--review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_additional8x18_v3_reviewed_20261007.json'))
    parser.add_argument('--parent-report', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_additional8x18_v3_20261007.json'))
    parser.add_argument('--prefix-pairs', type=int, default=22)
    parser.add_argument('--start', type=int, default=22)
    parser.add_argument('--stop', type=int, default=64)
    parser.add_argument('--shard-output', type=Path)
    parser.add_argument('--shards', nargs='*', type=Path, default=[])
    parser.add_argument('--merged-output', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_additional8x18_v3_merged_20261007.json'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.65)
    args = parser.parse_args()
    if args.command == 'collect':
        if args.shard_output is None:
            parser.error('--shard-output required for collect')
        collect(args)
    else:
        if not args.shards:
            parser.error('--shards required for merge')
        merge(args)
