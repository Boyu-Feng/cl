"""From-empty online ALFWorld: replace source vector after each own success.

This isolates running-mean dilution from the frozen actor and hypernetwork.
No future feedback or task-family information enters the policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_context_vector_mean_v1 import vector_hash
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash, records_from_episode
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import (
    bounded_source_text, checked_targets as checked_parent_targets,
)
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields


def parent_targets(args):
    original_review = args.review
    try:
        args.review = args.parent_persistent_review
        return checked_parent_targets(args)
    finally:
        args.review = original_review


def target_content(args, index, row):
    return {'index': index, 'game': row['game'],
        'game_sha256': file_hash(args.data_root / row['game']),
        'parent_target_sha256': row['input_content_sha256'],
        'parent_review_sha256': file_hash(args.parent_persistent_review),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'method': f'online_{args.policy}_own_success_contextual_vector'}


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    parent = parent_targets(args)
    targets = []
    for index, row in enumerate(parent):
        content = target_content(args, index, row)
        targets.append({**content, 'input_content_sha256': digest(content),
            'reviewed_target': True,
            'review_basis': 'Fresh content-bound train target; only own completed successful live trajectory may update source'})
    value = {'protocol': f'Reviewed from-empty {args.policy}-success contextual LoRA update',
        'parent_review_sha256': file_hash(args.parent_persistent_review),
        'checkpoint_sha256': file_hash(args.checkpoint), 'targets': targets}
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'reviewed_targets': len(targets)}), flush=True)


def checked_targets(args):
    parent = parent_targets(args)
    review = json.loads(args.review.read_text())
    if (review['parent_review_sha256'] != file_hash(args.parent_persistent_review) or
            review['checkpoint_sha256'] != file_hash(args.checkpoint) or
            len(review['targets']) != len(parent)):
        raise ValueError('Changed latest-success target review lineage')
    for index, row in enumerate(parent):
        content = target_content(args, index, row)
        approved = review['targets'][index]
        if (approved['reviewed_target'] is not True or
                approved['input_content_sha256'] != digest(content) or
                any(approved[key] != val for key, val in content.items())):
            raise ValueError('Changed latest-success target content')
    return review['targets']


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    targets = checked_targets(args)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != 'contextual' or not agent.task_conditioned or
            agent.task_context_scope != 'current' or
            agent.task_pair_pooling != 'mean'):
        raise ValueError('Expected frozen task-conditioned hypernetwork')
    fields = None
    prior = []
    updates = 0
    latest_source = None
    report = {'protocol': f'From-empty, frozen actor and hypernetwork; {args.policy} own official success source vector; no replay of old text, no task-family routing; 50 steps/64 tokens/2-turn history/constrained greedy/two-repeat loop guard',
        'policy': args.policy,
        'review_sha256': file_hash(args.review),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'context_tokens': args.context_tokens, 'max_steps': 50,
        'max_new_tokens': 64, 'actor_history_turns': 2,
        'loop_guard_max': 2, 'games': [], 'failures': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for row in targets:
        before = vector_hash(fields)
        episode = run_episode(agent, tokenizer,
            args.data_root / row['game'], fields or {},
            adapter=fields is not None, device=args.device,
            max_steps=50, max_new_tokens=64, constrain_actions=True,
            actor_history_turns=2, loop_guard_max=2)
        entry = {'game': row['game'],
            'input_content_sha256': row['input_content_sha256'],
            'prior_episode_count': len(prior),
            'prior_episodes_sha256': digest(prior),
            'source_vector_sha256_before': before,
            'latest_source_game_before': latest_source,
            'episode': episode}
        if episode['status'] == 'complete':
            try:
                records = records_from_episode(episode)
                entry['new_records_sha256'] = digest(records)
                entry['memory_updated'] = bool(episode['reward']) and (
                    args.policy == 'latest' or fields is None)
                if entry['memory_updated']:
                    source_text, original, retained = bounded_source_text(
                        tokenizer, records, args.context_tokens)
                    fields = contextual_text_fields(agent, tokenizer,
                        source_text, args.device, args.context_tokens,
                        pooling='both')
                    updates += 1
                    latest_source = row['game']
                    entry.update({'source_tokens_original': original,
                                  'source_tokens_retained': retained})
                prior.append({'game': row['game'], 'reward': episode['reward'],
                              'records': records})
            except Exception as exc:
                report['failures'].append({'game': row['game'],
                    'error': f'{type(exc).__name__}: {exc}'})
        else:
            report['failures'].append({'game': row['game'],
                'error': episode.get('error', 'incomplete rollout')})
        entry['source_vector_sha256_after'] = vector_hash(fields)
        entry['update_count_after'] = updates
        entry['latest_source_game_after'] = latest_source
        report['games'].append(entry)
        report['summary'] = {'n': len(report['games']),
            'success': sum(x['episode'].get('reward', 0) for x in report['games']),
            'failures': len(report['failures'])}
        temp = args.output.with_suffix(args.output.suffix + '.tmp')
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        temp.replace(args.output)
        print(json.dumps(report['summary']), flush=True)
        if report['failures']:
            raise RuntimeError('Latest-success online failure recorded')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'evaluate'))
    parser.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--plan', type=Path, default=Path('ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json'))
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--parent-review', type=Path, default=Path('data/annotations/alfworld_online_from_empty_train_seq6_6_reviewed_20261007.json'))
    parser.add_argument('--parent-persistent-review', type=Path, default=Path('data/annotations/alfworld_online_context_persistent_train_seq6_6_reviewed_20261007.json'))
    parser.add_argument('--review', type=Path, default=Path('data/annotations/alfworld_online_latest_success_train_seq6_6_reviewed_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_online_latest_success_train_seq6_6_20261007.json'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--policy', choices=('latest', 'first'), default='latest')
    parser.add_argument('--gpu-fraction', type=float, default=.65)
    parser.add_argument('--context-tokens', type=int, default=2048)
    args = parser.parse_args()
    if args.context_tokens < 2:
        parser.error('Invalid source token budget')
    (prepare if args.command == 'prepare' else evaluate)(args)

if __name__ == '__main__':
    main()
