"""Exploratory valid_unseen replay of latest own-success source-vector LoRA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_context_vector_mean_v1 import vector_hash
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash, records_from_episode
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import bounded_source_text
from ttcl.trajectory_hyperlora.alfworld_online_reward_gate_valid_unseen_v1 import checked_targets as checked_parent_targets
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields


def parents(args):
    original = args.review
    try:
        args.review = args.parent_online_review
        return checked_parent_targets(args)
    finally:
        args.review = original


def target_content(args, index, row):
    return {'index': index, 'game': row['game'],
        'family': row['family'],
        'game_sha256': file_hash(args.data_root / row['game']),
        'parent_target_sha256': row['input_content_sha256'],
        'parent_review_sha256': file_hash(args.parent_online_review),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'method': 'from_empty_latest_own_success_contextual_vector_valid_unseen'}


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    old = parents(args)
    targets = []
    for index, row in enumerate(old):
        content = target_content(args, index, row)
        targets.append({**content, 'input_content_sha256': digest(content),
            'reviewed_target': True,
            'review_basis': 'Fresh content binding to frozen official target; only future own official success updates latest source'})
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps({'protocol': 'Fresh content-bound latest-success official valid_unseen exploratory target review',
        'parent_review_sha256': file_hash(args.parent_online_review),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'per_family': args.per_family, 'targets': targets},
        ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'reviewed_targets': len(targets)}), flush=True)


def checked_targets(args):
    parent = parents(args)
    review = json.loads(args.review.read_text())
    if (review['parent_review_sha256'] != file_hash(args.parent_online_review) or
            review['checkpoint_sha256'] != file_hash(args.checkpoint) or
            review['per_family'] != args.per_family or
            len(review['targets']) != len(parent)):
        raise ValueError('Changed valid_unseen latest source target lineage')
    for index, row in enumerate(parent):
        content = target_content(args, index, row)
        approved = review['targets'][index]
        if (approved['reviewed_target'] is not True or
                approved['input_content_sha256'] != digest(content) or
                any(approved[key] != value for key, value in content.items())):
            raise ValueError('Changed valid_unseen latest target content')
    return review['targets']


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    targets = checked_targets(args)
    reference = json.loads(args.reference.read_text())
    if (reference['review_sha256'] != file_hash(args.parent_online_review) or
            reference['checkpoint_sha256'] != file_hash(args.checkpoint) or
            reference['failures'] or len(reference['games']) != len(targets)):
        raise ValueError('Changed previous mean/base pairing')
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != 'contextual' or
            agent.task_context_scope != 'current' or
            agent.task_pair_pooling != 'mean'):
        raise ValueError('Expected frozen task-conditioned hypernetwork')
    fields = None
    updates = 0
    latest_source = None
    prior = []
    result = {'protocol': 'Exploratory previously used official valid_unseen 36; from-empty latest own-success contextual LoRA, frozen model, same base and mean reference, 50-step/64-token/2-history/constrained greedy/loop guard2',
        'review_sha256': file_hash(args.review),
        'reference_sha256': file_hash(args.reference),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'max_steps': 50, 'max_new_tokens': 64,
        'actor_history_turns': 2, 'loop_guard_max': 2,
        'context_tokens': args.context_tokens,
        'games': [], 'failures': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(targets):
        old = reference['games'][index]
        if old['game'] != row['game']:
            raise ValueError('Frozen valid_unseen order changed')
        before = vector_hash(fields)
        episode = run_episode(agent, tokenizer,
            args.data_root / row['game'], fields or {},
            adapter=fields is not None, device=args.device,
            max_steps=50, max_new_tokens=64,
            constrain_actions=True, actor_history_turns=2,
            loop_guard_max=2)
        entry = {'game': row['game'], 'family': row['family'],
            'input_content_sha256': row['input_content_sha256'],
            'prior_episode_count': len(prior),
            'prior_episodes_sha256': digest(prior),
            'source_vector_sha256_before': before,
            'latest_source_game_before': latest_source,
            'episode': episode}
        if episode['status'] == 'complete':
            try:
                if episode['initial_observation'] != old['base']['initial_observation']:
                    raise ValueError('Changed target reset')
                records = records_from_episode(episode)
                entry['new_records_sha256'] = digest(records)
                entry['memory_updated'] = bool(episode['reward'])
                if episode['reward']:
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
                result['failures'].append({'game': row['game'],
                    'error': f'{type(exc).__name__}: {exc}'})
        else:
            result['failures'].append({'game': row['game'],
                'error': episode.get('error', 'incomplete rollout')})
        entry['source_vector_sha256_after'] = vector_hash(fields)
        entry['latest_source_game_after'] = latest_source
        entry['update_count_after'] = updates
        result['games'].append(entry)
        result['summary'] = {'n': len(result['games']),
            'base': sum(x['base']['reward'] for x in reference['games'][:len(result['games'])]),
            'old_mean': sum(x['online']['reward'] for x in reference['games'][:len(result['games'])]),
            'latest': sum(x['episode'].get('reward', 0) for x in result['games']),
            'failures': len(result['failures'])}
        temp = args.output.with_suffix(args.output.suffix + '.tmp')
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        temp.replace(args.output)
        print(json.dumps(result['summary']), flush=True)
        if result['failures']:
            raise RuntimeError('Latest valid_unseen failure recorded')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'evaluate'))
    parser.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--parent-review', type=Path, default=Path('data/annotations/alf_memrl134_hyperlora_train_sources_reviewed_20261006.json'))
    parser.add_argument('--parent-online-review', type=Path, default=Path('data/annotations/alf_online_reward_gate_valid_unseen_36_reviewed_20261007.json'))
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--review', type=Path, default=Path('data/annotations/alf_online_latest_success_valid_unseen36_reviewed_20261007.json'))
    parser.add_argument('--reference', type=Path, default=Path('results/trajectory_hyperlora/alf_online_reward_gate_valid_unseen_36_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_online_latest_success_valid_unseen36_20261007.json'))
    parser.add_argument('--per-family', type=int, default=6)
    parser.add_argument('--context-tokens', type=int, default=2048)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.65)
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else evaluate)(args)

if __name__ == '__main__':
    main()
