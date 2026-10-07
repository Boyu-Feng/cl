"""From-empty self-trajectory text-memory control for online LoRA studies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash, records_from_episode
from ttcl.trajectory_hyperlora.alfworld_online_latest_success_v1 import parent_targets
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import bounded_source_text
from ttcl.trajectory_hyperlora.alfworld_replica30_text_summary import summarize
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode


def content(args, index, parent):
    return {'index': index, 'game': parent['game'],
        'game_sha256': file_hash(args.data_root / parent['game']),
        'parent_target_sha256': parent['input_content_sha256'],
        'parent_review_sha256': file_hash(args.parent_persistent_review),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'method': f'online_self_success_text_{args.policy}_{args.text_mode}'}


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    parent = parent_targets(args)
    targets = []
    for index, row in enumerate(parent):
        bound = content(args, index, row)
        targets.append({**bound, 'input_content_sha256': digest(bound),
                        'reviewed_target': True})
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps({'protocol': 'Fresh content-bound online own-text memory target review',
        'parent_review_sha256': file_hash(args.parent_persistent_review),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'policy': args.policy, 'text_mode': args.text_mode,
        'targets': targets}, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'reviewed_targets': len(targets)}), flush=True)


def checked_targets(args):
    parent = parent_targets(args)
    review = json.loads(args.review.read_text())
    if (review['parent_review_sha256'] != file_hash(args.parent_persistent_review) or
            review['checkpoint_sha256'] != file_hash(args.checkpoint) or
            review['policy'] != args.policy or review['text_mode'] != args.text_mode or
            len(review['targets']) != len(parent)):
        raise ValueError('Changed text target lineage')
    for index, row in enumerate(parent):
        bound = content(args, index, row)
        approved = review['targets'][index]
        if (approved['reviewed_target'] is not True or
                approved['input_content_sha256'] != digest(bound) or
                any(approved[key] != value for key, value in bound.items())):
            raise ValueError('Changed text target content')
    return review['targets']


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    targets = checked_targets(args)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    memory = None
    source_game = None
    prior = []
    updates = 0
    report = {'protocol': 'From-empty own-success text memory with frozen actor; same 50-step/64-token/2-turn history/constrained greedy/two-repeat loop guard as online LoRA; raw=2048 source tokens, summary<=112 generated tokens; no LoRA',
        'policy': args.policy, 'text_mode': args.text_mode,
        'review_sha256': file_hash(args.review),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'max_steps': 50, 'max_new_tokens': 64,
        'actor_history_turns': 2, 'loop_guard_max': 2,
        'games': [], 'failures': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for row in targets:
        before = hashlib.sha256((memory or '').encode()).hexdigest()
        episode = run_episode(agent, tokenizer, args.data_root / row['game'],
            {}, adapter=False, device=args.device,
            max_steps=50, max_new_tokens=64,
            constrain_actions=True, actor_history_turns=2,
            loop_guard_max=2, memory_text=memory)
        entry = {'game': row['game'],
            'input_content_sha256': row['input_content_sha256'],
            'prior_episode_count': len(prior),
            'prior_episodes_sha256': digest(prior),
            'source_game_before': source_game,
            'memory_sha256_before': before, 'episode': episode}
        if episode['status'] == 'complete':
            try:
                records = records_from_episode(episode)
                entry['new_records_sha256'] = digest(records)
                entry['memory_updated'] = bool(episode['reward']) and (
                    args.policy == 'latest' or memory is None)
                if entry['memory_updated']:
                    bounded, original, retained = bounded_source_text(
                        tokenizer, records, 2048)
                    memory = (bounded if args.text_mode == 'raw' else
                              summarize(agent, tokenizer, records, args.device))
                    if not memory:
                        raise ValueError('Empty generated text memory')
                    source_game = row['game']
                    updates += 1
                    entry.update({'source_tokens_original': original,
                        'source_tokens_retained': retained,
                        'memory_tokens': len(tokenizer(memory,
                            add_special_tokens=False).input_ids),
                        'memory_text': memory})
                prior.append({'game': row['game'], 'reward': episode['reward'],
                              'records': records})
            except Exception as exc:
                report['failures'].append({'game': row['game'],
                    'error': f'{type(exc).__name__}: {exc}'})
        else:
            report['failures'].append({'game': row['game'],
                'error': episode.get('error', 'incomplete rollout')})
        entry['source_game_after'] = source_game
        entry['memory_sha256_after'] = hashlib.sha256((memory or '').encode()).hexdigest()
        entry['update_count_after'] = updates
        report['games'].append(entry)
        report['summary'] = {'n': len(report['games']),
            'success': sum(x['episode'].get('reward', 0) for x in report['games']),
            'failures': len(report['failures'])}
        temp = args.output.with_suffix(args.output.suffix + '.tmp')
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        temp.replace(args.output)
        print(json.dumps(report['summary']), flush=True)
        if report['failures']:
            raise RuntimeError('Online text-memory failure recorded')
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
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policy', choices=('latest', 'first'), default='latest')
    parser.add_argument('--text-mode', choices=('raw', 'summary'), default='raw')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.65)
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else evaluate)(args)

if __name__ == '__main__':
    main()
