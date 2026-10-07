"""Audit the causal latest-success online LoRA chain and paired old mean run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash, records_from_episode
from ttcl.trajectory_hyperlora.alfworld_online_latest_success_v1 import checked_targets
from ttcl.trajectory_hyperlora.alfworld_online_context_vector_mean_v1 import vector_hash
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import bounded_source_text
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields


def audit(args):
    if args.audit.exists():
        raise FileExistsError(args.audit)
    targets = checked_targets(args)
    latest = json.loads(args.output.read_text())
    mean = json.loads(args.mean_report.read_text())
    if (latest['review_sha256'] != file_hash(args.review) or
            latest['checkpoint_sha256'] != file_hash(args.checkpoint) or
            mean['checkpoint_sha256'] != file_hash(args.checkpoint) or
            latest['failures'] or mean['failures'] or
            len(latest['games']) != len(targets) or
            len(mean['games']) != len(targets) or
            latest['context_tokens'] != 2048 or
            latest['max_steps'] != 50 or mean['max_steps'] != 50 or
            latest['max_new_tokens'] != 64 or mean['max_new_tokens'] != 64 or
            latest['actor_history_turns'] != 2 or
            mean['actor_history_turns'] != 2 or
            latest['loop_guard_max'] != 2 or mean['loop_guard_max'] != 2):
        raise ValueError('Incomplete or changed latest/mean pairing')
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    fields = None
    prior = []
    updates = 0
    source = None
    rows = []
    for index, (target, row, old) in enumerate(zip(
            targets, latest['games'], mean['games'], strict=True)):
        episode = row['episode']
        old_episode = old['episode']
        if (row['game'] != target['game'] or old['game'] != target['game'] or
                row['input_content_sha256'] != target['input_content_sha256'] or
                row['prior_episode_count'] != len(prior) or
                row['prior_episodes_sha256'] != digest(prior) or
                row['source_vector_sha256_before'] != vector_hash(fields) or
                row['latest_source_game_before'] != source or
                episode['status'] != 'complete' or
                old_episode['status'] != 'complete' or
                episode['steps'] > 50 or
                episode['steps'] != len(episode['trajectory']) or
                episode['reward'] != float(episode['termination'] == 'success') or
                episode['invalid_commands'] != 0):
            raise ValueError(f'Changed target, memory, actor budget, or reward at {index}')
        records = records_from_episode(episode)
        if (row['new_records_sha256'] != digest(records) or
                row['memory_updated'] != (bool(episode['reward']) and
                    (args.policy == 'latest' or fields is None))):
            raise ValueError(f'Changed own trajectory at {index}')
        if row['memory_updated']:
            text, original, retained = bounded_source_text(tokenizer, records, 2048)
            if (row['source_tokens_original'] != original or
                    row['source_tokens_retained'] != retained):
                raise ValueError(f'Changed own source truncation at {index}')
            fields = contextual_text_fields(agent, tokenizer, text,
                args.device, 2048, pooling='both')
            updates += 1
            source = row['game']
        prior.append({'game': row['game'], 'reward': episode['reward'],
                      'records': records})
        if (row['source_vector_sha256_after'] != vector_hash(fields) or
                row['update_count_after'] != updates or
                row['latest_source_game_after'] != source):
            raise ValueError(f'Changed latest source state at {index}')
        rows.append({'index': index, 'game': row['game'],
            'source_before': row['latest_source_game_before'],
            'latest_reward': episode['reward'], 'mean_reward': old_episode['reward'],
            'same_trajectory': episode['trajectory'] == old_episode['trajectory'],
            'memory_updated': row['memory_updated']})
    if latest['summary'] != {'n': len(rows),
                            'success': sum(x['latest_reward'] for x in rows),
                            'failures': 0}:
        raise ValueError('Latest summary changed')
    summary = {'targets': len(rows),
        'candidate_success': sum(x['latest_reward'] for x in rows),
        'mean_success': sum(x['mean_reward'] for x in rows),
        'latest_only': sum(x['latest_reward'] > x['mean_reward'] for x in rows),
        'mean_only': sum(x['latest_reward'] < x['mean_reward'] for x in rows),
        'same_trajectory': sum(x['same_trajectory'] for x in rows),
        'memory_updates': updates}
    result = {'protocol': f'Content-bound replay of {args.policy} own-success source vectors; paired frozen old mean run on same reviewed training targets',
        'policy': args.policy,
        'latest_report_sha256': file_hash(args.output),
        'mean_report_sha256': file_hash(args.mean_report),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'review_sha256': file_hash(args.review),
        'summary': summary, 'games': rows}
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--plan', type=Path, default=Path('ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json'))
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--parent-review', type=Path, default=Path('data/annotations/alfworld_online_from_empty_train_seq6_6_reviewed_20261007.json'))
    parser.add_argument('--parent-persistent-review', type=Path, default=Path('data/annotations/alfworld_online_context_persistent_train_seq6_6_reviewed_20261007.json'))
    parser.add_argument('--review', type=Path, default=Path('data/annotations/alfworld_online_latest_success_train_seq6_6_reviewed_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_online_latest_success_train_seq6_6_20261007.json'))
    parser.add_argument('--mean-report', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq6_6_20261007.json'))
    parser.add_argument('--audit', type=Path, default=Path('results/trajectory_hyperlora/alf_online_latest_success_train_seq6_6_audited_20261007.json'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--policy', choices=('latest', 'first'), default='latest')
    parser.add_argument('--gpu-fraction', type=float, default=.65)
    args = parser.parse_args()
    audit(args)

if __name__ == '__main__':
    main()
