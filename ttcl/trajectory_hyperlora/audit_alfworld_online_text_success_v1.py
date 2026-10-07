"""Audit from-empty self-trajectory text-memory chains against frozen LoRA arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash, records_from_episode
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import bounded_source_text
from ttcl.trajectory_hyperlora.alfworld_online_text_success_v1 import checked_targets
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent


def hash_text(value):
    return hashlib.sha256((value or '').encode()).hexdigest()


def audit(args):
    if args.audit.exists():
        raise FileExistsError(args.audit)
    targets = checked_targets(args)
    text_run = json.loads(args.output.read_text())
    lora = json.loads(args.lora_report.read_text())
    if (text_run['review_sha256'] != file_hash(args.review) or
            text_run['checkpoint_sha256'] != file_hash(args.checkpoint) or
            lora['checkpoint_sha256'] != file_hash(args.checkpoint) or
            text_run['policy'] != args.policy or
            text_run['text_mode'] != args.text_mode or
            text_run['failures'] or lora['failures'] or
            len(text_run['games']) != len(targets) or
            len(lora['games']) != len(targets) or
            text_run['max_steps'] != 50 or
            text_run['max_new_tokens'] != 64 or
            text_run['actor_history_turns'] != 2 or
            text_run['loop_guard_max'] != 2):
        raise ValueError('Incomplete or changed text/LoRA paired evaluation')
    _, tokenizer = load_agent(args.model, args.checkpoint,
                              args.device, args.gpu_fraction)
    memory = None
    source = None
    updates = 0
    prior = []
    rows = []
    for index, (target, row, comparison) in enumerate(zip(
            targets, text_run['games'], lora['games'], strict=True)):
        episode = row['episode']
        lora_episode = comparison.get('episode', comparison.get('online'))
        if (row['game'] != target['game'] or
                comparison['game'] != target['game'] or
                row['input_content_sha256'] != target['input_content_sha256'] or
                row['prior_episode_count'] != len(prior) or
                row['prior_episodes_sha256'] != digest(prior) or
                row['source_game_before'] != source or
                row['memory_sha256_before'] != hash_text(memory) or
                episode['memory_sha256'] != hash_text(memory) or
                episode['status'] != 'complete' or
                lora_episode['status'] != 'complete' or
                episode['steps'] > 50 or
                episode['steps'] != len(episode['trajectory']) or
                episode['reward'] != float(episode['termination'] == 'success') or
                episode['invalid_commands'] != 0 or
                episode['initial_observation'] != lora_episode['initial_observation']):
            raise ValueError(f'Changed text state, target, actor budget, or reward at {index}')
        records = records_from_episode(episode)
        should_update = bool(episode['reward']) and (
            args.policy == 'latest' or memory is None)
        if (row['new_records_sha256'] != digest(records) or
                row['memory_updated'] != should_update):
            raise ValueError(f'Changed own trajectory at {index}')
        if should_update:
            bounded, original, retained = bounded_source_text(tokenizer, records, 2048)
            memory = row['memory_text']
            if (row['source_tokens_original'] != original or
                    row['source_tokens_retained'] != retained or
                    not memory or
                    (args.text_mode == 'raw' and memory != bounded) or
                    row['memory_tokens'] != len(tokenizer(memory,
                        add_special_tokens=False).input_ids)):
                raise ValueError(f'Changed own text source at {index}')
            source = row['game']
            updates += 1
        prior.append({'game': row['game'], 'reward': episode['reward'],
                      'records': records})
        if (row['source_game_after'] != source or
                row['memory_sha256_after'] != hash_text(memory) or
                row['update_count_after'] != updates):
            raise ValueError(f'Changed text update chain at {index}')
        rows.append({'index': index, 'game': row['game'],
            'text_reward': episode['reward'],
            'lora_reward': lora_episode['reward'],
            'memory_source_before': row['source_game_before'],
            'memory_updated': should_update})
    if text_run['summary'] != {'n': len(rows),
                              'success': sum(r['text_reward'] for r in rows),
                              'failures': 0}:
        raise ValueError('Changed text run summary')
    summary = {'targets': len(rows), 'text_success': sum(r['text_reward'] for r in rows),
        'lora_success': sum(r['lora_reward'] for r in rows),
        'text_only': sum(r['text_reward'] > r['lora_reward'] for r in rows),
        'lora_only': sum(r['text_reward'] < r['lora_reward'] for r in rows),
        'memory_updates': updates}
    result = {'protocol': 'Content-bound own text trajectory and official paired LoRA reward audit; no new annotation labels inherited by history ID',
        'text_report_sha256': file_hash(args.output),
        'lora_report_sha256': file_hash(args.lora_report),
        'review_sha256': file_hash(args.review),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'policy': args.policy, 'text_mode': args.text_mode,
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
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lora-report', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--policy', choices=('latest', 'first'), default='latest')
    parser.add_argument('--text-mode', choices=('raw', 'summary'), default='raw')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.65)
    args = parser.parse_args()
    audit(args)

if __name__ == '__main__':
    main()
