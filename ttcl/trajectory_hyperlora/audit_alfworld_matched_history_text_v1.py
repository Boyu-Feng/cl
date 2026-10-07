"""Audit source-identical causal-past text versus online LoRA targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_matched_history_text_v1 import checked, source_for
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import bounded_source_text
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent


def audit(args):
    if args.audit.exists():
        raise FileExistsError(args.audit)
    targets, source_run, bindings = checked(args)
    result = json.loads(args.output.read_text())
    if (result['review_sha256'] != file_hash(args.review) or
            result['source_run_sha256'] != file_hash(args.source_run) or
            result['checkpoint_sha256'] != file_hash(args.checkpoint) or
            result['failures'] or len(result['games']) != len(targets) or
            result['max_steps'] != 50 or result['max_new_tokens'] != 64 or
            result['actor_history_turns'] != 2 or
            result['loop_guard_max'] != 2):
        raise ValueError('Incomplete or changed matched-history text run')
    _, tokenizer = load_agent(args.model, args.checkpoint,
                              args.device, args.gpu_fraction)
    rows = []
    for index, (target, binding, row, source_row) in enumerate(zip(
            targets, bindings, result['games'], source_run['games'], strict=True)):
        source_index, records = source_for(source_run, index)
        if records is None:
            memory = None
            original = retained = 0
        else:
            memory, original, retained = bounded_source_text(tokenizer, records, 2048)
        episode = row['episode']
        if (row['game'] != target['game'] or
                row['input_content_sha256'] != binding['input_content_sha256'] or
                row['source_index'] != source_index or
                row['source_records_sha256'] != binding['source_records_sha256'] or
                row['source_tokens_original'] != original or
                row['source_tokens_retained'] != retained or
                row['memory_text'] != memory or
                episode['memory_sha256'] != hashlib.sha256((memory or '').encode()).hexdigest() or
                episode['status'] != 'complete' or
                episode['steps'] > 50 or
                episode['steps'] != len(episode['trajectory']) or
                episode['invalid_commands'] != 0 or
                episode['reward'] != float(episode['termination'] == 'success') or
                episode['initial_observation'] !=
                    source_row['episode']['initial_observation']):
            raise ValueError(f'Changed paired target or source at {index}')
        rows.append({'index': index, 'game': target['game'],
            'source_index': source_index,
            'text_reward': episode['reward'],
            'lora_reward': source_row['episode']['reward']})
    summary = {'targets': len(rows),
        'text': sum(row['text_reward'] for row in rows),
        'lora': sum(row['lora_reward'] for row in rows),
        'text_only': sum(row['text_reward'] > row['lora_reward'] for row in rows),
        'lora_only': sum(row['text_reward'] < row['lora_reward'] for row in rows),
        'source_bearing_targets': sum(row['source_index'] is not None for row in rows)}
    if result['summary'] != {'n': len(rows), 'text': summary['text'],
                             'lora': summary['lora'], 'failures': 0}:
        raise ValueError('Changed matched-history summary')
    audited = {'protocol': 'Content-bound, exact-own-source raw text versus dynamic LoRA with identical target actor budget; text histories fixed to LoRA chain for causal representation comparison',
        'review_sha256': file_hash(args.review),
        'text_report_sha256': file_hash(args.output),
        'source_run_sha256': file_hash(args.source_run),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'summary': summary, 'games': rows}
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audited, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--plan', type=Path, default=Path('ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json'))
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--parent-review', type=Path, default=Path('data/annotations/alfworld_online_from_empty_train_seq6_6_reviewed_20261007.json'))
    parser.add_argument('--parent-persistent-review', type=Path, default=Path('data/annotations/alfworld_online_context_persistent_train_seq6_6_reviewed_20261007.json'))
    parser.add_argument('--parent-latest-review', type=Path, default=Path('data/annotations/alfworld_online_latest_success_train_seq6_6_reviewed_20261007.json'))
    parser.add_argument('--source-run', type=Path, default=Path('results/trajectory_hyperlora/alf_online_latest_success_train_seq6_6_20261007.json'))
    parser.add_argument('--review', type=Path, default=Path('data/annotations/alf_matched_history_raw_text_train_seq6_6_reviewed_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_matched_history_raw_text_train_seq6_6_20261007.json'))
    parser.add_argument('--audit', type=Path, default=Path('results/trajectory_hyperlora/alf_matched_history_raw_text_train_seq6_6_audited_20261007.json'))
    parser.add_argument('--policy', choices=('latest',), default='latest')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.65)
    args = parser.parse_args()
    audit(args)

if __name__ == '__main__':
    main()
