"""Audit latest-success valid_unseen memory and paired frozen controls."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_context_vector_mean_v1 import vector_hash
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash, records_from_episode
from ttcl.trajectory_hyperlora.alfworld_online_latest_success_valid_unseen_v1 import checked_targets
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import bounded_source_text
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields


def audit(args):
    if args.audit.exists():
        raise FileExistsError(args.audit)
    targets = checked_targets(args)
    latest = json.loads(args.output.read_text())
    mean = json.loads(args.reference.read_text())
    first = json.loads(args.first_report.read_text())
    if (latest['review_sha256'] != file_hash(args.review) or
            latest['reference_sha256'] != file_hash(args.reference) or
            latest['checkpoint_sha256'] != file_hash(args.checkpoint) or
            mean['checkpoint_sha256'] != file_hash(args.checkpoint) or
            first['checkpoint_sha256'] != file_hash(args.checkpoint) or
            latest['failures'] or mean['failures'] or first['failures']):
        raise ValueError('Changed or incomplete valid_unseen arms')
    if not (len(latest['games']) == len(mean['games']) ==
            len(first['games']) == len(targets) == 36):
        raise ValueError('Missing valid_unseen games')
    if (latest['max_steps'] != 50 or latest['max_new_tokens'] != 64 or
            latest['actor_history_turns'] != 2 or
            latest['loop_guard_max'] != 2 or
            latest['context_tokens'] != 2048):
        raise ValueError('Changed latest actor budget')
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    fields = None
    source_game = None
    updates = 0
    prior = []
    rows = []
    for index, (target, row, old, frozen) in enumerate(zip(
            targets, latest['games'], mean['games'], first['games'], strict=True)):
        episode = row['episode']
        if (row['game'] != target['game'] or
                old['game'] != target['game'] or
                frozen['game'] != target['game'] or
                row['family'] != target['family'] or
                row['input_content_sha256'] != target['input_content_sha256'] or
                row['prior_episode_count'] != len(prior) or
                row['prior_episodes_sha256'] != digest(prior) or
                row['source_vector_sha256_before'] != vector_hash(fields) or
                row['latest_source_game_before'] != source_game or
                episode['status'] != 'complete' or
                episode['steps'] > 50 or
                episode['steps'] != len(episode['trajectory']) or
                episode['reward'] != float(episode['termination'] == 'success') or
                episode['invalid_commands'] != 0 or
                episode['initial_observation'] != old['base']['initial_observation']):
            raise ValueError(f'Changed latest memory, actor or reward at {index}')
        records = records_from_episode(episode)
        if (row['new_records_sha256'] != digest(records) or
                row['memory_updated'] != bool(episode['reward'])):
            raise ValueError(f'Changed own trajectory at {index}')
        if episode['reward']:
            bounded, original, retained = bounded_source_text(tokenizer, records, 2048)
            if (row['source_tokens_original'] != original or
                    row['source_tokens_retained'] != retained):
                raise ValueError(f'Changed source truncation at {index}')
            fields = contextual_text_fields(agent, tokenizer, bounded,
                args.device, 2048, pooling='both')
            source_game = row['game']
            updates += 1
        prior.append({'game': row['game'], 'reward': episode['reward'],
                      'records': records})
        if (row['source_vector_sha256_after'] != vector_hash(fields) or
                row['latest_source_game_after'] != source_game or
                row['update_count_after'] != updates):
            raise ValueError(f'Changed source update at {index}')
        rows.append({'index': index, 'game': row['game'],
            'family': row['family'], 'base': old['base']['reward'],
            'mean': old['online']['reward'],
            'first': frozen['episode']['reward'],
            'latest': episode['reward'],
            'same_trajectory_mean': episode['trajectory'] == old['online']['trajectory'],
            'same_trajectory_first': episode['trajectory'] == frozen['episode']['trajectory']})
    if latest['summary'] != {'n': len(rows),
            'base': sum(x['base'] for x in rows),
            'old_mean': sum(x['mean'] for x in rows),
            'latest': sum(x['latest'] for x in rows), 'failures': 0}:
        raise ValueError('Changed latest summary')
    family = defaultdict(lambda: {'n': 0, 'base': 0, 'mean': 0, 'first': 0, 'latest': 0})
    for row in rows:
        item = family[row['family']]
        item['n'] += 1
        for arm in ('base', 'mean', 'first', 'latest'):
            item[arm] += row[arm]
    summary = {'n': len(rows), 'updates': updates,
        'base': sum(x['base'] for x in rows),
        'mean': sum(x['mean'] for x in rows),
        'first': sum(x['first'] for x in rows),
        'latest': sum(x['latest'] for x in rows),
        'latest_only_vs_first': sum(x['latest'] > x['first'] for x in rows),
        'first_only_vs_latest': sum(x['latest'] < x['first'] for x in rows),
        'latest_only_vs_mean': sum(x['latest'] > x['mean'] for x in rows),
        'mean_only_vs_latest': sum(x['latest'] < x['mean'] for x in rows),
        'same_trajectory_mean': sum(x['same_trajectory_mean'] for x in rows),
        'same_trajectory_first': sum(x['same_trajectory_first'] for x in rows),
        'family': dict(family)}
    result = {'protocol': 'Content-bound latest own-success replay on previously used official valid_unseen 36, paired with old mean/base and fixed-first; exploratory not untouched test',
        'latest_report_sha256': file_hash(args.output),
        'mean_report_sha256': file_hash(args.reference),
        'first_report_sha256': file_hash(args.first_report),
        'review_sha256': file_hash(args.review),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'summary': summary, 'games': rows}
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--parent-review', type=Path, default=Path('data/annotations/alf_memrl134_hyperlora_train_sources_reviewed_20261006.json'))
    parser.add_argument('--parent-online-review', type=Path, default=Path('data/annotations/alf_online_reward_gate_valid_unseen_36_reviewed_20261007.json'))
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--review', type=Path, default=Path('data/annotations/alf_online_latest_success_valid_unseen36_reviewed_20261007.json'))
    parser.add_argument('--reference', type=Path, default=Path('results/trajectory_hyperlora/alf_online_reward_gate_valid_unseen_36_20261007.json'))
    parser.add_argument('--first-report', type=Path, default=Path('results/trajectory_hyperlora/alf_online_fixed_first_success_valid_unseen36_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_online_latest_success_valid_unseen36_20261007.json'))
    parser.add_argument('--audit', type=Path, default=Path('results/trajectory_hyperlora/alf_online_latest_success_valid_unseen36_audited_20261007.json'))
    parser.add_argument('--per-family', type=int, default=6)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.65)
    args = parser.parse_args()
    audit(args)

if __name__ == '__main__':
    main()
