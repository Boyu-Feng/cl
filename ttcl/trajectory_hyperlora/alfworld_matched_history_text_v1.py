"""Causal text control using exactly the online LoRA chain's past sources.

The text actor never supplies its own history to this control. For target i it
sees the latest successful own trajectory that the frozen LoRA arm had at i.
This isolates memory representation, not fully autonomous text learning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash, records_from_episode
from ttcl.trajectory_hyperlora.alfworld_online_latest_success_v1 import checked_targets as checked_parent_targets
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import bounded_source_text
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode


def parent_targets(args):
    original = args.review
    try:
        args.review = args.parent_latest_review
        return checked_parent_targets(args)
    finally:
        args.review = original


def source_for(report, index):
    previous_game = report['games'][index]['latest_source_game_before']
    if previous_game is None:
        return None, None
    matches = [(i, row) for i, row in enumerate(report['games'][:index])
               if row['game'] == previous_game]
    if len(matches) != 1:
        raise ValueError('Latest source is not in causal past')
    source_index, row = matches[0]
    episode = row['episode']
    records = records_from_episode(episode)
    if (episode['reward'] != 1 or
            row['new_records_sha256'] != digest(records)):
        raise ValueError('Matched source is not an audited own success')
    return source_index, records


def bindings(args):
    targets = parent_targets(args)
    source_run = json.loads(args.source_run.read_text())
    if (source_run['review_sha256'] != file_hash(args.parent_latest_review) or
            source_run['checkpoint_sha256'] != file_hash(args.checkpoint) or
            source_run['failures'] or len(source_run['games']) != len(targets)):
        raise ValueError('Changed own-source lineage')
    rows = []
    for index, target in enumerate(targets):
        source_index, records = source_for(source_run, index)
        content = {'index': index, 'target_game': target['game'],
            'target_game_sha256': file_hash(args.data_root / target['game']),
            'target_input_content_sha256': target['input_content_sha256'],
            'parent_target_review_sha256': file_hash(args.parent_latest_review),
            'source_run_sha256': file_hash(args.source_run),
            'source_index': source_index,
            'source_records_sha256': digest(records) if records else None,
            'checkpoint_sha256': file_hash(args.checkpoint),
            'method': 'same_own_success_history_raw_text_vs_dynamic_lora'}
        rows.append({**content, 'input_content_sha256': digest(content),
                     'reviewed_target': True})
    return targets, source_run, rows


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    _, _, rows = bindings(args)
    value = {'protocol': 'Fresh reviewed causal-past source record and official target content binding for fixed-history text control',
        'source_run_sha256': file_hash(args.source_run),
        'checkpoint_sha256': file_hash(args.checkpoint), 'targets': rows}
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'reviewed_targets': len(rows)}), flush=True)


def checked(args):
    targets, source_run, rows = bindings(args)
    review = json.loads(args.review.read_text())
    if (review['source_run_sha256'] != file_hash(args.source_run) or
            review['checkpoint_sha256'] != file_hash(args.checkpoint) or
            review['targets'] != rows):
        raise ValueError('Changed matched-history text review')
    return targets, source_run, rows


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    targets, source_run, rows = checked(args)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    result = {'protocol': 'Causal-past identical latest-success source trajectory from online LoRA chain rendered as <=2048 raw text tokens; frozen no-LoRA actor, same official target and 50-command/64-token/2-history/greedy/loop-guard2 protocol',
        'review_sha256': file_hash(args.review),
        'source_run_sha256': file_hash(args.source_run),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'max_steps': 50, 'max_new_tokens': 64,
        'actor_history_turns': 2, 'loop_guard_max': 2,
        'games': [], 'failures': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, (target, binding) in enumerate(zip(targets, rows, strict=True)):
        source_index, records = source_for(source_run, index)
        if source_index != binding['source_index']:
            raise ValueError('Changed source index')
        if records is None:
            memory = None
            original = retained = 0
        else:
            memory, original, retained = bounded_source_text(tokenizer, records, 2048)
        episode = run_episode(agent, tokenizer,
            args.data_root / target['game'], {}, adapter=False,
            device=args.device, max_steps=50, max_new_tokens=64,
            constrain_actions=True, actor_history_turns=2,
            loop_guard_max=2, memory_text=memory)
        row = {'game': target['game'],
            'input_content_sha256': binding['input_content_sha256'],
            'source_index': source_index,
            'source_records_sha256': binding['source_records_sha256'],
            'source_tokens_original': original,
            'source_tokens_retained': retained,
            'memory_text': memory,
            'episode': episode}
        if (episode['status'] != 'complete' or
                episode['initial_observation'] !=
                source_run['games'][index]['episode']['initial_observation']):
            result['failures'].append({'game': target['game'],
                'error': episode.get('error', 'incomplete or changed target reset')})
        result['games'].append(row)
        result['summary'] = {'n': len(result['games']),
            'text': sum(x['episode'].get('reward', 0) for x in result['games']),
            'lora': sum(x['episode']['reward'] for x in source_run['games'][:len(result['games'])]),
            'failures': len(result['failures'])}
        temp = args.output.with_suffix(args.output.suffix + '.tmp')
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        temp.replace(args.output)
        print(json.dumps(result['summary']), flush=True)
        if result['failures']:
            raise RuntimeError('Matched-history text failure recorded')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'evaluate'))
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
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.65)
    parser.add_argument('--policy', choices=('latest',), default='latest')
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else evaluate)(args)

if __name__ == '__main__':
    main()
