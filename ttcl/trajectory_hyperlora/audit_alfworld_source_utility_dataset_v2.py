"""Audit the 8x18 own-trajectory future-reward matrix and split integrity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v2 import checked


def validate_episode(episode, game):
    if (episode['status'] != 'complete' or
            episode['steps'] > 50 or
            episode['steps'] != len(episode['trajectory']) or
            episode['invalid_commands'] != 0 or
            episode['reward'] != float(episode['termination'] == 'success')):
        raise ValueError(f'Changed official episode score or budget: {game}')


def audit(args):
    if args.audit.exists():
        raise FileExistsError(args.audit)
    sources, targets, pairs = checked(args)
    report = json.loads(args.output.read_text())
    old_training = json.loads(args.checkpoint_training_candidates.read_text())
    old_training_games = {row['target_game'] for row in old_training['rows']}
    warmstart = json.loads(args.warmstart_candidates.read_text())
    warmstart_games = {row['target_game'] for row in warmstart['candidates']
                       if row['split'] == 'train'}
    checkpoint_label_files = (args.contextual_labels, args.onpolicy_labels)
    checkpoint_label_games = set()
    for label_file in checkpoint_label_files:
        labels = json.loads(label_file.read_text())
        if not isinstance(labels.get('tasks'), list):
            raise ValueError('Changed checkpoint label lineage')
        checkpoint_label_games.update(row['target_game'] for row in labels['tasks'])
    if not checkpoint_label_games <= old_training_games:
        raise ValueError('Candidate pool does not cover checkpoint training labels')
    if len(old_training['rows']) != 600:
        raise ValueError('Changed checkpoint training candidate count')
    prior_target_games = set()
    prior_sequences = set()
    if args.target_policy == 'independent12':
        prior_review = json.loads(args.prior_train_review.read_text())
        prior_target_games = {row['game'] for row in prior_review['targets']}
        prior_sequences = {row['sequence_index'] for row in prior_review['targets']}
        if (any(target['game'] in old_training_games or
                target['game'] in warmstart_games or
                target['game'] in checkpoint_label_games or
                target['game'] in prior_target_games or
                target['sequence_index'] in prior_sequences
                for target in targets)):
            raise ValueError('Holdout overlaps checkpoint or collection training')
    if (report['review_sha256'] != file_hash(args.review) or
            report['checkpoint_sha256'] != file_hash(args.checkpoint) or
            report['failures'] or len(report['base']) != len(targets) or
            len(report['pairs']) != len(pairs) or
            report['max_steps'] != 50 or report['max_new_tokens'] != 64 or
            report['actor_history_turns'] != 2 or
            report['loop_guard_max'] != 2 or
            report['context_tokens'] != 2048):
        raise ValueError('Incomplete or changed cross-source collection')
    base_rewards = []
    base_observations = []
    for target, row in zip(targets, report['base'], strict=True):
        if (row['target_id'] != target['target_id'] or
                row['game'] != target['game'] or
                row['game_sha256'] != target['game_sha256']):
            raise ValueError('Changed base target content')
        validate_episode(row['episode'], row['game'])
        base_rewards.append(row['episode']['reward'])
        base_observations.append(row['episode']['initial_observation'])
    rows = []
    for binding, actual in zip(pairs, report['pairs'], strict=True):
        si, ti = binding['source_id'], binding['target_id']
        target, source = targets[ti], sources[si]
        if (actual['input_content_sha256'] != binding['input_content_sha256'] or
                actual['source_id'] != si or actual['target_id'] != ti or
                actual['source_tokens_retained'] > 2048 or
                actual['episode']['initial_observation'] != base_observations[ti]):
            raise ValueError('Changed source-target binding or reset')
        validate_episode(actual['episode'], target['game'])
        rows.append({'source_id': si, 'target_id': ti,
            'source_game': source['game'], 'target_game': target['game'],
            'source_records_sha256': source['records_sha256'],
            'input_content_sha256': binding['input_content_sha256'],
            'target_family': target['family'],
            'checkpoint_candidate_overlap': target['game'] in old_training_games,
            'base_reward': base_rewards[ti],
            'lora_reward': actual['episode']['reward']})
    groups = [[x for x in rows if x['target_id'] == ti] for ti in range(len(targets))]
    source_totals = {str(si): sum(x['lora_reward'] for x in rows
                                  if x['source_id'] == si)
                     for si in range(len(sources))}
    summary = {'sources': len(sources), 'targets': len(targets),
        'pairs': len(rows), 'base_success': sum(base_rewards),
        'pair_success': sum(x['lora_reward'] for x in rows),
        'source_totals': source_totals,
        'source_dependent_targets': sum(
            len({x['lora_reward'] for x in group}) > 1 for group in groups),
        'best_static_source': max(source_totals.values()),
        'oracle': sum(max(x['lora_reward'] for x in group) for group in groups),
        'checkpoint_training_candidate_overlap': sum(
            target['game'] in old_training_games for target in targets),
        'checkpoint_training_candidate_overlap_target_ids': [
            target['target_id'] for target in targets
            if target['game'] in old_training_games],
        'warmstart_training_target_overlap': sum(
            target['game'] in warmstart_games for target in targets),
        'warmstart_training_overlap_target_ids': [
            target['target_id'] for target in targets
            if target['game'] in warmstart_games],
        'actual_checkpoint_label_target_overlap': sum(
            target['game'] in checkpoint_label_games for target in targets),
        'actual_checkpoint_label_overlap_target_ids': [
            target['target_id'] for target in targets
            if target['game'] in checkpoint_label_games],
        'prior_training_game_overlap': sum(target['game'] in prior_target_games
                                           for target in targets),
        'prior_training_sequence_overlap': sum(
            target['sequence_index'] in prior_sequences for target in targets)}
    result = {'protocol': ('Content-bound official-reward audit for independent 8x12 holdout, disjoint from prior collection and checkpoint training'
        if args.target_policy == 'independent12' else
        'Content-bound official-reward audit for 8x18 train-domain matrix; full matrix is training data, not independent validation'),
        'target_policy': args.target_policy,
        'review_sha256': file_hash(args.review),
        'raw_report_sha256': file_hash(args.output),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'checkpoint_training_candidates_sha256': file_hash(
            args.checkpoint_training_candidates),
        'warmstart_candidates_sha256': file_hash(args.warmstart_candidates),
        'checkpoint_label_sha256': {str(path): file_hash(path)
            for path in checkpoint_label_files},
        'summary': summary, 'pairs': rows}
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--checkpoint-training-candidates', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json'))
    parser.add_argument('--warmstart-candidates', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_next_task_candidates_20261005.json'))
    parser.add_argument('--contextual-labels', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_sibling_train120_labels_20261006.json'))
    parser.add_argument('--onpolicy-labels', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_sibling_simple_train40_onpolicy_best_labels_20261006.json'))
    parser.add_argument('--source-report-seq0', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq0_6_20261007.json'))
    parser.add_argument('--source-report-seq6', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq6_6_20261007.json'))
    parser.add_argument('--plan', type=Path, default=Path('ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json'))
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--target-policy', choices=('train18', 'independent12'),
                        default='train18')
    parser.add_argument('--prior-train-review', type=Path, default=Path(
        'data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_20261007.json'))
    parser.add_argument('--audit', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_audited_20261007.json'))
    audit(parser.parse_args())

if __name__ == '__main__':
    main()
