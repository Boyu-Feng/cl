"""Audit paired official ALFWorld outcomes of a frozen feedback LoRA residual."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.train_alf_future_utility_selector_v1 import target_reward_matrix


def audit(args):
    if args.audit.exists():
        raise FileExistsError(args.audit)
    run = json.loads(args.output.read_text())
    review = json.loads(args.review.read_text())
    matrix = json.loads(args.matrix.read_text())
    training = json.loads(args.training_report.read_text())
    training_audit = json.loads(args.training_audit.read_text())
    holdout_audit = json.loads(args.holdout_audit.read_text())
    if (run['residual_sha256'] != file_hash(args.residual) or
            run['training_report_sha256'] != file_hash(args.training_report) or
            run['review_sha256'] != file_hash(args.review) or
            run['audit_sha256'] != file_hash(args.holdout_audit) or
            run['checkpoint_sha256'] != file_hash(args.checkpoint) or
            training['trained_model_sha256'] != file_hash(args.residual) or
            training_audit['report_sha256'] != file_hash(args.training_report) or
            holdout_audit['raw_report_sha256'] != file_hash(args.matrix) or
            holdout_audit['review_sha256'] != file_hash(args.review) or
            matrix['failures'] or run['failures'] or
            len(run['games']) != 12 or len(matrix['base']) != 12 or
            run['source_id'] not in range(8) or
            run['max_steps'] != 50 or run['max_new_tokens'] != 64 or
            run['actor_history_turns'] != 2 or
            run['loop_guard_max'] != 2):
        raise ValueError('Changed or incomplete development outcome lineage')
    source_id = run['source_id']
    old = target_reward_matrix(matrix, 8, 12)
    for index, (row, target) in enumerate(zip(run['games'],
            review['targets'], strict=True)):
        episode = row['new']
        if (row['target_id'] != index or row['game'] != target['game'] or
                row['game_sha256'] != target['game_sha256'] or
                row['old_source_reward'] != float(old[index, source_id]) or
                row['base_reward'] != matrix['base'][index]['episode']['reward'] or
                episode['status'] != 'complete' or
                episode['initial_observation'] !=
                    matrix['base'][index]['episode']['initial_observation'] or
                episode['steps'] != len(episode['trajectory']) or
                episode['steps'] > 50 or episode['invalid_commands'] != 0 or
                episode['reward'] != float(episode['termination'] == 'success')):
            raise ValueError(f'Changed or invalid official target result {index}')
    gains = sum(row['new']['reward'] > row['old_source_reward']
                for row in run['games'])
    losses = sum(row['new']['reward'] < row['old_source_reward']
                 for row in run['games'])
    summary = {'targets': 12, 'source_id': source_id,
        'residual_scale': run.get('residual_scale', 1.),
        'old': sum(row['old_source_reward'] for row in run['games']),
        'trained': sum(row['new']['reward'] for row in run['games']),
        'no_lora': sum(row['base_reward'] for row in run['games']),
        'trained_only': gains, 'old_only': losses,
        'mean_code_norm': sum(row['mean_code_norm']
            for row in run['games']) / 12}
    audited = {'protocol': 'Content-bound official score audit of frozen reward-trained LoRA residual against same-source old LoRA on previously exposed train-domain development tasks',
        'raw_report_sha256': file_hash(args.output),
        'training_audit_sha256': file_hash(args.training_audit),
        'holdout_audit_sha256': file_hash(args.holdout_audit),
        'residual_sha256': file_hash(args.residual),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'summary': summary}
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audited, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_dev12_seed42_36_20261007.json'))
    parser.add_argument('--audit', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_dev12_seed42_36_audited_20261007.json'))
    parser.add_argument('--training-report', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_seed42_36_20261007.json'))
    parser.add_argument('--training-audit', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_seed42_36_audited_20261007.json'))
    parser.add_argument('--residual', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_seed42_36_20261007.pt'))
    parser.add_argument('--review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_holdout_8x12_v3c_reviewed_20261007.json'))
    parser.add_argument('--matrix', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_holdout_8x12_v3c_merged_20261007.json'))
    parser.add_argument('--holdout-audit', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_holdout_8x12_v3c_merged_audited_20261007.json'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    audit(parser.parse_args())


if __name__ == '__main__':
    main()
