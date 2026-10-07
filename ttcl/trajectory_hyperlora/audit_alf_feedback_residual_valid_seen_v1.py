"""Audit official valid_seen paired controls and frozen residual results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alf_feedback_residual_valid_seen_v1 import checked
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash


def verify_episode(episode, initial=None):
    if (episode['status'] != 'complete' or
            episode['steps'] != len(episode['trajectory']) or
            episode['steps'] > 50 or episode['invalid_commands'] != 0 or
            episode['reward'] != float(episode['termination'] == 'success') or
            (initial is not None and
             episode['initial_observation'] != initial)):
        raise ValueError('Changed official ALFWorld result or budget')


def audit(args):
    if args.audit.exists():
        raise FileExistsError(args.audit)
    targets, source = checked(args)
    report = json.loads(args.report.read_text())
    controls = json.loads(args.controls.read_text())
    if (controls['review_sha256'] != file_hash(args.review) or
            controls['checkpoint_sha256'] != file_hash(args.checkpoint) or
            controls['source_records_sha256'] != source['records_sha256'] or
            controls['failures'] or len(controls['games']) != len(targets) or
            controls['max_steps'] != 50 or
            controls['max_new_tokens'] != 64 or
            controls['actor_history_turns'] != 2 or
            controls['loop_guard_max'] != 2):
        raise ValueError('Incomplete paired official controls')
    for target, control in zip(targets, controls['games'], strict=True):
        if (control['game'] != target['game'] or
                control['input_content_sha256'] !=
                target['input_content_sha256']):
            raise ValueError('Changed reviewed control target')
        verify_episode(control['base'])
        initial = control['base']['initial_observation']
        verify_episode(control['old_lora'], initial)
        verify_episode(control['raw_text'], initial)
    if args.command == 'controls':
        if file_hash(args.report) != file_hash(args.controls):
            raise ValueError('Controls input must be the paired control run')
        summary = {'targets': len(targets),
            'base': sum(x['base']['reward'] for x in controls['games']),
            'old_lora': sum(x['old_lora']['reward']
                            for x in controls['games']),
            'raw_text': sum(x['raw_text']['reward']
                            for x in controls['games'])}
    else:
        training = json.loads(args.training_report.read_text())
        train_audit = json.loads(args.training_audit.read_text())
        if (report['review_sha256'] != file_hash(args.review) or
                report['controls_sha256'] != file_hash(args.controls) or
                report['training_report_sha256'] !=
                    file_hash(args.training_report) or
                report['residual_sha256'] != file_hash(args.residual) or
                report['checkpoint_sha256'] != file_hash(args.checkpoint) or
                train_audit['report_sha256'] !=
                    file_hash(args.training_report) or
                train_audit['model_sha256'] != file_hash(args.residual) or
                training['trained_model_sha256'] !=
                    file_hash(args.residual) or
                report['source_id'] != args.source_id or
                report['failures'] or len(report['games']) != len(targets) or
                report['max_steps'] != 50 or
                report['max_new_tokens'] != 64 or
                report['actor_history_turns'] != 2 or
                report['loop_guard_max'] != 2):
            raise ValueError('Changed or incomplete frozen residual evaluation')
        for target, control, row in zip(targets, controls['games'],
                                        report['games'], strict=True):
            if (row['game'] != target['game'] or
                    row['input_content_sha256'] !=
                        target['input_content_sha256'] or
                    row['base_reward'] != control['base']['reward'] or
                    row['old_lora_reward'] != control['old_lora']['reward'] or
                    row['raw_text_reward'] != control['raw_text']['reward'] or
                    row['mean_code_norm'] < 0):
                raise ValueError('Changed paired residual target')
            verify_episode(row['new'],
                           control['base']['initial_observation'])
        summary = {'targets': len(targets),
            'residual_scale': report.get('residual_scale', 1.),
            'base': sum(x['base_reward'] for x in report['games']),
            'old_lora': sum(x['old_lora_reward']
                            for x in report['games']),
            'raw_text': sum(x['raw_text_reward']
                            for x in report['games']),
            'residual': sum(x['new']['reward'] for x in report['games']),
            'residual_only_vs_old': sum(x['new']['reward'] >
                x['old_lora_reward'] for x in report['games']),
            'old_only_vs_residual': sum(x['new']['reward'] <
                x['old_lora_reward'] for x in report['games'])}
    result = {'protocol': 'Content-bound official valid_seen one-attempt audit of fixed source controls and frozen reward-trained residual; six task families, two games each',
        'review_sha256': file_hash(args.review),
        'controls_sha256': file_hash(args.controls),
        'report_sha256': file_hash(args.report),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'summary': summary}
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('controls', 'residual'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--source-report-seq0', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq0_6_20261007.json'))
    parser.add_argument('--source-report-seq6', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq6_6_20261007.json'))
    parser.add_argument('--plan', type=Path, default=Path('ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json'))
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--checkpoint-training-candidates', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json'))
    parser.add_argument('--warmstart-candidates', type=Path, default=Path('results/trajectory_hyperlora/alf_next_task_candidates_20261005.json'))
    parser.add_argument('--prior-train-review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--training-review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--review', type=Path, default=Path('data/annotations/alf_feedback_residual_valid_seen12_reviewed_20261007.json'))
    parser.add_argument('--source-id', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261007)
    parser.add_argument('--controls', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_valid_seen12_controls_20261007.json'))
    parser.add_argument('--report', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_valid_seen12_controls_20261007.json'))
    parser.add_argument('--residual', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_balanced_seed42_72_20261007.pt'))
    parser.add_argument('--training-report', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_balanced_seed42_72_20261007.json'))
    parser.add_argument('--training-audit', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_balanced_seed42_72_audited_20261007.json'))
    parser.add_argument('--audit', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_valid_seen12_controls_audited_20261007.json'))
    audit(parser.parse_args())


if __name__ == '__main__':
    main()
