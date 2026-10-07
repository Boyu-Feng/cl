"""Audit real-reward sampled latent LoRA residual training and lineage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.train_alf_future_utility_selector_v1 import target_reward_matrix


def audit(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.report.read_text())
    review = json.loads(args.review.read_text())
    matrix = json.loads(args.matrix.read_text())
    replay = json.loads(args.replay.read_text())
    prior_audit = json.loads(args.matrix_audit.read_text())
    model = torch.load(args.model, map_location='cpu', weights_only=True)
    if (report['review_sha256'] != file_hash(args.review) or
            report['audit_sha256'] != file_hash(args.matrix_audit) or
            report['features_sha256'] != file_hash(args.features) or
            report['checkpoint_sha256'] != file_hash(args.checkpoint) or
            report['trained_model_sha256'] != file_hash(args.model) or
            prior_audit['raw_report_sha256'] != file_hash(args.matrix) or
            prior_audit['review_sha256'] != file_hash(args.review) or
            prior_audit['checkpoint_sha256'] != file_hash(args.checkpoint) or
            report['failures'] or matrix['failures'] or
            len(report['train']) != report['max_rollouts'] or
            len(matrix['pairs']) != 144 or
            model['config']['review_sha256'] != report['review_sha256'] or
            model['config']['audit_sha256'] != report['audit_sha256'] or
            model['config']['features_sha256'] != report['features_sha256'] or
            model['config']['checkpoint_sha256'] !=
                report['checkpoint_sha256'] or
            model['config']['seed'] != report['seed']):
        raise ValueError('Incomplete or changed reward training lineage')
    if (replay['training_report_sha256'] != file_hash(args.report) or
            replay['residual_sha256'] != file_hash(args.model) or
            replay['review_sha256'] != file_hash(args.review) or
            replay['checkpoint_sha256'] != file_hash(args.checkpoint) or
            replay['failures'] or len(replay['games']) != len(report['train']) or
            replay['max_steps'] != 50 or replay['max_new_tokens'] != 64 or
            replay['actor_history_turns'] != 2 or
            replay['loop_guard_max'] != 2):
        raise ValueError('Incomplete or changed full-trajectory reward replay')
    prior = target_reward_matrix(matrix, 8, 18)
    bindings = review['pair_bindings']
    seen = set()
    for index, row in enumerate(report['train']):
        si, ti = row['source_id'], row['target_id']
        if (not 0 <= si < 8 or not 0 <= ti < 18 or
                (si, ti) in seen or row['epoch'] != 0 or
                row['input_content_sha256'] !=
                    bindings[si * 18 + ti]['input_content_sha256'] or
                row['baseline_reward'] != float(prior[ti, si]) or
                row['sampled_reward'] not in (0., 1.) or
                not 1 <= row['steps'] <= 50 or
                len(row['sampled_code']) != report['code_dim'] or
                not all(isinstance(x, (int, float)) and
                        abs(x) < 1e4 for x in row['sampled_code'])):
            raise ValueError(f'Changed or duplicate sampled training row {index}')
        seen.add((si, ti))
        replay_row = replay['games'][index]
        episode = replay_row['episode']
        if (replay_row['training_index'] != index or
                replay_row['source_id'] != si or
                replay_row['target_id'] != ti or
                replay_row['input_content_sha256'] !=
                    row['input_content_sha256'] or
                not replay_row['reward_and_steps_match'] or
                episode['status'] != 'complete' or
                episode['reward'] != row['sampled_reward'] or
                episode['steps'] != row['steps'] or
                episode['steps'] != len(episode['trajectory']) or
                episode['invalid_commands'] != 0 or
                episode['reward'] != float(episode['termination'] == 'success')):
            raise ValueError(f'Unreproduced official reward sample {index}')
    basis = model['basis'].float()
    if (basis.shape != (128, report['code_dim']) or
            not torch.allclose(basis.T @ basis,
                torch.eye(report['code_dim']), atol=1e-4, rtol=1e-4) or
            not all(torch.isfinite(value).all()
                    for value in model['correction'].values())):
        raise ValueError('Invalid trained low-dimensional LoRA residual')
    gain = sum(row['sampled_reward'] > row['baseline_reward']
               for row in report['train'])
    loss = sum(row['sampled_reward'] < row['baseline_reward']
               for row in report['train'])
    summary = {'rollouts': len(report['train']),
        'sampled_success': sum(row['sampled_reward'] for row in report['train']),
        'paired_old_success': sum(row['baseline_reward']
                                  for row in report['train']),
        'sampled_only': gain, 'old_only': loss,
        'informative_reward_differences': gain + loss,
        'max_mean_code_norm': max(row['mean_code_norm']
                                  for row in report['train'])}
    output = {'protocol': 'Content-bound audit of low-dimensional stochastic residual trained from actual official ALFWorld future reward; original actor and hypernetwork checkpoint immutable',
        'report_sha256': file_hash(args.report),
        'trained_model_sha256': file_hash(args.model),
        'replay_sha256': file_hash(args.replay),
        'review_sha256': file_hash(args.review),
        'matrix_audit_sha256': file_hash(args.matrix_audit),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'summary': summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_seed42_36_20261007.json'))
    parser.add_argument('--model', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_seed42_36_20261007.pt'))
    parser.add_argument('--replay', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_seed42_36_replayed_20261007.json'))
    parser.add_argument('--review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--matrix', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_20261007.json'))
    parser.add_argument('--matrix-audit', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_lineage_audited_20261007.json'))
    parser.add_argument('--features', type=Path, default=Path('results/trajectory_hyperlora/alf_future_reward_train8x18_features_20261007.pt'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_seed42_36_audited_20261007.json'))
    audit(parser.parse_args())


if __name__ == '__main__':
    main()
