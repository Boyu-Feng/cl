"""Audit reward-stratified sampled LoRA training with complete trajectories."""

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
    parent = json.loads(args.matrix_audit.read_text())
    model = torch.load(args.model, map_location='cpu', weights_only=True)
    if (report['review_sha256'] != file_hash(args.review) or
            report['audit_sha256'] != file_hash(args.matrix_audit) or
            report['features_sha256'] != file_hash(args.features) or
            report['checkpoint_sha256'] != file_hash(args.checkpoint) or
            report['trained_model_sha256'] != file_hash(args.model) or
            parent['raw_report_sha256'] != file_hash(args.matrix) or
            parent['review_sha256'] != file_hash(args.review) or
            parent['checkpoint_sha256'] != file_hash(args.checkpoint) or
            report['failures'] or matrix['failures'] or
            len(report['train']) != report['max_rollouts'] or
            len(report['pair_schedule']) != len(report['train']) or
            len(matrix['pairs']) != 144 or
            model['config']['review_sha256'] != report['review_sha256'] or
            model['config']['audit_sha256'] != report['audit_sha256'] or
            model['config']['features_sha256'] != report['features_sha256'] or
            model['config']['checkpoint_sha256'] !=
                report['checkpoint_sha256'] or
            model['config']['seed'] != report['seed']):
        raise ValueError('Incomplete or changed reward training lineage')
    old = target_reward_matrix(matrix, 8, 18)
    seen = set()
    count = {0.: 0, 1.: 0}
    for index, row in enumerate(report['train']):
        si, ti = row['source_id'], row['target_id']
        episode = row['episode']
        if (not 0 <= si < 8 or not 0 <= ti < 18 or
                (si, ti) in seen or
                report['pair_schedule'][index] !=
                    {'source_id': si, 'target_id': ti} or
                row['epoch'] != 0 or
                row['input_content_sha256'] !=
                    review['pair_bindings'][si * 18 + ti]['input_content_sha256'] or
                row['baseline_reward'] != float(old[ti, si]) or
                row['sampled_reward'] not in (0., 1.) or
                len(row['sampled_code']) != report['code_dim'] or
                episode['status'] != 'complete' or
                episode['steps'] != row['steps'] or
                episode['steps'] != len(episode['trajectory']) or
                not 1 <= episode['steps'] <= 50 or
                episode['invalid_commands'] != 0 or
                episode['reward'] != row['sampled_reward'] or
                episode['reward'] != float(episode['termination'] == 'success') or
                episode['initial_observation'] !=
                    matrix['base'][ti]['episode']['initial_observation']):
            raise ValueError(f'Changed official reward sample {index}')
        seen.add((si, ti))
        count[row['baseline_reward']] += 1
    if abs(count[0.] - count[1.]) > 1:
        raise ValueError('Changed reward-stratified sample schedule')
    basis = model['basis'].float()
    if (basis.shape != (128, report['code_dim']) or
            not torch.allclose(basis.T @ basis,
                torch.eye(report['code_dim']), atol=1e-4, rtol=1e-4) or
            not all(torch.isfinite(value).all()
                    for value in model['correction'].values())):
        raise ValueError('Invalid trained low-dimensional residual')
    summary = {'rollouts': len(report['train']),
        'baseline_zero': count[0.], 'baseline_one': count[1.],
        'sampled_success': sum(x['sampled_reward'] for x in report['train']),
        'paired_old_success': sum(x['baseline_reward'] for x in report['train']),
        'sampled_only': sum(x['sampled_reward'] > x['baseline_reward']
                             for x in report['train']),
        'old_only': sum(x['sampled_reward'] < x['baseline_reward']
                         for x in report['train']),
        'last_mean_code_norm': report['train'][-1]['mean_code_norm'],
        'max_mean_code_norm': max(x['mean_code_norm']
                                  for x in report['train'])}
    result = {'protocol': 'Content-bound full-trajectory audit of balanced stochastic latent LoRA residual training from official ALFWorld reward',
        'report_sha256': file_hash(args.report),
        'model_sha256': file_hash(args.model),
        'matrix_audit_sha256': file_hash(args.matrix_audit),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'summary': summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_balanced_v2_20261007.json'))
    parser.add_argument('--model', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_balanced_v2_20261007.pt'))
    parser.add_argument('--review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--matrix', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_20261007.json'))
    parser.add_argument('--matrix-audit', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_lineage_audited_20261007.json'))
    parser.add_argument('--features', type=Path, default=Path('results/trajectory_hyperlora/alf_future_reward_train8x18_features_20261007.pt'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_balanced_v2_audited_20261007.json'))
    audit(parser.parse_args())


if __name__ == '__main__':
    main()
