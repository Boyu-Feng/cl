"""Independently replay anisotropic reward-trained trajectory LoRA episodes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.experience_evolution.environment import make_env
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.train_alf_action_sensitive_reward_v5 import expected


def replay(game: Path, episode: dict, initial: str):
    if (episode['status'] != 'complete' or
            episode['steps'] != len(episode['trajectory']) or
            not 1 <= episode['steps'] <= 50 or
            episode['invalid_commands'] != 0 or
            episode['initial_observation'] != initial):
        raise ValueError('Changed full reward-training episode')
    env = make_env(game)
    try:
        state = env.reset()
        if str(state['feedback']) != initial:
            raise ValueError('Changed official game reset')
        for turn, step in enumerate(episode['trajectory']):
            if (step['turn'] != turn or step['valid'] is not True or
                    step['command'] not in state['admissible_commands']):
                raise ValueError('Unexecutable trained actor action')
            state, _, done = env.step(step['command'])
            if (str(state['feedback']) != step['observation'] or
                    bool(state['won']) != step['won'] or
                    done and turn != episode['steps']-1):
                raise ValueError('Changed official environment transition')
        reward = float(bool(state['won']))
        if (reward != episode['reward'] or episode['termination'] !=
                ('success' if reward else 'budget_or_environment_done')):
            raise ValueError('Changed official terminal reward')
        return reward
    finally:
        env.close()


def audit(args):
    if args.audit_output.exists():
        raise FileExistsError(args.audit_output)
    rows = expected(args)
    review = json.loads(args.train_review.read_text())
    report = json.loads(args.output.read_text())
    old = json.loads(args.reference_report.read_text())
    model = torch.load(args.save_model, map_location='cpu',
                       weights_only=True)
    reference_model = torch.load(args.reference_model, map_location='cpu',
                                 weights_only=True)
    matrix = json.loads(args.matrix.read_text())
    if (review['rows'] != rows or
            report['train_review_sha256'] != file_hash(args.train_review) or
            report['review_sha256'] != file_hash(args.review) or
            report['matrix_sha256'] != file_hash(args.matrix) or
            report['audit_sha256'] != file_hash(args.matrix_audit) or
            report['features_sha256'] != file_hash(args.features) or
            report['reference_report_sha256'] !=
                file_hash(args.reference_report) or
            report['reference_audit_sha256'] !=
                file_hash(args.reference_audit) or
            report['reference_model_sha256'] !=
                file_hash(args.reference_model) or
            report['checkpoint_sha256'] != file_hash(args.checkpoint) or
            report['trained_model_sha256'] != file_hash(args.save_model) or
            report['failures'] or report['max_pairs'] != args.expected_pairs or
            report['max_rollouts'] != args.expected_pairs*2 or
            len(report['train']) != args.expected_pairs or
            len(report['pair_schedule']) != args.expected_pairs or
            model['model_kind'] != 'centered_bilinear_action_sensitive_v5' or
            model['config']['train_review_sha256'] !=
                file_hash(args.train_review) or
            model['config']['checkpoint_sha256'] !=
                file_hash(args.checkpoint) or
            model['config']['features_sha256'] !=
                file_hash(args.features) or
            model['config']['reference_model_sha256'] !=
                file_hash(args.reference_model) or
            not torch.equal(model['basis'], reference_model['basis']) or
            any(not torch.isfinite(value).all() for value in
                model['correction'].values())):
        raise ValueError('Changed or incomplete v5 reward-training lineage')
    successes = {'positive': 0., 'negative': 0.}
    signal = {'positive_only': 0, 'negative_only': 0, 'same': 0}
    baseline = 0.
    targets = set()
    for index, (binding, row) in enumerate(zip(rows, report['train'],
                                              strict=False)):
        if index >= args.expected_pairs:
            break
        si, ti = binding['source_id'], binding['target_id']
        if (row['source_id'] != si or row['target_id'] != ti or
                report['pair_schedule'][index] !=
                    {'source_id': si, 'target_id': ti} or
                row['input_content_sha256'] !=
                    binding['input_content_sha256'] or
                row['source_target_binding'] !=
                    binding['source_target_binding'] or
                row['baseline_reward'] !=
                    matrix['pairs'][si*18+ti]['episode']['reward'] or
                row['action_one'] != binding['action_one'] or
                row['action_two'] != binding['action_two'] or
                row['noise'] != old['train'][index]['noise']):
            raise ValueError('Changed reviewed source-target perturbation')
        direction = torch.tensor(row['direction'], dtype=torch.float64)
        noise = torch.tensor(row['noise'], dtype=torch.float64)
        delta = torch.tensor(row['delta'], dtype=torch.float64)
        precision = torch.tensor(row['precision_delta'],
                                 dtype=torch.float64)
        positive_code = torch.tensor(row['positive_code'],
                                     dtype=torch.float64)
        negative_code = torch.tensor(row['negative_code'],
                                     dtype=torch.float64)
        if (any(x.shape != (8,) or not torch.isfinite(x).all() for x in
                (direction, noise, delta, precision,
                 positive_code, negative_code)) or
                abs(direction.norm().item()-1.) > 1e-5 or
                row['direction_gradient_norm'] < 1e-8 or
                not torch.allclose(positive_code-negative_code,
                    2*delta, atol=1e-5, rtol=1e-5)):
            raise ValueError('Broken antithetic parameter factors')
        parallel = torch.dot(noise, direction)*direction
        orthogonal = noise-parallel
        if (not torch.allclose(delta,
                report['sigma_parallel']*parallel+
                report['sigma_perp']*orthogonal, atol=1e-5, rtol=1e-5) or
            not torch.allclose(precision,
                parallel/report['sigma_parallel']+
                orthogonal/report['sigma_perp'], atol=1e-5, rtol=1e-5)):
            raise ValueError('Broken inverse-covariance reward gradient')
        initial = matrix['base'][ti]['episode']['initial_observation']
        for sign in ('positive', 'negative'):
            reward = replay(args.data_root / binding['game'],
                row[sign], initial)
            if reward != row[f'{sign}_reward']:
                raise ValueError('Changed logged antithetic reward')
            successes[sign] += reward
        if row['positive_reward'] > row['negative_reward']:
            signal['positive_only'] += 1
        elif row['positive_reward'] < row['negative_reward']:
            signal['negative_only'] += 1
        else:
            signal['same'] += 1
        baseline += row['baseline_reward']
        targets.add(ti)
    summary = {'pairs': args.expected_pairs,
        'rollouts': args.expected_pairs*2,
        'different_targets': len(targets),
        'paired_old_success': baseline,
        'positive_success': successes['positive'],
        'negative_success': successes['negative'],
        **signal, 'nonzero_reward_pairs':
            signal['positive_only']+signal['negative_only'],
        'last_mean_code_norm': report['train'][-1]['mean_code_norm'],
        'failed_replays': 0}
    result = {'protocol': 'Independent original ALFWorld replay of reviewed anisotropic score-function hyper-LoRA reward training; Gaussian draws and covariance inverse checked against same-budget isotropic reference',
        'train_review_sha256': file_hash(args.train_review),
        'raw_report_sha256': file_hash(args.output),
        'trained_model_sha256': file_hash(args.save_model),
        'reference_report_sha256': file_hash(args.reference_report),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'summary': summary}
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(result, ensure_ascii=False,
                                             indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--source-report-seq0', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq0_6_20261007.json'))
    parser.add_argument('--source-report-seq6', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq6_6_20261007.json'))
    parser.add_argument('--plan', type=Path, default=Path('ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json'))
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--target-policy', choices=('train18',), default='train18')
    parser.add_argument('--checkpoint-training-candidates', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json'))
    parser.add_argument('--warmstart-candidates', type=Path, default=Path('results/trajectory_hyperlora/alf_next_task_candidates_20261005.json'))
    parser.add_argument('--prior-train-review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--matrix', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_20261007.json'))
    parser.add_argument('--matrix-audit', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_lineage_audited_20261007.json'))
    parser.add_argument('--features', type=Path, default=Path('results/trajectory_hyperlora/alf_future_reward_train8x18_features_20261007.pt'))
    parser.add_argument('--reference-report', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_antithetic_v4_seed42_20261007.json'))
    parser.add_argument('--reference-audit', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_antithetic_v4_seed42_audited_20261007.json'))
    parser.add_argument('--reference-model', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_antithetic_v4_seed42_20261007.pt'))
    parser.add_argument('--train-review', type=Path, default=Path('data/annotations/alf_action_sensitive_reward_train72_reviewed_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_20261007.json'))
    parser.add_argument('--save-model', type=Path, default=Path('results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_20261007.pt'))
    parser.add_argument('--audit-output', type=Path, default=Path('results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_audited_20261007.json'))
    parser.add_argument('--expected-pairs', type=int, default=72)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sigma-parallel', type=float, default=1.25)
    parser.add_argument('--sigma-perp', type=float, default=.25)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--kl-weight', type=float, default=.001)
    args = parser.parse_args()
    audit(args)
