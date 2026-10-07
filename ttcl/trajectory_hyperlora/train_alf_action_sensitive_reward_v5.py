"""Train a trajectory-to-LoRA residual with actor-sensitive reward exploration.

The same 72 reviewed source-target pairs, Gaussian draws, frozen actor,
feature basis and rollout budget as v4 are reused. Only the sampling
covariance and its correct score-function gradient change. Every rollout is
an official ALFWorld train episode; this is not actor policy-gradient RL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

from ttcl.experience_evolution.environment import make_env
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v2 import checked, save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.centered_feedback_residual_v3 import CenteredEvidenceResidual
from ttcl.trajectory_hyperlora.train_alf_feedback_residual_v4_antithetic import (
    ResidualInjection, source_fields,
)
from ttcl.trajectory_hyperlora.alf_action_sensitive_antithetic_v5 import sensitive_direction


def inputs(args):
    sources, targets, bindings = checked(args)
    review = json.loads(args.review.read_text())
    matrix = json.loads(args.matrix.read_text())
    matrix_audit = json.loads(args.matrix_audit.read_text())
    features = torch.load(args.features, map_location='cpu', weights_only=True)
    reference = json.loads(args.reference_report.read_text())
    reference_audit = json.loads(args.reference_audit.read_text())
    reference_model = torch.load(args.reference_model, map_location='cpu',
                                 weights_only=True)
    if (args.target_policy != 'train18' or
            matrix_audit['target_policy'] != 'train18' or
            matrix_audit['review_sha256'] != file_hash(args.review) or
            matrix_audit['raw_report_sha256'] != file_hash(args.matrix) or
            matrix_audit['checkpoint_sha256'] != file_hash(args.checkpoint) or
            matrix['review_sha256'] != file_hash(args.review) or
            matrix['checkpoint_sha256'] != file_hash(args.checkpoint) or
            matrix['failures'] or len(matrix['base']) != 18 or
            len(matrix['pairs']) != 144 or
            features['review_sha256'] != file_hash(args.review) or
            features['checkpoint_sha256'] != file_hash(args.checkpoint) or
            features['target_policy'] != 'train18' or
            reference_audit['report_sha256'] != file_hash(args.reference_report) or
            reference_audit['model_sha256'] != file_hash(args.reference_model) or
            reference['audit_sha256'] != file_hash(args.matrix_audit) or
            reference['features_sha256'] != file_hash(args.features) or
            reference['checkpoint_sha256'] != file_hash(args.checkpoint) or
            reference['review_sha256'] != file_hash(args.review) or
            reference['failures'] or len(reference['train']) != 72 or
            len(reference['pair_schedule']) != 72 or
            reference_model['model_kind'] != 'centered_bilinear_antithetic' or
            reference_model['basis'].shape != (128, 8) or
            reference_model['config']['features_sha256'] !=
                file_hash(args.features) or len(sources) != 8 or
            len(targets) != 18 or len(bindings) != 144 or
            len(review['pair_bindings']) != 144 or
            args.seed != 42 or args.sigma_parallel != 1.25 or
            args.sigma_perp != .25 or args.lr != .001 or
            args.kl_weight != .001):
        raise ValueError('Changed fixed same-budget anisotropic reward protocol')
    arms = {}
    first = defaultdict(dict)
    for row, binding in zip(matrix['pairs'], bindings, strict=True):
        key = row['source_id'], row['target_id']
        if (key in arms or row['input_content_sha256'] !=
                binding['input_content_sha256'] or
                row['episode']['status'] != 'complete' or
                row['episode']['steps'] < 1):
            raise ValueError('Changed frozen source-target arm')
        arms[key] = row
        first[row['target_id']][row['source_id']] = (
            row['episode']['trajectory'][0]['command'])
    if len(arms) != 144:
        raise ValueError('Incomplete reward ancestry')
    return sources, targets, matrix, features, reference, reference_model, arms, first


def expected(args):
    _, targets, matrix, _, reference, _, arms, first = inputs(args)
    initial_states = {}
    rows = []
    for index, scheduled in enumerate(reference['pair_schedule']):
        si, ti = scheduled['source_id'], scheduled['target_id']
        episode = arms[si, ti]['episode']
        target = targets[ti]
        if ti not in initial_states:
            env = make_env(args.data_root / target['game'])
            try:
                state = env.reset()
                available = list(state['admissible_commands'])
                initial = str(state['feedback'])
            finally:
                env.close()
            if (initial != episode['initial_observation'] or
                    hashlib.sha256(json.dumps(available).encode()).hexdigest() !=
                    episode['initial_commands_sha256']):
                raise ValueError('Changed official target reset')
            initial_states[ti] = available
        available = initial_states[ti]
        action_one = first[ti][si]
        action_two = next((first[ti][j] for j in sorted(first[ti])
                           if first[ti][j] != action_one), None)
        if action_two is None:
            action_two = next((x for x in available if x != action_one), None)
        if (action_one not in available or action_two not in available or
                action_one == action_two or
                reference['train'][index]['source_id'] != si or
                reference['train'][index]['target_id'] != ti):
            raise ValueError('Changed precommitted actor action contrast')
        noise = reference['train'][index]['noise']
        if len(noise) != 8:
            raise ValueError('Changed Gaussian draw in reference schedule')
        content = {'source_id': si, 'target_id': ti,
            'target_game_sha256': target['game_sha256'],
            'source_target_binding': arms[si, ti]['input_content_sha256'],
            'reference_episode_sha256': digest(episode),
            'action_one': action_one, 'action_two': action_two,
            'initial_commands_sha256': episode['initial_commands_sha256'],
            'reference_noise_sha256': digest(noise),
            'reference_report_sha256': file_hash(args.reference_report),
            'matrix_sha256': file_hash(args.matrix),
            'checkpoint_sha256': file_hash(args.checkpoint),
            'sigma_parallel': args.sigma_parallel,
            'sigma_perp': args.sigma_perp,
            'max_steps': 50, 'max_new_tokens': 64,
            'actor_history_turns': 2, 'loop_guard_max': 2}
        rows.append({'game': target['game'], **content,
                     'input_content_sha256': digest(content)})
    if len(rows) != 72 or len({(x['source_id'], x['target_id'])
                                for x in rows}) != 72:
        raise ValueError('Changed frozen pair schedule')
    return rows


def prepare(args):
    if args.train_review.exists():
        raise FileExistsError(args.train_review)
    rows = expected(args)
    result = {'protocol': 'New reviewed source-target-perturbation inputs for same 72 official train pairs and Gaussian draws as v4; actor first-decision contrast from observed admissible commands; no new reward used in candidate selection',
        'original_review_sha256': file_hash(args.review),
        'matrix_sha256': file_hash(args.matrix),
        'matrix_audit_sha256': file_hash(args.matrix_audit),
        'features_sha256': file_hash(args.features),
        'reference_report_sha256': file_hash(args.reference_report),
        'reference_audit_sha256': file_hash(args.reference_audit),
        'reference_model_sha256': file_hash(args.reference_model),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'sigma_parallel': args.sigma_parallel,
        'sigma_perp': args.sigma_perp,
        'rows': rows}
    args.train_review.parent.mkdir(parents=True, exist_ok=True)
    args.train_review.write_text(json.dumps(result, ensure_ascii=False,
                                            indent=2)+'\n')
    print(json.dumps({'reviewed_pairs': len(rows),
        'different_targets': len({row['target_id'] for row in rows})}),
        flush=True)


def train(args):
    if args.output.exists() or args.save_model.exists():
        raise FileExistsError('Use fresh reward-training report and model')
    if not 1 <= args.max_pairs <= 72:
        raise ValueError('Pair budget exceeds reviewed training schedule')
    sources, targets, matrix, features, reference, saved, _, _ = inputs(args)
    rows = expected(args)
    review = json.loads(args.train_review.read_text())
    if (review['rows'] != rows or
            review['original_review_sha256'] != file_hash(args.review) or
            review['matrix_sha256'] != file_hash(args.matrix) or
            review['matrix_audit_sha256'] != file_hash(args.matrix_audit) or
            review['features_sha256'] != file_hash(args.features) or
            review['reference_report_sha256'] != file_hash(args.reference_report) or
            review['reference_audit_sha256'] != file_hash(args.reference_audit) or
            review['reference_model_sha256'] != file_hash(args.reference_model) or
            review['checkpoint_sha256'] != file_hash(args.checkpoint) or
            review['sigma_parallel'] != args.sigma_parallel or
            review['sigma_perp'] != args.sigma_perp):
        raise ValueError('Changed new trajectory annotation targets')
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != 'contextual' or
            agent.task_context_scope != 'current' or
            agent.task_pair_pooling != 'mean'):
        raise ValueError('Changed frozen actor family')
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    basis = saved['basis'].to(args.device).float()
    correction = CenteredEvidenceResidual.from_state(
        saved['correction']).to(args.device)
    correction.load_state_dict(saved['correction'])
    with torch.no_grad():
        correction.weight.zero_()
    optimizer = torch.optim.AdamW(correction.parameters(), lr=args.lr,
                                  weight_decay=0)
    fields = source_fields(agent, tokenizer, sources, args)
    deltas = torch.stack([features['deltas'][i, :int(count)].mean(0)
        for i, count in enumerate(features['event_counts'])])
    report = {'protocol': 'Train-only official ALFWorld environment-reward hyper-LoRA: same 72 source-target pairs, Gaussian draws, fixed feature/basis/actor and 144 rollout budget as v4; actor-sensitive detached covariance, exact inverse-covariance antithetic mean score gradient; no actor-policy-gradient or online-generalization claim',
        'seed': args.seed, 'max_pairs': args.max_pairs,
        'max_rollouts': 2*args.max_pairs, 'sigma_parallel': args.sigma_parallel,
        'sigma_perp': args.sigma_perp, 'lr': args.lr,
        'kl_weight': args.kl_weight, 'code_dim': 8,
        'train_review_sha256': file_hash(args.train_review),
        'review_sha256': file_hash(args.review),
        'matrix_sha256': file_hash(args.matrix),
        'audit_sha256': file_hash(args.matrix_audit),
        'features_sha256': file_hash(args.features),
        'reference_report_sha256': file_hash(args.reference_report),
        'reference_audit_sha256': file_hash(args.reference_audit),
        'reference_model_sha256': file_hash(args.reference_model),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'pair_schedule': [{'source_id': x['source_id'],
                           'target_id': x['target_id']}
                          for x in rows[:args.max_pairs]],
        'train': [], 'failures': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, report)
    completed = False
    try:
        for index, row in enumerate(rows[:args.max_pairs]):
            si, ti = row['source_id'], row['target_id']
            old = reference['train'][index]
            game = args.data_root / targets[ti]['game']
            env = make_env(game)
            try:
                state = env.reset()
                initial = str(state['feedback'])
                available = list(state['admissible_commands'])
            finally:
                env.close()
            gradient, margin = sensitive_direction(agent, tokenizer,
                fields[si], initial, available,
                (row['action_one'], row['action_two']), basis, args.device)
            direction = gradient/gradient.norm()
            noise = torch.tensor(old['noise'], device=args.device,
                                 dtype=torch.float32)
            parallel = torch.dot(noise, direction)*direction
            orthogonal = noise-parallel
            shift = (args.sigma_parallel*parallel+
                     args.sigma_perp*orthogonal)
            precision_shift = (parallel/args.sigma_parallel+
                               orthogonal/args.sigma_perp)
            mean = correction(features['global'][si].to(args.device),
                deltas[si].to(args.device),
                features['targets'][ti].to(args.device))
            episodes = []
            for sign in (1, -1):
                sampled = mean.detach()+sign*shift
                with ResidualInjection(agent, sampled @ basis.T):
                    with torch.no_grad():
                        episode = run_episode(agent, tokenizer, game,
                            fields[si], adapter=True, device=args.device,
                            max_steps=50, max_new_tokens=64,
                            constrain_actions=True, actor_history_turns=2,
                            loop_guard_max=2)
                if (episode['status'] != 'complete' or
                        episode['invalid_commands'] != 0 or
                        episode['initial_observation'] != initial):
                    raise ValueError('Incomplete or changed anisotropic reward sample')
                episodes.append(episode)
            positive, negative = episodes
            advantage = float(positive['reward']-negative['reward'])
            loss = (-.5*advantage*(mean*precision_shift).sum() +
                    args.kl_weight*mean.square().sum()/(2*.25))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                correction.parameters(), 1.)
            optimizer.step()
            report['train'].append({'source_id': si, 'target_id': ti,
                'input_content_sha256': row['input_content_sha256'],
                'source_target_binding': row['source_target_binding'],
                'baseline_reward': matrix['pairs'][si*18+ti]
                    ['episode']['reward'],
                'action_one': row['action_one'],
                'action_two': row['action_two'],
                'initial_action_margin': margin,
                'direction': direction.detach().cpu().tolist(),
                'direction_gradient_norm': float(gradient.norm()),
                'noise': noise.cpu().tolist(),
                'delta': shift.detach().cpu().tolist(),
                'precision_delta': precision_shift.detach().cpu().tolist(),
                'mean_code_norm': float(mean.detach().norm()),
                'positive_code': (mean.detach()+shift).cpu().tolist(),
                'negative_code': (mean.detach()-shift).cpu().tolist(),
                'positive_reward': float(positive['reward']),
                'negative_reward': float(negative['reward']),
                'positive': positive, 'negative': negative,
                'loss': float(loss.detach()),
                'grad_norm': float(grad_norm)})
            save(args.output, report)
            print(json.dumps({'pairs': len(report['train']),
                'positive_reward': positive['reward'],
                'negative_reward': negative['reward'],
                'mean_norm': float(mean.detach().norm()),
                'failures': 0}), flush=True)
        completed = True
    except Exception as exc:
        report['failures'].append({'pair_index': len(report['train']),
                                   'reason': repr(exc)})
        raise
    finally:
        if completed:
            args.save_model.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model_kind': 'centered_bilinear_action_sensitive_v5',
                'correction': correction.cpu().state_dict(),
                'basis': basis.detach().cpu(),
                'config': {key: report[key] for key in (
                    'protocol', 'seed', 'max_pairs', 'max_rollouts',
                    'sigma_parallel', 'sigma_perp', 'lr', 'kl_weight',
                    'code_dim', 'train_review_sha256', 'review_sha256',
                    'audit_sha256', 'features_sha256',
                    'reference_model_sha256', 'checkpoint_sha256')}},
                args.save_model)
            report['trained_model_sha256'] = file_hash(args.save_model)
        save(args.output, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'train'))
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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-pairs', type=int, default=72)
    parser.add_argument('--sigma-parallel', type=float, default=1.25)
    parser.add_argument('--sigma-perp', type=float, default=.25)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--kl-weight', type=float, default=.001)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.7)
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else train)(args)
