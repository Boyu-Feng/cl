"""Independently replay equal-norm ALFWorld LoRA exploration rollouts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from ttcl.experience_evolution.environment import make_env
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alf_action_sensitive_antithetic_v5 import expected


BRANCHES = ('random_positive', 'random_negative',
            'sensitive_positive', 'sensitive_negative')


def replay(game: Path, episode: dict, initial: str,
           initial_commands_sha256: str):
    if (episode['status'] != 'complete' or
            episode['steps'] != len(episode['trajectory']) or
            not 1 <= episode['steps'] <= 50 or
            episode['invalid_commands'] != 0 or
            episode['initial_observation'] != initial or
            episode['initial_commands_sha256'] != initial_commands_sha256):
        raise ValueError('Changed pilot episode structure')
    env = make_env(game)
    try:
        state = env.reset()
        if (str(state['feedback']) != initial or
                hashlib.sha256(json.dumps(list(state['admissible_commands']))
                                .encode()).hexdigest() != initial_commands_sha256):
            raise ValueError('Changed original task reset')
        for turn, row in enumerate(episode['trajectory']):
            if (row['turn'] != turn or row['valid'] is not True or
                    row['command'] not in state['admissible_commands']):
                raise ValueError('Unexecutable pilot actor command')
            state, _, done = env.step(row['command'])
            if (str(state['feedback']) != row['observation'] or
                    bool(state['won']) != row['won'] or
                    done and turn != episode['steps']-1):
                raise ValueError('Changed original task transition')
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
    review = json.loads(args.pilot_review.read_text())
    run = json.loads(args.output.read_text())
    rows = expected(args)
    if (review['rows'] != rows or
            run['pilot_review_sha256'] != file_hash(args.pilot_review) or
            run['checkpoint_sha256'] != file_hash(args.checkpoint) or
            run['prior_model_sha256'] != file_hash(args.prior_model) or
            run['sigma'] != args.sigma or run['seed'] != args.seed or
            run['max_steps'] != 50 or run['max_new_tokens'] != 64 or
            run['actor_history_turns'] != 2 or
            run['loop_guard_max'] != 2 or
            run['failures'] or len(rows) != 24 or len(run['rows']) != 24):
        raise ValueError('Changed or incomplete exploration input')
    pair_counts = {'random': 0, 'sensitive': 0}
    first_action_different = {'random': 0, 'sensitive': 0}
    reward_totals = {branch: 0. for branch in BRANCHES}
    target_difference = set()
    for index, (source, observed) in enumerate(zip(rows, run['rows'],
                                                  strict=True)):
        if (source['input_content_sha256'] !=
                observed['input_content_sha256'] or
                observed['source_id'] != source['source_id'] or
                observed['target_id'] != source['target_id'] or
                observed['source_tokens_retained'] > 2048 or
                set(observed['episodes']) != set(BRANCHES)):
            raise ValueError('Changed source-target perturbation binding')
        gradient = torch.tensor(observed['gradient'], dtype=torch.float64)
        random_delta = torch.tensor(observed['random_delta'],
                                    dtype=torch.float64)
        sensitive_delta = torch.tensor(observed['sensitive_delta'],
                                       dtype=torch.float64)
        if (any(value.shape != (8,) for value in
               (gradient, random_delta, sensitive_delta)) or not all(
                   torch.isfinite(value).all() for value in
                   (gradient, random_delta, sensitive_delta)) or
                gradient.norm() < 1e-8 or
                abs(random_delta.norm().item()-args.sigma*math.sqrt(8)) > 1e-5 or
                abs(sensitive_delta.norm().item()-args.sigma*math.sqrt(8)) > 1e-5 or
                torch.dot(sensitive_delta, gradient).item() <= 0 or
                torch.nn.functional.cosine_similarity(
                    sensitive_delta, gradient, dim=0).item() < .99999):
            raise ValueError('Changed actor-sensitive perturbation geometry')
        generator = torch.Generator(device='cpu').manual_seed(args.seed+index)
        expected_random = torch.randn(8, generator=generator).double()
        expected_random *= args.sigma*math.sqrt(8)/expected_random.norm()
        if not torch.allclose(random_delta, expected_random,
                              rtol=1e-5, atol=1e-5):
            raise ValueError('Changed precommitted isotropic control')
        values = {}
        for branch in BRANCHES:
            if (digest(observed['episodes'][branch]['initial_observation']) !=
                    source['initial_observation_sha256']):
                raise ValueError('Changed reviewed initial observation')
            values[branch] = replay(args.data_root / source['game'],
                observed['episodes'][branch],
                observed['episodes'][branch]['initial_observation'],
                source['initial_commands_sha256'])
            reward_totals[branch] += values[branch]
        for method in pair_counts:
            if (observed['episodes'][f'{method}_positive']['trajectory'][0]
                    ['command'] !=
                    observed['episodes'][f'{method}_negative']['trajectory'][0]
                    ['command']):
                first_action_different[method] += 1
            if (values[f'{method}_positive'] !=
                    values[f'{method}_negative']):
                pair_counts[method] += 1
                target_difference.add(source['target_id'])
    summary = {'pairs': len(rows), 'episodes': len(rows)*4,
        'different_targets': len({row['target_id'] for row in rows}),
        'first_action_different_pairs': first_action_different,
        'reward_different_pairs': pair_counts,
        'targets_with_any_reward_difference': len(target_difference),
        'reward_totals': reward_totals, 'failed_replays': 0}
    result = {'protocol': 'Independent original-ALFWorld replay of all equal-norm random and actor-sensitive antithetic LoRA episodes; source-target-perturbation bindings and deterministic controls checked; train-domain mechanism probe only',
        'pilot_review_sha256': file_hash(args.pilot_review),
        'raw_report_sha256': file_hash(args.output),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'prior_model_sha256': file_hash(args.prior_model),
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
    parser.add_argument('--matrix-audit', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_audited_20261007.json'))
    parser.add_argument('--prior-train', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_antithetic_v4_seed42_20261007.json'))
    parser.add_argument('--prior-audit', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_antithetic_v4_seed42_audited_20261007.json'))
    parser.add_argument('--prior-model', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_antithetic_v4_seed42_20261007.pt'))
    parser.add_argument('--pilot-review', type=Path, default=Path('data/annotations/alf_action_sensitive_antithetic24_reviewed_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_action_sensitive_antithetic24_v5_20261007.json'))
    parser.add_argument('--audit-output', type=Path, default=Path('results/trajectory_hyperlora/alf_action_sensitive_antithetic24_v5_audited_20261007.json'))
    parser.add_argument('--max-pairs', type=int, default=24)
    parser.add_argument('--sigma', type=float, default=.5)
    parser.add_argument('--seed', type=int, default=20261007)
    args = parser.parse_args()
    audit(args)
