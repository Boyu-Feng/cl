"""Replay sampled LoRA codes to validate reward-training episode records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v2 import checked, save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.train_alf_feedback_residual_v1 import (
    ResidualInjection, source_fields,
)


def replay(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    sources, targets, _ = checked(args)
    training = json.loads(args.training_report.read_text())
    matrix = json.loads(args.matrix.read_text())
    model = torch.load(args.residual, map_location='cpu', weights_only=True)
    if (args.target_policy != 'train18' or
            training['review_sha256'] != file_hash(args.review) or
            training['checkpoint_sha256'] != file_hash(args.checkpoint) or
            training['trained_model_sha256'] != file_hash(args.residual) or
            training['failures'] or
            len(training['train']) != training['max_rollouts'] or
            model['config']['review_sha256'] != file_hash(args.review) or
            model['config']['checkpoint_sha256'] !=
                file_hash(args.checkpoint) or len(matrix['base']) != len(targets)):
        raise ValueError('Changed or incomplete sampled training run')
    basis = model['basis'].to(args.device)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    agent.eval()
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    fields = source_fields(agent, tokenizer, sources, args)
    result = {'protocol': 'Deterministic full-trajectory replay of every saved stochastic low-dimensional LoRA code under the original frozen actor and official ALFWorld environment',
        'training_report_sha256': file_hash(args.training_report),
        'residual_sha256': file_hash(args.residual),
        'review_sha256': file_hash(args.review),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'max_steps': 50, 'max_new_tokens': 64,
        'actor_history_turns': 2, 'loop_guard_max': 2,
        'games': [], 'failures': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(training['train']):
        si, ti = row['source_id'], row['target_id']
        code = torch.tensor(row['sampled_code'], device=args.device,
                             dtype=basis.dtype)
        shift = code @ basis.T
        with ResidualInjection(agent, shift):
            episode = run_episode(agent, tokenizer,
                args.data_root / targets[ti]['game'], fields[si],
                adapter=True, device=args.device, max_steps=50,
                max_new_tokens=64, constrain_actions=True,
                actor_history_turns=2, loop_guard_max=2)
        match = (episode['status'] == 'complete' and
            episode['reward'] == row['sampled_reward'] and
            episode['steps'] == row['steps'] and
            episode['initial_observation'] ==
                matrix['base'][ti]['episode']['initial_observation'])
        result['games'].append({'training_index': index,
            'source_id': si, 'target_id': ti,
            'input_content_sha256': row['input_content_sha256'],
            'episode': episode, 'reward_and_steps_match': match})
        if not match:
            result['failures'].append({'training_index': index,
                'reason': 'replay mismatch'})
        save(args.output, result)
        print(json.dumps({'replayed': len(result['games']),
            'failures': len(result['failures'])}), flush=True)
        if not match:
            raise RuntimeError('Stochastic LoRA code failed deterministic replay')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--residual', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_seed42_36_20261007.pt'))
    parser.add_argument('--training-report', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_seed42_36_20261007.json'))
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
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_seed42_36_replayed_20261007.json'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.65)
    replay(parser.parse_args())


if __name__ == '__main__':
    main()
