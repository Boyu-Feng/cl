"""Evaluate frozen centered evidence LoRA residual on reviewed development tasks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v2 import checked, save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.train_alf_feedback_residual_v1 import (
    ResidualInjection, source_fields,
)
from ttcl.trajectory_hyperlora.centered_feedback_residual_v3 import (
    CenteredEvidenceResidual,
)
from ttcl.trajectory_hyperlora.train_alf_future_utility_selector_v1 import target_reward_matrix


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if not math.isfinite(args.residual_scale) or args.residual_scale == 0:
        raise ValueError('Need finite nonzero residual scale')
    sources, targets, _ = checked(args)
    model = torch.load(args.residual, map_location='cpu', weights_only=True)
    features = torch.load(args.features, map_location='cpu', weights_only=True)
    matrix = json.loads(args.matrix.read_text())
    audit = json.loads(args.audit.read_text())
    training = json.loads(args.training_report.read_text())
    config = model['config']
    if (args.target_policy != 'independent12' or
            audit['target_policy'] != 'independent12' or
            audit['review_sha256'] != file_hash(args.review) or
            audit['raw_report_sha256'] != file_hash(args.matrix) or
            audit['checkpoint_sha256'] != file_hash(args.checkpoint) or
            any(audit['summary'][key] for key in (
                'checkpoint_training_candidate_overlap',
                'warmstart_training_target_overlap',
                'actual_checkpoint_label_target_overlap',
                'prior_training_game_overlap',
                'prior_training_sequence_overlap')) or
            features['target_policy'] != 'independent12' or
            features['review_sha256'] != file_hash(args.review) or
            features['checkpoint_sha256'] != file_hash(args.checkpoint) or
            digest([row['episode']['initial_observation']
                    for row in matrix['base']]) !=
                    features['base_observations_sha256'] or
            matrix['failures'] or len(matrix['pairs']) != 96 or
            config['checkpoint_sha256'] != audit['checkpoint_sha256'] or
            config['audit_sha256'] != training['audit_sha256'] or
            training['trained_model_sha256'] != file_hash(args.residual) or
            len(sources) != 8 or len(targets) != 12):
        raise ValueError('Changed reward model or non-disjoint development data')
    rewards = target_reward_matrix(matrix, 8, 12)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    agent.eval()
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    source_bank = source_fields(agent, tokenizer, sources, args)
    if model.get('model_kind') != 'centered_bilinear':
        raise ValueError('Expected centered bilinear correction')
    residual = CenteredEvidenceResidual.from_state(
        model['correction']).to(args.device)
    residual.load_state_dict(model['correction'])
    residual.eval()
    basis = model['basis'].to(args.device)
    if basis.shape != (128, config['code_dim']):
        raise ValueError('Changed latent residual basis')
    deltas = torch.stack([features['deltas'][i, :int(count)].mean(0)
        for i, count in enumerate(features['event_counts'])])
    result = {'protocol': 'Previously exposed official ALFWorld train-domain 12-task development comparison; frozen reward-trained feedback residual, same actor/source/target and budget as old matrix; not blind final validation',
        'residual_sha256': file_hash(args.residual),
        'training_report_sha256': file_hash(args.training_report),
        'review_sha256': file_hash(args.review),
        'audit_sha256': file_hash(args.audit),
        'features_sha256': file_hash(args.features),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'source_id': args.source_id,
        'residual_scale': args.residual_scale,
        'max_steps': 50, 'max_new_tokens': 64,
        'actor_history_turns': 2, 'loop_guard_max': 2,
        'games': [], 'failures': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for target in targets:
        ti = target['target_id']
        si = args.source_id
        with torch.no_grad():
            mean = residual(features['global'][si].to(args.device),
                deltas[si].to(args.device),
                features['targets'][ti].to(args.device))
            shift = args.residual_scale * (mean @ basis.T)
            with ResidualInjection(agent, shift):
                episode = run_episode(agent, tokenizer,
                    args.data_root / target['game'], source_bank[si],
                    adapter=True, device=args.device, max_steps=50,
                    max_new_tokens=64, constrain_actions=True,
                    actor_history_turns=2, loop_guard_max=2)
        if (episode['status'] != 'complete' or
                episode['initial_observation'] !=
                matrix['base'][ti]['episode']['initial_observation']):
            result['failures'].append({'target_id': ti,
                'episode': episode})
            save(args.output, result)
            raise RuntimeError('Changed or failed reward evaluation')
        row = {'target_id': ti, 'game': target['game'],
            'game_sha256': target['game_sha256'],
            'old_source_reward': float(rewards[ti, si]),
            'base_reward': matrix['base'][ti]['episode']['reward'],
            'new': episode, 'mean_code_norm': float(mean.norm())}
        result['games'].append(row)
        save(args.output, result)
        print(json.dumps({'n': len(result['games']),
            'old': sum(x['old_source_reward'] for x in result['games']),
            'new': sum(x['new']['reward'] for x in result['games']),
            'failures': len(result['failures'])}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--residual', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_centered_v3_20261007.pt'))
    parser.add_argument('--training-report', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_centered_v3_20261007.json'))
    parser.add_argument('--source-report-seq0', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq0_6_20261007.json'))
    parser.add_argument('--source-report-seq6', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq6_6_20261007.json'))
    parser.add_argument('--plan', type=Path, default=Path('ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json'))
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--target-policy', choices=('independent12',), default='independent12')
    parser.add_argument('--checkpoint-training-candidates', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json'))
    parser.add_argument('--warmstart-candidates', type=Path, default=Path('results/trajectory_hyperlora/alf_next_task_candidates_20261005.json'))
    parser.add_argument('--prior-train-review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_holdout_8x12_v3c_reviewed_20261007.json'))
    parser.add_argument('--matrix', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_holdout_8x12_v3c_merged_20261007.json'))
    parser.add_argument('--audit', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_holdout_8x12_v3c_merged_audited_20261007.json'))
    parser.add_argument('--features', type=Path, default=Path('results/trajectory_hyperlora/alf_future_reward_holdout8x12_v3c_features_20261007.pt'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_dev12_centered_v3_20261007.json'))
    parser.add_argument('--source-id', type=int, default=3)
    parser.add_argument('--residual-scale', type=float, default=1.)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.65)
    evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
