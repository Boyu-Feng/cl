"""Compare frozen reward-trained LoRA with old LoRA and raw source text.

The twelve official train-domain development games and their plan sequences
are disjoint from reward-training targets. This set was exposed in prior
research, so it is a development comparison rather than a blind test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import bounded_source_text
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v2 import checked, save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.centered_feedback_residual_v3 import CenteredEvidenceResidual
from ttcl.trajectory_hyperlora.train_alf_feedback_residual_v4_antithetic import (
    ResidualInjection, source_fields,
)


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    sources, targets, bindings = checked(args)
    review = json.loads(args.review.read_text())
    matrix = json.loads(args.matrix.read_text())
    audit = json.loads(args.audit.read_text())
    features = torch.load(args.features, map_location='cpu',
                          weights_only=True)
    model = torch.load(args.residual, map_location='cpu',
                       weights_only=True)
    training = json.loads(args.training_report.read_text())
    training_audit = json.loads(args.training_audit.read_text())
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
            matrix['failures'] or len(matrix['base']) != 12 or
            len(matrix['pairs']) != 96 or len(sources) != 8 or
            len(targets) != 12 or len(bindings) != 96 or
            features['target_policy'] != 'independent12' or
            features['review_sha256'] != file_hash(args.review) or
            features['checkpoint_sha256'] != file_hash(args.checkpoint) or
            digest([row['episode']['initial_observation']
                    for row in matrix['base']]) !=
                    features['base_observations_sha256'] or
            training_audit['raw_report_sha256'] !=
                file_hash(args.training_report) or
            training_audit['trained_model_sha256'] !=
                file_hash(args.residual) or
            training_audit['summary']['pairs'] != 72 or
            training_audit['summary']['failed_replays'] != 0 or
            training['trained_model_sha256'] != file_hash(args.residual) or
            training['checkpoint_sha256'] != file_hash(args.checkpoint) or
            training['max_pairs'] != 72 or
            model['model_kind'] !=
                'centered_bilinear_action_sensitive_v5' or
            model['config']['checkpoint_sha256'] !=
                file_hash(args.checkpoint) or
            model['config']['features_sha256'] !=
                training['features_sha256'] or
            not 0 <= args.source_id < 8):
        raise ValueError('Changed trained model or non-disjoint development inputs')
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    source_bank = source_fields(agent, tokenizer, sources, args)
    bounded, original, retained = bounded_source_text(tokenizer,
        sources[args.source_id]['records'], 2048)
    if retained > 2048:
        raise ValueError('Changed raw source text budget')
    correction = CenteredEvidenceResidual.from_state(
        model['correction']).to(args.device)
    correction.load_state_dict(model['correction'])
    correction.eval()
    basis = model['basis'].to(args.device).float()
    deltas = torch.stack([features['deltas'][i, :int(n)].mean(0)
        for i, n in enumerate(features['event_counts'])])
    report = {'protocol': 'Frozen action-sensitive reward-trained LoRA versus same-own-source old LoRA, no-LoRA and raw source text on previously exposed but train-game/plan-sequence-disjoint official ALFWorld development tasks; identical one-attempt actor step budgets, text inference context cost not matched',
        'review_sha256': file_hash(args.review),
        'matrix_sha256': file_hash(args.matrix),
        'audit_sha256': file_hash(args.audit),
        'features_sha256': file_hash(args.features),
        'training_report_sha256': file_hash(args.training_report),
        'training_audit_sha256': file_hash(args.training_audit),
        'residual_sha256': file_hash(args.residual),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'source_id': args.source_id,
        'source_records_sha256': sources[args.source_id]['records_sha256'],
        'source_tokens_original': original,
        'source_tokens_retained': retained,
        'max_steps': 50, 'max_new_tokens': 64,
        'actor_history_turns': 2, 'loop_guard_max': 2,
        'games': [], 'failures': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for target in targets:
        ti = target['target_id']
        si = args.source_id
        binding = bindings[si*12+ti]
        old_arm = matrix['pairs'][si*12+ti]
        if (old_arm['input_content_sha256'] !=
                binding['input_content_sha256'] or
                old_arm['episode']['status'] != 'complete'):
            raise ValueError('Changed old same-source LoRA arm')
        initial = matrix['base'][ti]['episode']['initial_observation']
        with torch.no_grad():
            mean = correction(features['global'][si].to(args.device),
                deltas[si].to(args.device),
                features['targets'][ti].to(args.device))
            shift = mean @ basis.T
            with ResidualInjection(agent, shift):
                new = run_episode(agent, tokenizer,
                    args.data_root / target['game'], source_bank[si],
                    adapter=True, device=args.device, max_steps=50,
                    max_new_tokens=64, constrain_actions=True,
                    actor_history_turns=2, loop_guard_max=2)
            raw_text = run_episode(agent, tokenizer,
                args.data_root / target['game'], {}, adapter=False,
                device=args.device, max_steps=50, max_new_tokens=64,
                constrain_actions=True, actor_history_turns=2,
                loop_guard_max=2, memory_text=bounded)
        if (new['status'] != 'complete' or
                raw_text['status'] != 'complete' or
                new['invalid_commands'] != 0 or
                raw_text['invalid_commands'] != 0 or
                new['initial_observation'] != initial or
                raw_text['initial_observation'] != initial):
            report['failures'].append({'target_id': ti,
                'new_status': new['status'],
                'text_status': raw_text['status']})
            save(args.output, report)
            raise RuntimeError('Changed or incomplete development episode')
        report['games'].append({'target_id': ti,
            'game': target['game'],
            'game_sha256': target['game_sha256'],
            'source_target_binding': binding['input_content_sha256'],
            'base_reward': matrix['base'][ti]['episode']['reward'],
            'old_lora_reward': old_arm['episode']['reward'],
            'mean_code_norm': float(mean.norm()),
            'new': new, 'raw_text': raw_text})
        save(args.output, report)
        print(json.dumps({'done': len(report['games']),
            'base': sum(x['base_reward'] for x in report['games']),
            'old': sum(x['old_lora_reward'] for x in report['games']),
            'new': sum(x['new']['reward'] for x in report['games']),
            'text': sum(x['raw_text']['reward'] for x in report['games']),
            'failures': 0}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--residual', type=Path, default=Path('results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_20261007.pt'))
    parser.add_argument('--training-report', type=Path, default=Path('results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_20261007.json'))
    parser.add_argument('--training-audit', type=Path, default=Path('results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_audited_20261007.json'))
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
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_action_sensitive_reward_dev12_v5_20261007.json'))
    parser.add_argument('--source-id', type=int, default=3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.7)
    evaluate(parser.parse_args())
