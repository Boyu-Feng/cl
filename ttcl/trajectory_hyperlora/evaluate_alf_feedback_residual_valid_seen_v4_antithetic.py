"""Frozen official valid_seen evaluation of antithetic reward-trained LoRA."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alf_feedback_residual_valid_seen_v1 import checked
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import bounded_source_text
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v2 import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields, task_context_text
from ttcl.trajectory_hyperlora.train_alf_feedback_residual_v1 import (
    ResidualInjection,
)
from ttcl.trajectory_hyperlora.centered_feedback_residual_v3 import (
    CenteredEvidenceResidual,
)


def episode(agent, tokenizer, game, args, source_fields=None, memory=None,
            shift=None):
    if shift is None:
        return run_episode(agent, tokenizer, game,
            source_fields if source_fields is not None else {},
            adapter=source_fields is not None, device=args.device,
            max_steps=50, max_new_tokens=64, constrain_actions=True,
            actor_history_turns=2, loop_guard_max=2,
            memory_text=memory)
    with ResidualInjection(agent, shift):
        return run_episode(agent, tokenizer, game, source_fields,
            adapter=True, device=args.device, max_steps=50,
            max_new_tokens=64, constrain_actions=True,
            actor_history_turns=2, loop_guard_max=2)


def verify_episode(result, initial=None):
    if (result['status'] != 'complete' or result['steps'] > 50 or
            result['steps'] != len(result['trajectory']) or
            result['invalid_commands'] != 0 or
            result['reward'] != float(result['termination'] == 'success') or
            (initial is not None and
             result['initial_observation'] != initial)):
        raise ValueError('Failed or changed official ALFWorld episode')


def controls(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    targets, source = checked(args)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    bounded, original, retained = bounded_source_text(tokenizer,
        source['records'], 2048)
    with torch.no_grad():
        fields = contextual_text_fields(agent, tokenizer, bounded,
            args.device, 2048, pooling='both')
    report = {'protocol': 'Frozen official valid_seen twelve-target paired no-LoRA, old same-own-success LoRA, and identical-source raw text; one attempt per target, 50-step/64-token/2-history/loopguard2 actor',
        'review_sha256': file_hash(args.review),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'source_records_sha256': source['records_sha256'],
        'source_tokens_original': original,
        'source_tokens_retained': retained,
        'max_steps': 50, 'max_new_tokens': 64,
        'actor_history_turns': 2, 'loop_guard_max': 2,
        'games': [], 'failures': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for target in targets:
        game = args.data_root / target['game']
        base = episode(agent, tokenizer, game, args)
        verify_episode(base)
        old = episode(agent, tokenizer, game, args, source_fields=fields)
        text = episode(agent, tokenizer, game, args, memory=bounded)
        verify_episode(old, base['initial_observation'])
        verify_episode(text, base['initial_observation'])
        report['games'].append({'game': target['game'],
            'input_content_sha256': target['input_content_sha256'],
            'base': base, 'old_lora': old, 'raw_text': text})
        save(args.output, report)
        print(json.dumps({'n': len(report['games']),
            'base': sum(x['base']['reward'] for x in report['games']),
            'old': sum(x['old_lora']['reward'] for x in report['games']),
            'text': sum(x['raw_text']['reward'] for x in report['games']),
            'failures': len(report['failures'])}), flush=True)
    return report


def residual(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if not math.isfinite(args.residual_scale) or args.residual_scale == 0:
        raise ValueError('Need finite nonzero residual scale')
    targets, source = checked(args)
    controls_report = json.loads(args.controls.read_text())
    model = torch.load(args.residual_model, map_location='cpu',
                       weights_only=True)
    training = json.loads(args.training_report.read_text())
    encoded = torch.load(args.training_features, map_location='cpu',
                         weights_only=True)
    if (controls_report['review_sha256'] != file_hash(args.review) or
            controls_report['checkpoint_sha256'] != file_hash(args.checkpoint) or
            controls_report['failures'] or
            len(controls_report['games']) != len(targets) or
            training['trained_model_sha256'] != file_hash(args.residual_model) or
            training['checkpoint_sha256'] != file_hash(args.checkpoint) or
            model['config']['checkpoint_sha256'] != file_hash(args.checkpoint) or
            model['config']['features_sha256'] !=
                file_hash(args.training_features) or
            encoded['target_policy'] != 'train18' or
            encoded['review_sha256'] != training['review_sha256']):
        raise ValueError('Changed baseline or unfrozen reward model')
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    bounded, _, _ = bounded_source_text(tokenizer, source['records'], 2048)
    with torch.no_grad():
        fields = contextual_text_fields(agent, tokenizer, bounded,
            args.device, 2048, pooling='both')
    if model.get('model_kind') != 'centered_bilinear_antithetic':
        raise ValueError('Expected frozen centered evidence model')
    correction = CenteredEvidenceResidual.from_state(
        model['correction']).to(args.device)
    correction.load_state_dict(model['correction'])
    correction.eval()
    basis = model['basis'].to(args.device)
    si = args.source_id
    source_vector = encoded['global'][si].to(args.device)
    count = int(encoded['event_counts'][si])
    delta = encoded['deltas'][si, :count].mean(0).to(args.device)
    report = {'protocol': 'Frozen antithetic reward-trained feedback-conditioned LoRA residual on official valid_seen; source and target identical to paired controls; no valid_seen result used during training',
        'review_sha256': file_hash(args.review),
        'controls_sha256': file_hash(args.controls),
        'training_report_sha256': file_hash(args.training_report),
        'residual_sha256': file_hash(args.residual_model),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'source_id': si,
        'residual_scale': args.residual_scale,
        'max_steps': 50, 'max_new_tokens': 64,
        'actor_history_turns': 2, 'loop_guard_max': 2,
        'games': [], 'failures': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for target, row in zip(targets, controls_report['games'], strict=True):
        if (row['game'] != target['game'] or
                row['input_content_sha256'] !=
                target['input_content_sha256']):
            raise ValueError('Changed paired target order')
        initial = row['base']['initial_observation']
        with torch.no_grad():
            query = contextual_text_fields(agent, tokenizer,
                task_context_text(initial, initial), args.device,
                2048, pooling='both')['pair_contextual'].float().flatten()
            mean = correction(source_vector, delta, query)
            shift = args.residual_scale * (mean @ basis.T)
        outcome = episode(agent, tokenizer, args.data_root / target['game'],
                          args, source_fields=fields, shift=shift)
        verify_episode(outcome, initial)
        report['games'].append({'game': target['game'],
            'input_content_sha256': target['input_content_sha256'],
            'old_lora_reward': row['old_lora']['reward'],
            'base_reward': row['base']['reward'],
            'raw_text_reward': row['raw_text']['reward'],
            'mean_code_norm': float(mean.norm()), 'new': outcome})
        save(args.output, report)
        print(json.dumps({'n': len(report['games']),
            'base': sum(x['base_reward'] for x in report['games']),
            'old': sum(x['old_lora_reward'] for x in report['games']),
            'text': sum(x['raw_text_reward'] for x in report['games']),
            'new': sum(x['new']['reward'] for x in report['games']),
            'failures': len(report['failures'])}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('controls', 'residual'))
    parser.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
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
    parser.add_argument('--residual-model', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_antithetic_v4_seed42_20261007.pt'))
    parser.add_argument('--training-report', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_antithetic_v4_seed42_20261007.json'))
    parser.add_argument('--training-features', type=Path, default=Path('results/trajectory_hyperlora/alf_future_reward_train8x18_features_20261007.pt'))
    parser.add_argument('--residual-scale', type=float, default=1.)
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_valid_seen12_antithetic_v4_seed42_20261007.json'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.65)
    args = parser.parse_args()
    (controls if args.command == 'controls' else residual)(args)


if __name__ == '__main__':
    main()
