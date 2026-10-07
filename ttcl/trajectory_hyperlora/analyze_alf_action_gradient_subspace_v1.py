"""Measure whether the frozen LoRA code subspace can alter critical actions.

On reviewed, policy-conditional action-loss states, compute the gradient of
the successful-versus-failed action log-probability margin with respect to
the 128-dimensional residual code. Compare its projection into the old
trajectory-generator PCA basis. These are local, off-policy diagnostics, not
terminal reward or independent generalization measurements.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import bounded_source_text
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.centered_feedback_residual_v3 import CenteredEvidenceResidual
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields, task_context_text
from ttcl.trajectory_hyperlora.train_alf_self_future_soft_distill_v1 import action_logits, messages_for, prompt_ids
from ttcl.trajectory_hyperlora.probe_alf_own_only_first_divergence_v2 import save


def score(agent, tokenizer, prefix, action: str, device: str):
    answer = tokenizer(action + tokenizer.eos_token,
                       add_special_tokens=False).input_ids
    logits = action_logits(agent, prefix, answer, device)
    ids = torch.tensor(answer, device=device)[:, None]
    return F.log_softmax(logits, -1).gather(-1, ids).mean()


def analyze(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    causal = json.loads(args.causal_targets.read_text())
    causal_audit = json.loads(args.causal_audit.read_text())
    source_bank = json.loads(args.source_bank.read_text())
    target_bank = json.loads(args.target_bank.read_text())
    plan = json.loads(args.pairs.read_text())
    features = torch.load(args.features, map_location='cpu', weights_only=True)
    distilled = torch.load(args.distilled, map_location='cpu', weights_only=True)
    training = json.loads(args.training_report.read_text())
    if (causal_audit['targets_review_sha256'] !=
            file_hash(args.causal_targets) or
            causal_audit['summary']['reviewed_targets'] != 42 or
            features['source_bank_sha256'] != file_hash(args.source_bank) or
            features['target_bank_sha256'] != file_hash(args.target_bank) or
            features['pairs_review_sha256'] != file_hash(args.pairs) or
            distilled['model_kind'] != 'own_only_soft_distill_v2' or
            distilled['config']['features_sha256'] != file_hash(args.features) or
            training['trained_model_sha256'] != file_hash(args.distilled) or
            training['checkpoint_sha256'] != file_hash(args.checkpoint)):
        raise ValueError('Changed causal action or frozen generator lineage')
    selected = [row for row in causal['rows']
                if row['policy_advantage50'] == 1.]
    if not selected:
        raise ValueError('No 50-step policy-conditional action effects')
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    agent.model.config.use_cache = False
    correction = CenteredEvidenceResidual.from_state(
        distilled['correction']).to(args.device)
    correction.load_state_dict(distilled['correction'])
    correction.eval()
    basis = distilled['basis'].to(args.device).float()
    if basis.shape != (128, 8):
        raise ValueError('Changed prior code basis')
    sources = {row['game']: row for row in source_bank['rows']}
    targets = {row['game']: row for row in target_bank['rows']}
    source_index = {row['game']: i for i, row in enumerate(source_bank['rows'])}
    pair_source = {}
    for pair in plan['pairs']:
        pair_source.setdefault(pair['target_game'], pair['source_game'])
    delta = torch.stack([features['deltas'][i, :int(n)].mean(0)
        for i, n in enumerate(features['event_counts'])])
    fields = {}
    for source in sorted(set(pair_source.values())):
        with torch.no_grad():
            bounded, _, _ = bounded_source_text(tokenizer,
                sources[source]['records'], training['context_tokens'])
            fields[source] = contextual_text_fields(agent, tokenizer,
                bounded, args.device, training['context_tokens'], pooling='both')
    report = {'protocol': 'Train-only off-policy local action-margin sensitivity for verified 50-step policy-conditional branch losses; old latent-PCA projection and actual distilled-code alignment; no causal outcome claim',
        'causal_targets_sha256': file_hash(args.causal_targets),
        'causal_audit_sha256': file_hash(args.causal_audit),
        'distilled_sha256': file_hash(args.distilled),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'rows': [], 'failures': []}
    save(args.output, report)
    for target in selected[:args.max_cases]:
        source = pair_source[target['game']]
        game = targets[target['game']]
        turn = target['turn']
        reviewed = game['reviewed_targets'][turn]
        if (reviewed['observation'] != target['observation'] or
                reviewed['admissible_commands'] !=
                    target['admissible_commands'] or
                reviewed['target_action'] != target['positive_action']):
            raise ValueError('Changed critical decision state')
        messages = messages_for(game, turn)
        prefix = prompt_ids(tokenizer, messages, 4096)
        initial = game['reviewed_targets'][0]['observation']
        with torch.no_grad():
            q = contextual_text_fields(agent, tokenizer,
                task_context_text(initial, target['observation']),
                args.device, 2048, pooling='both')['pair_contextual'].float().flatten()
            query = {'contextual': q[None], 'pair_contextual': q[None]}
            agent.set_source(fields[source], target_fields=query)
        z = torch.zeros(128, device=args.device, requires_grad=True)
        old_b = [adapter.b for adapter in agent.adapters]
        try:
            for adapter, head, prior in zip(agent.adapters, agent.b_heads,
                                            old_b, strict=True):
                adapter.b = prior + F.linear(
                    z.to(head.weight.dtype), head.weight).reshape_as(prior)
            margin = (score(agent, tokenizer, prefix,
                            target['positive_action'], args.device) -
                      score(agent, tokenizer, prefix,
                            target['negative_action'], args.device))
            grad, = torch.autograd.grad(margin, z)
            with torch.no_grad():
                i = source_index[source]
                code = correction(features['global'][i].to(args.device),
                    delta[i].to(args.device), q)
                shift = code @ basis.T
                projected = basis @ (basis.T @ grad.float())
                grad_norm = grad.float().norm()
                ratio = projected.norm()/grad_norm.clamp_min(1e-12)
                alignment = F.cosine_similarity(grad.float(),
                    shift.float(), dim=0)
                linear_gain = (grad.float()*shift.float()).sum()
                for adapter, head, prior in zip(agent.adapters,
                                                agent.b_heads, old_b,
                                                strict=True):
                    adapter.b = prior + F.linear(
                        shift.to(head.weight.dtype),
                        head.weight).reshape_as(prior)
                actual_margin = (score(agent, tokenizer, prefix,
                    target['positive_action'], args.device) -
                    score(agent, tokenizer, prefix,
                    target['negative_action'], args.device))
                row = {'input_content_sha256': target['input_content_sha256'],
                    'game': target['game'], 'family': target['family'],
                    'source_game': source, 'turn': turn,
                    'margin_before_residual': float(margin.detach()),
                    'gradient_norm': float(grad_norm),
                    'old_basis_gradient_projection_ratio': float(ratio),
                    'distilled_code_norm': float(code.norm()),
                    'distilled_shift_norm': float(shift.norm()),
                    'distilled_shift_gradient_cosine': float(alignment),
                    'first_order_margin_gain': float(linear_gain),
                    'margin_after_residual': float(actual_margin),
                    'actual_margin_gain': float(actual_margin-margin.detach())}
                report['rows'].append(row)
                save(args.output, report)
                print(json.dumps({'n': len(report['rows']),
                    'ratio': row['old_basis_gradient_projection_ratio'],
                    'linear_gain': row['first_order_margin_gain']}),
                    flush=True)
        finally:
            agent.set_source(None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--causal-targets', type=Path, default=Path('data/annotations/alf_own_only_causal_action42_reviewed_20261007.json'))
    parser.add_argument('--causal-audit', type=Path, default=Path('results/trajectory_hyperlora/alf_own_only_causal_action42_audited_20261007.json'))
    parser.add_argument('--source-bank', type=Path, default=Path('data/annotations/alf_self_base57_replay_reviewed_20261007.json'))
    parser.add_argument('--target-bank', type=Path, default=Path('data/annotations/alf_own_only42_replay_reviewed_20261007.json'))
    parser.add_argument('--pairs', type=Path, default=Path('data/annotations/alf_own_only_future_pairs_reviewed_20261007.json'))
    parser.add_argument('--features', type=Path, default=Path('results/trajectory_hyperlora/alf_own_only_future_features_v2_20261007.pt'))
    parser.add_argument('--distilled', type=Path, default=Path('results/trajectory_hyperlora/alf_own_only_soft_distill_v2_seed42_64_retry_20261007.pt'))
    parser.add_argument('--training-report', type=Path, default=Path('results/trajectory_hyperlora/alf_own_only_soft_distill_v2_seed42_64_retry_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_own_only_action_gradient_subspace_v1_20261007.json'))
    parser.add_argument('--max-cases', type=int, default=14)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.7)
    args = parser.parse_args()
    if args.max_cases < 1:
        parser.error('Need at least one verified action effect')
    analyze(args)
