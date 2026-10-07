"""Reconstruct exact effective LoRA matrices for audited update contrasts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_online_context_vector_mean_v1 import update_fields, vector_hash
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v2 import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.centered_feedback_residual_v3 import CenteredEvidenceResidual
from ttcl.trajectory_hyperlora.collect_alf_incremental_update_v9 import checked
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields, task_context_text
from ttcl.trajectory_hyperlora.evaluate_alf_fresh_seen_online_v6 import encode_source
from ttcl.trajectory_hyperlora.train_alf_feedback_residual_v4_antithetic import ResidualInjection


def flatten(parts):
    return torch.cat([part.detach().float().cpu().reshape(-1)
                      for part in parts])


def matrix(agent, source_fields, target_fields, shift, projections):
    with ResidualInjection(agent, shift):
        with torch.no_grad():
            agent.set_source(source_fields, target_fields=target_fields)
            reduced = [layer.b.detach().float().cpu().squeeze(0) @ right /
                       layer.rank
                       for layer, right in zip(agent.adapters, projections,
                                                strict=True)]
    return flatten(reduced)


def analyze(args):
    if args.matrix_output.exists():
        raise FileExistsError(args.matrix_output)
    sources, _, state, reviewed = checked(args)
    report = json.loads(args.output.read_text())
    audit = json.loads(args.audit_output.read_text())
    if (audit['raw_report_sha256'] != file_hash(args.output) or
            audit['summary']['pairs'] != 108 or
            audit['summary']['failed_replays'] != 0 or
            report['failures'] or len(report['rows']) != 108):
        raise ValueError('Changed or unaudited future-reward matrix')
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != 'contextual' or
            agent.task_context_scope != 'current' or
            agent.task_pair_pooling != 'mean'):
        raise ValueError('Changed frozen actor')
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    residual = CenteredEvidenceResidual.from_state(
        state['correction']).to(args.device)
    residual.load_state_dict(state['correction'])
    residual.eval()
    for parameter in residual.parameters():
        parameter.requires_grad_(False)
    basis = state['basis'].to(args.device).float()
    projections = []
    for layer in agent.adapters:
        a = layer.a.detach().float().cpu()
        _, right = torch.linalg.qr(a.T, mode='reduced')
        projections.append(right.T)
    source_cache = {}
    target_cache = {}
    rows = []
    for binding, observation in zip(reviewed, report['rows'], strict=True):
        if binding['input_content_sha256'] != observation['input_content_sha256']:
            raise ValueError('Changed reviewed result order')
        first_id, next_id = (observation['first_source_id'],
                             observation['next_source_id'])
        for source_id in (first_id, next_id):
            if source_id not in source_cache:
                source_cache[source_id] = encode_source(agent, tokenizer,
                    sources[source_id]['records'], args.device)
        first_fields, first_global, first_delta, _, _ = source_cache[first_id]
        next_fields, next_global, next_delta, _, _ = source_cache[next_id]
        updated_fields = update_fields(first_fields, next_fields, 1)
        if (vector_hash(first_fields) != observation['first_vector_sha256'] or
                vector_hash(updated_fields) !=
                observation['updated_vector_sha256']):
            raise ValueError('Actual source encoding differs from rollout')
        target_id = observation['target_id']
        initial = observation['freeze']['initial_observation']
        if target_id not in target_cache:
            with torch.no_grad():
                target_cache[target_id] = (initial,
                    contextual_text_fields(agent, tokenizer,
                        task_context_text(initial, initial), args.device,
                        2048, pooling='both'))
        cached_initial, target_fields = target_cache[target_id]
        if cached_initial != initial or observation['update'][
                'initial_observation'] != initial:
            raise ValueError('Same future target changed initial state')
        q = target_fields['pair_contextual'].float().flatten()
        old_mean = residual(first_global, first_delta, q)
        new_mean = residual((first_global + next_global)/2,
                            (first_delta + next_delta)/2, q)
        if (abs(old_mean.norm().item() -
                observation['freeze_code_norm']) > 1e-4 or
                abs(new_mean.norm().item() -
                    observation['update_code_norm']) > 1e-4):
            raise ValueError('Reconstructed residual code changed')
        old = matrix(agent, first_fields, target_fields,
                     old_mean @ basis.T, projections)
        new = matrix(agent, updated_fields, target_fields,
                     new_mean @ basis.T, projections)
        change = new-old
        relative = (change.norm() / old.norm().clamp_min(1e-8)).item()
        cosine = F.cosine_similarity(old[None], new[None]).item()
        effect = (observation['update']['reward'] -
                  observation['freeze']['reward'])
        row = {'transition_index': observation['transition_index'],
            'target_id': target_id, 'reward_difference': effect,
            'changed_trajectory': (observation['freeze']['trajectory'] !=
                                   observation['update']['trajectory']),
            'effective_delta_w_change_norm': change.norm().item(),
            'effective_delta_w_relative_change': relative,
            'effective_delta_w_cosine': cosine,
            'freeze_effective_delta_w_norm': old.norm().item(),
            'update_effective_delta_w_norm': new.norm().item()}
        if not all(math.isfinite(value) for value in row.values()
                   if isinstance(value, float)):
            raise ValueError('Nonfinite exact LoRA matrix statistic')
        rows.append(row)
        print(json.dumps({'done': len(rows),
            'reward_difference': effect,
            'relative_matrix_change': relative}), flush=True)
    by_outcome = {}
    for label, sign in (('write_helps', 1), ('write_hurts', -1),
                        ('same_reward', 0)):
        group = [row for row in rows if row['reward_difference'] == sign]
        if not group:
            by_outcome[label] = {'n': 0}
            continue
        by_outcome[label] = {'n': len(group),
            'changed_trajectories': sum(x['changed_trajectory'] for x in group),
            'mean_relative_matrix_change': sum(
                x['effective_delta_w_relative_change'] for x in group)/len(group),
            'mean_matrix_cosine': sum(
                x['effective_delta_w_cosine'] for x in group)/len(group)}
    value = {'protocol': 'Read-only exact B*A/r effective matrix reconstruction of the audited train-domain freeze/update contrasts; 8 own-success source encodings and 18 future target contexts, no environment interaction or fitting; outcome-stratified quantities are descriptive, not causal proof from matrix magnitude',
        'report_sha256': file_hash(args.output),
        'audit_sha256': file_hash(args.audit_output),
        'review_sha256': file_hash(args.update_review),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'residual_sha256': file_hash(args.residual),
        'rows': rows, 'by_outcome': by_outcome}
    save(args.matrix_output, value)
    print(json.dumps(by_outcome), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path(
        'current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--source-report-seq0', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq0_6_20261007.json'))
    parser.add_argument('--source-report-seq6', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq6_6_20261007.json'))
    parser.add_argument('--plan', type=Path, default=Path(
        'ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json'))
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--target-policy', choices=('train18',), default='train18')
    parser.add_argument('--checkpoint-training-candidates', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json'))
    parser.add_argument('--warmstart-candidates', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_next_task_candidates_20261005.json'))
    parser.add_argument('--prior-train-review', type=Path, default=Path(
        'data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--review', type=Path, default=Path(
        'data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--residual', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_20261007.pt'))
    parser.add_argument('--training-report', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_20261007.json'))
    parser.add_argument('--training-audit', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_audited_20261007.json'))
    parser.add_argument('--update-review', type=Path, default=Path(
        'data/annotations/alf_incremental_update108_v9_reviewed_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_incremental_update108_v9_20261007.json'))
    parser.add_argument('--audit-output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_incremental_update108_v9_audited_20261007.json'))
    parser.add_argument('--matrix-output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_incremental_update108_v9_effective_matrix_20261007.json'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.7)
    analyze(parser.parse_args())
