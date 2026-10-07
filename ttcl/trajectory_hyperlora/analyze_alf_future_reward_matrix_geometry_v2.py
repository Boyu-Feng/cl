"""Measure effective delta-W geometry, after the frozen LoRA A projection.

This is a post hoc mechanism diagnostic on the reviewed 8x18 training matrix.
It neither chooses a source nor evaluates on the independent holdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v2 import checked
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import bounded_source_text
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields, task_context_text
from ttcl.trajectory_hyperlora.train_alf_future_utility_selector_v1 import target_reward_matrix


def summarize(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    sources, targets, _ = checked(args)
    report = json.loads(args.matrix.read_text())
    audit = json.loads(args.audit.read_text())
    if (audit['target_policy'] != 'train18' or
            audit['review_sha256'] != file_hash(args.review) or
            audit['raw_report_sha256'] != file_hash(args.matrix) or
            audit['checkpoint_sha256'] != file_hash(args.checkpoint) or
            report['failures'] or len(sources) != 8 or len(targets) != 18):
        raise ValueError('Need audited training-only cross reward matrix')
    rewards = target_reward_matrix(report, 8, 18)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != 'contextual' or
            agent.task_pair_pooling != 'mean'):
        raise ValueError('Changed contextual source model')
    source_fields = []
    # If A^T=QR, then ||BA||_F=||BR^T||_F. The huge base matrix
    # need never be materialized for exact Frobenius geometry.
    projections = [torch.linalg.qr(layer.a.detach().float().T,
        mode='reduced').R.T.cpu() / layer.rank for layer in agent.adapters]
    with torch.no_grad():
        for source in sources:
            bounded, _, _ = bounded_source_text(tokenizer, source['records'], 2048)
            source_fields.append(contextual_text_fields(agent, tokenizer,
                bounded, args.device, 2048, pooling='both'))
        factors = []
        for target, base in zip(targets, report['base'], strict=True):
            if (base['target_id'] != target['target_id'] or
                    base['episode']['status'] != 'complete'):
                raise ValueError('Changed target reset')
            observation = base['episode']['initial_observation']
            query = contextual_text_fields(agent, tokenizer,
                task_context_text(observation, observation), args.device,
                2048, pooling='both')
            task_factors = []
            for source in source_fields:
                agent.set_source(source, target_fields=query)
                task_factors.append(torch.cat([(layer.b.detach().float()
                    .squeeze(0).cpu() @ projection).flatten()
                    for layer, projection in zip(agent.adapters,
                                                  projections, strict=True)]))
            agent.set_source(None)
            factors.append(torch.stack(task_factors))
    matrices = torch.stack(factors).double()
    rows = []
    for target_id, matrix in enumerate(matrices):
        centered = matrix - matrix.mean(0)
        singular = torch.linalg.svdvals(centered)
        energies = singular.square()
        separation = float((centered.norm(dim=-1).mean() /
            matrix.norm(dim=-1).mean()).item())
        outcomes = rewards[target_id].bool()
        between = None
        if outcomes.any() and (~outcomes).any():
            between = float((matrix[outcomes].mean(0) -
                matrix[~outcomes].mean(0)).norm().item() /
                matrix.norm(dim=-1).mean().item())
        cosines = F.normalize(matrix, dim=-1) @ F.normalize(matrix, dim=-1).T
        upper = torch.triu_indices(8, 8, offset=1)
        rows.append({'target_id': target_id,
            'base_reward': float(report['base'][target_id]['episode']['reward']),
            'source_successes': int(outcomes.sum()),
            'source_relative_factor_spread': separation,
            'winner_loser_mean_factor_separation': between,
            'first_source_pc_energy': float(energies[0] /
                energies.sum().clamp_min(1e-12)),
            'mean_pairwise_factor_cosine': float(cosines[upper[0],upper[1]].mean())})
    variable = [row for row in rows if row['winner_loser_mean_factor_separation']
                is not None]
    summary = {'targets': len(rows), 'source_dependent_targets': len(variable),
        'mean_source_relative_factor_spread': sum(
            x['source_relative_factor_spread'] for x in rows) / len(rows),
        'mean_first_source_pc_energy': sum(x['first_source_pc_energy']
            for x in rows) / len(rows),
        'mean_pairwise_factor_cosine': sum(x['mean_pairwise_factor_cosine']
            for x in rows) / len(rows),
        'variable_target_winner_loser_separation': sum(
            x['winner_loser_mean_factor_separation'] for x in variable) /
            len(variable)}
    result = {'protocol': 'Post hoc fixed-initial-task exact Frobenius geometry of effective delta-W=BA/rank, via thin QR of frozen A; eight reviewed self-success sources; outcomes from audited train18 matrix only',
        'review_sha256': file_hash(args.review),
        'reward_audit_sha256': file_hash(args.audit),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'summary': summary, 'targets': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


def main():
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
    parser.add_argument('--audit', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_audited_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_future_reward_effective_matrix_geometry_train8x18_20261007.json'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.65)
    summarize(parser.parse_args())


if __name__ == '__main__':
    main()
