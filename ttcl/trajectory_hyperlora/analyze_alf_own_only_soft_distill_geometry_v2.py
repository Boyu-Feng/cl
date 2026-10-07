"""Measure effective LoRA matrix geometry after own-only soft distillation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.centered_feedback_residual_v3 import (
    CenteredEvidenceResidual,
)


def geometry(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    model = torch.load(args.residual, map_location='cpu', weights_only=True)
    features = torch.load(args.features, map_location='cpu', weights_only=True)
    training = json.loads(args.training_report.read_text())
    audit = json.loads(args.training_audit.read_text())
    if (model.get('model_kind') != 'own_only_soft_distill_v2' or
            model['config']['features_sha256'] != file_hash(args.features) or
            training['trained_model_sha256'] != file_hash(args.residual) or
            audit['model_sha256'] != file_hash(args.residual) or
            audit['report_sha256'] != file_hash(args.training_report) or
            features['global'].shape[0] != 57 or
            len(features['targets']) != 126):
        raise ValueError('Changed or unaudited centered training model')
    correction = CenteredEvidenceResidual.from_state(model['correction'])
    correction.load_state_dict(model['correction'])
    correction.eval()
    delta = torch.stack([features['deltas'][i, :int(n)].mean(0)
                         for i, n in enumerate(features['event_counts'])])
    target_vectors = torch.stack([features['targets'][key]
        for key in sorted(features['targets'])])
    with torch.no_grad():
        code = torch.stack([torch.stack([
            correction(features['global'][si], delta[si],
                       target_vectors[ti])
            for si in range(57)]) for ti in range(126)])
    basis = model['basis'].float()
    latent = code @ basis.T
    agent, _ = load_agent(args.model, args.checkpoint,
                          args.device, args.gpu_fraction)
    reduced = []
    for adapter, head in zip(agent.adapters, agent.b_heads, strict=True):
        a = adapter.a.detach().float()
        # A.T = Q R, hence A = R.T Q.T and ||B A||_F = ||B R.T||_F.
        r_transpose = torch.linalg.qr(a.T, mode='reduced').R.T
        b = F.linear(latent.to(head.weight.device),
                     head.weight.detach().float()).reshape(
                         126, 57, adapter.base.out_features, adapter.rank)
        effective = (b @ r_transpose.to(b.device) / adapter.rank).cpu()
        reduced.append(effective)
    effective_flat = torch.cat([x.reshape(126, 57, -1)
                                for x in reduced], dim=-1)
    norm = effective_flat.norm(dim=-1)
    centered = effective_flat - effective_flat.mean(1, keepdim=True)
    unit = F.normalize(effective_flat, dim=-1)
    cos = unit @ unit.transpose(-1, -2)
    upper = torch.triu_indices(57, 57, offset=1)
    summary = {'targets': 126, 'sources': 57,
        'distillation_steps': audit['summary']['steps'],
        'model_parameter_count': sum(p.numel() for p in correction.parameters()),
        'mean_code_norm': float(code.norm(dim=-1).mean()),
        'maximum_fixed_target_mean_code_norm': float(code.mean(1).norm(dim=-1).max()),
        'mean_effective_delta_w_frobenius': float(norm.mean()),
        'relative_effective_source_spread': float(
            centered.norm(dim=-1).mean() / norm.mean().clamp_min(1e-12)),
        'mean_fixed_target_effective_source_cosine': float(
            cos[:, upper[0], upper[1]].mean()),
        'layer_mean_effective_delta_w_frobenius': [
            float(x.norm(dim=(-2, -1)).mean()) for x in reduced],
        'layer_max_effective_delta_w_rank': [
            int((torch.linalg.svdvals(x.reshape(-1, *x.shape[-2:])) > 1e-6)
                .sum(-1).max()) for x in reduced]}
    result = {'protocol': 'Exact effective delta-W geometry, not reward or causality, on content-bound 57 reviewed train sources by 126 own-only successful train action states; frozen residual and actor',
        'residual_sha256': file_hash(args.residual),
        'training_audit_sha256': file_hash(args.training_audit),
        'features_sha256': file_hash(args.features),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'summary': summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--residual', type=Path, default=Path('results/trajectory_hyperlora/alf_own_only_soft_distill_v2_seed42_64_retry_20261007.pt'))
    parser.add_argument('--training-report', type=Path, default=Path('results/trajectory_hyperlora/alf_own_only_soft_distill_v2_seed42_64_retry_20261007.json'))
    parser.add_argument('--training-audit', type=Path, default=Path('results/trajectory_hyperlora/alf_own_only_soft_distill_v2_seed42_64_retry_audited_20261007.json'))
    parser.add_argument('--features', type=Path, default=Path('results/trajectory_hyperlora/alf_own_only_future_features_v2_20261007.pt'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_own_only_soft_distill_v2_geometry_seed42_20261007.json'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.65)
    geometry(parser.parse_args())
