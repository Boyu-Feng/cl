"""Separate common bias from source- and target-dependent latent LoRA signal."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import torch
from torch.nn import functional as F

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.train_alf_feedback_residual_v1 import FeedbackResidual


def analyze(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    features = torch.load(args.features, map_location='cpu', weights_only=True)
    if (features['target_policy'] != 'train18' or
            features['global'].shape[0] != 8 or
            features['targets'].shape[0] != 18):
        raise ValueError('Expected frozen full-cross training features')
    delta = torch.stack([features['deltas'][i, :int(count)].mean(0)
        for i, count in enumerate(features['event_counts'])])
    trained = {}
    rows = []
    for seed, model_path, audit_path in zip(args.seeds, args.models,
                                            args.audits, strict=True):
        model = torch.load(model_path, map_location='cpu', weights_only=True)
        audit = json.loads(audit_path.read_text())
        if (audit['model_sha256'] != file_hash(model_path) or
                model['config']['features_sha256'] != file_hash(args.features) or
                model['config']['seed'] != seed or
                audit['summary']['rollouts'] != 72):
            raise ValueError('Changed or unaudited reward-trained residual')
        correction = FeedbackResidual(features['global'].shape[-1],
            model['config']['code_dim'])
        correction.load_state_dict(model['correction'])
        correction.eval()
        with torch.no_grad():
            inputs = torch.stack([torch.stack([torch.cat([
                F.layer_norm(item.float(), item.shape[-1:])
                for item in (features['global'][si], delta[si],
                             features['targets'][ti])])
                for si in range(8)]) for ti in range(18)])
            hidden = correction.net[:-1](inputs)
            head = correction.net[-1]
            feature_shift = (F.linear(hidden, head.weight) @
                             model['basis'].T).double()
            bias_shift = (head.bias @ model['basis'].T).double()
            shifts = feature_shift + bias_shift
            direct = torch.stack([torch.stack([
                correction(features['global'][si], delta[si],
                    features['targets'][ti]) @ model['basis'].T
                for si in range(8)]) for ti in range(18)]).double()
            if not torch.allclose(shifts, direct, atol=1e-6, rtol=1e-6):
                raise ValueError('Feature/bias decomposition does not reconstruct')
        trained[seed] = shifts
        grand = shifts.mean((0, 1))
        source_effect = shifts.mean(0) - grand
        target_effect = shifts.mean(1) - grand
        interaction = shifts - grand - source_effect.unsqueeze(0) - \
            target_effect.unsqueeze(1)
        total = (shifts - grand).square().sum()
        components = {'source': 18 * source_effect.square().sum(),
            'target': 8 * target_effect.square().sum(),
            'interaction': interaction.square().sum()}
        unit = F.normalize(shifts, dim=-1)
        upper = torch.triu_indices(8, 8, offset=1)
        cos = torch.matmul(unit, unit.transpose(-1, -2))
        rows.append({'seed': seed,
            'mean_latent_shift_norm': float(shifts.norm(dim=-1).mean()),
            'common_bias_norm': float(bias_shift.norm()),
            'mean_feature_dependent_norm': float(
                feature_shift.norm(dim=-1).mean()),
            'relative_source_spread': float(
                (shifts - shifts.mean(1, keepdim=True)).norm(dim=-1).mean() /
                shifts.norm(dim=-1).mean()),
            'relative_target_spread': float(
                (shifts - shifts.mean(0, keepdim=True)).norm(dim=-1).mean() /
                shifts.norm(dim=-1).mean()),
            'mean_fixed_target_source_cosine': float(
                cos[:, upper[0], upper[1]].mean()),
            'centered_variance_fraction': {key: float(value / total)
                for key, value in components.items()},
            'component_reconstruction_relative_error': float(
                abs(sum(components.values()) - total) / total),
            'model_sha256': file_hash(model_path),
            'audit_sha256': file_hash(audit_path)})
    comparisons = []
    for left, right in combinations(args.seeds, 2):
        a, b = trained[left].reshape(-1, 128), trained[right].reshape(-1, 128)
        comparisons.append({'seeds': [left, right],
            'mean_pairwise_cosine': float(F.cosine_similarity(a, b,
                dim=-1).mean()),
            'flattened_cosine': float(F.cosine_similarity(a.flatten()[None],
                b.flatten()[None]).item())})
    result = {'protocol': 'Post hoc exact feature-dependent versus final-head-bias decomposition of 72-reward-rollout frozen residuals over eight reviewed own-success sources and 18 train-domain target initial observations; source/target spread normalized by full latent shift, not by centered variance alone',
        'features_sha256': file_hash(args.features),
        'models': rows, 'cross_seed': comparisons}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'models': rows, 'cross_seed': comparisons}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=Path, default=Path('results/trajectory_hyperlora/alf_future_reward_train8x18_features_20261007.pt'))
    parser.add_argument('--seeds', nargs=3, type=int, default=[42, 43, 44])
    parser.add_argument('--models', nargs=3, type=Path, default=[Path(f'results/trajectory_hyperlora/alf_feedback_residual_train18_balanced_seed{seed}_72_20261007.pt') for seed in (42, 43, 44)])
    parser.add_argument('--audits', nargs=3, type=Path, default=[Path(f'results/trajectory_hyperlora/alf_feedback_residual_train18_balanced_seed{seed}_72_audited_20261007.json') for seed in (42, 43, 44)])
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_geometry_bias_decomposition_train18_balanced_20261007.json'))
    analyze(parser.parse_args())


if __name__ == '__main__':
    main()
