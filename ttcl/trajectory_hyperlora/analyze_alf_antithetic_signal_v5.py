"""Audit reward signal available to the frozen antithetic LoRA experiment.

This reconstructs each score-function gradient from reviewed rollouts and
frozen centered source/target features. Exact sign flips are a descriptive
coherence reference, not a hypothesis test for nonstationary RL updates.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.centered_feedback_residual_v3 import (
    CenteredEvidenceResidual, normalized_source, normalized_target,
)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator else float('nan')


def exact_sign_flip_tail(nonzero: np.ndarray) -> dict:
    if not 0 < len(nonzero) <= 20:
        raise ValueError('Exact diagnostic supports one to twenty nonzero pairs')
    observed = float(np.linalg.norm(nonzero.sum(0)))
    above = 0
    total = 0
    # Blocks keep both the latent and weight-space exact enumeration bounded.
    for block in itertools.batched(itertools.product((-1., 1.),
                                                      repeat=len(nonzero)), 4096):
        signs = np.asarray(block, dtype=np.float64)
        lengths = np.linalg.norm(signs @ nonzero, axis=1)
        above += int(np.count_nonzero(lengths >= observed-1e-10))
        total += len(lengths)
    return {'observed_sum_norm': observed, 'sign_flip_ge_count': above,
            'sign_flip_total': total, 'sign_flip_tail_fraction': above/total}


def analyze(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    run = json.loads(args.report.read_text())
    audit = json.loads(args.audit.read_text())
    review = json.loads(args.review.read_text())
    matrix = json.loads(args.matrix.read_text())
    saved = torch.load(args.model, map_location='cpu', weights_only=True)
    encoded = torch.load(args.features, map_location='cpu', weights_only=True)
    if (audit['report_sha256'] != file_hash(args.report) or
            audit['model_sha256'] != file_hash(args.model) or
            run['review_sha256'] != file_hash(args.review) or
            run['features_sha256'] != file_hash(args.features) or
            saved['config']['features_sha256'] != file_hash(args.features) or
            matrix['review_sha256'] != file_hash(args.review) or
            audit['summary']['pairs'] != 72 or
            audit['summary']['rollouts'] != 144 or
            len(run['train']) != 72 or run['failures'] or
            len(matrix['pairs']) != 144 or matrix['failures'] or
            saved['model_kind'] != 'centered_bilinear_antithetic' or
            len(review['pair_bindings']) != 144 or
            len(encoded['global']) != 8 or len(encoded['targets']) != 18 or
            run['sigma'] <= 0):
        raise ValueError('Changed antithetic reward lineage')
    prior = {(row['source_id'], row['target_id']): row
             for row in matrix['pairs']}
    binding = {(row['source_id'], row['target_id']): row
               for row in review['pair_bindings']}
    if len(prior) != 144 or len(binding) != 144:
        raise ValueError('Duplicate source-target cells or review bindings')
    variable = {target_id for target_id in range(18)
                if len({prior[source_id, target_id]['episode']['reward']
                        for source_id in range(8)}) > 1}
    correction = CenteredEvidenceResidual.from_state(saved['correction'])
    correction.load_state_dict(saved['correction'])
    mean_delta = torch.stack([encoded['deltas'][i, :int(count)].mean(0)
        for i, count in enumerate(encoded['event_counts'])])
    latent_gradients = []
    weight_gradients = []
    informative_variable = informative_constant = 0
    positive_only = negative_only = 0
    seen = set()
    for row, scheduled in zip(run['train'], run['pair_schedule'], strict=True):
        source_id, target_id = row['source_id'], row['target_id']
        key = source_id, target_id
        if (key in seen or key != (scheduled['source_id'],
                                   scheduled['target_id']) or
                row['input_content_sha256'] !=
                    binding[key]['input_content_sha256'] or
                row['baseline_reward'] != prior[key]['episode']['reward'] or
                row['positive_reward'] not in (0., 1.) or
                row['negative_reward'] not in (0., 1.)):
            raise ValueError('Changed scheduled source-target reward pair')
        seen.add(key)
        noise = np.asarray(row['noise'], dtype=np.float64)
        positive_code = np.asarray(row['positive_code'], dtype=np.float64)
        negative_code = np.asarray(row['negative_code'], dtype=np.float64)
        if (noise.shape != (run['code_dim'],) or
                not np.allclose(positive_code-negative_code,
                                2*run['sigma']*noise, rtol=1e-5, atol=1e-6)):
            raise ValueError('Changed antithetic latent factors')
        reward_delta = row['positive_reward']-row['negative_reward']
        positive_only += reward_delta > 0
        negative_only += reward_delta < 0
        if reward_delta:
            informative_variable += target_id in variable
            informative_constant += target_id not in variable
        source = ((normalized_source(encoded['global'][source_id],
                                     mean_delta[source_id])-
                   correction.source_center) @ correction.source_basis /
                  correction.source_scale)
        target = ((normalized_target(encoded['targets'][target_id])-
                   correction.target_center) @ correction.target_basis /
                  correction.target_scale)
        input_features = torch.cat((source, source*target)).numpy()
        latent = reward_delta*noise/(2*run['sigma'])
        latent_gradients.append(latent)
        weight_gradients.append(np.outer(latent, input_features).reshape(-1))
    latent_gradients = np.stack(latent_gradients)
    weight_gradients = np.stack(weight_gradients)
    nonzero = np.linalg.norm(latent_gradients, axis=1) > 0
    if (int(nonzero.sum()) != audit['summary']['positive_only']+
            audit['summary']['negative_only'] or
            positive_only != audit['summary']['positive_only'] or
            negative_only != audit['summary']['negative_only']):
        raise ValueError('Changed audited paired reward differences')
    summary = {'pairs': len(run['train']), 'reward_different': int(nonzero.sum()),
        'zero_reward_gradient': int((~nonzero).sum()),
        'positive_only': int(positive_only),
        'negative_only': int(negative_only),
        'source_sensitive_targets': len(variable),
        'informative_on_source_sensitive_target': informative_variable,
        'informative_on_constant_target': informative_constant,
        'latent_first_second_half_cosine': cosine(
            latent_gradients[:36].sum(0), latent_gradients[36:].sum(0)),
        'weight_first_second_half_cosine': cosine(
            weight_gradients[:36].sum(0), weight_gradients[36:].sum(0)),
        'latent_sign_flip': exact_sign_flip_tail(latent_gradients[nonzero]),
        'weight_sign_flip': exact_sign_flip_tail(weight_gradients[nonzero])}
    result = {'protocol': 'Reconstruction of the 72 audited antithetic LoRA reward updates and frozen 96-dimensional centered-feature parameter gradients; exact sign-flip coherence reference is descriptive, not a stationary-RL significance test',
        'report_sha256': file_hash(args.report),
        'audit_sha256': file_hash(args.audit),
        'model_sha256': file_hash(args.model),
        'features_sha256': file_hash(args.features),
        'matrix_sha256': file_hash(args.matrix),
        'review_sha256': file_hash(args.review),
        'summary': summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_antithetic_v4_seed42_20261007.json'))
    parser.add_argument('--audit', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_antithetic_v4_seed42_audited_20261007.json'))
    parser.add_argument('--model', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_antithetic_v4_seed42_20261007.pt'))
    parser.add_argument('--features', type=Path, default=Path('results/trajectory_hyperlora/alf_future_reward_train8x18_features_20261007.pt'))
    parser.add_argument('--matrix', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_20261007.json'))
    parser.add_argument('--review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_antithetic_reward_signal_v5_20261007.json'))
    analyze(parser.parse_args())
