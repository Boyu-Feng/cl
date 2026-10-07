"""Train-only grouped CV for a frozen-feature future-reward update gate.

This is a diagnostic, not an independent online benchmark. The v5 PCA frame
predates the new update labels; no ALFWorld family or action slot is used.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.centered_feedback_residual_v3 import (
    normalized_source, normalized_target,
)


LAMBDAS = (100., 10., 1., .1)
MODES = ('constant', 'source', 'target', 'interaction')


def inputs(args):
    report = json.loads(args.output.read_text())
    audit = json.loads(args.audit_output.read_text())
    review = json.loads(args.update_review.read_text())
    parent = json.loads(args.parent_review.read_text())
    features = torch.load(args.features, map_location='cpu', weights_only=True)
    state = torch.load(args.residual, map_location='cpu', weights_only=True)
    if (audit['raw_report_sha256'] != file_hash(args.output) or
            audit['review_sha256'] != file_hash(args.update_review) or
            audit['summary']['pairs'] != 108 or
            audit['summary']['failed_replays'] != 0 or
            report['review_sha256'] != file_hash(args.update_review) or
            report['failures'] or len(report['rows']) != 108 or
            review['parent_review_sha256'] != file_hash(args.parent_review) or
            features['review_sha256'] != file_hash(args.parent_review) or
            features['checkpoint_sha256'] != file_hash(args.checkpoint) or
            len(parent['targets']) != 18 or
            len({x['sequence_index'] for x in parent['targets']}) != 18 or
            state['model_kind'] != 'centered_bilinear_action_sensitive_v5'):
        raise ValueError('Changed reviewed counterfactual labels or frozen features')
    delta = torch.stack([features['deltas'][i, :int(n)].mean(0)
        for i, n in enumerate(features['event_counts'])])
    weights = state['correction']
    source = torch.stack([normalized_source(features['global'][i], delta[i])
        for i in range(8)])
    source = ((source - weights['source_center']) @
        weights['source_basis'] / weights['source_scale'])
    target = torch.stack([normalized_target(x) for x in features['targets']])
    target = ((target - weights['target_center']) @
        weights['target_basis'] / weights['target_scale'])
    arrays = {name: [] for name in MODES if name != 'constant'}
    labels = []
    groups = {'source_transition': [], 'future_target': []}
    for binding, row in zip(review['targets'], report['rows'], strict=True):
        if (binding['input_content_sha256'] !=
                row['input_content_sha256'] or
                binding['transition_index'] != row['transition_index'] or
                binding['target_id'] != row['target_id']):
            raise ValueError('Changed frozen training row order')
        old, new, task = (row['first_source_id'],
                          row['next_source_id'], row['target_id'])
        s, d, q = source[old], source[new] - source[old], target[task]
        arrays['source'].append(torch.cat((s, d)).numpy())
        arrays['target'].append(q.numpy())
        arrays['interaction'].append(torch.cat((s, d, q, d*q)).numpy())
        labels.append(row['update']['reward'] - row['freeze']['reward'])
        groups['source_transition'].append(row['transition_index'])
        groups['future_target'].append(
            parent['targets'][task]['sequence_index'])
    return ({key: np.stack(value).astype('float64')
             for key, value in arrays.items()},
            np.asarray(labels, dtype='float64'),
            {key: np.asarray(value) for key, value in groups.items()},
            report, audit)


def fit_predict(x, y, train, test, strength):
    left = x[train]
    center = left.mean(0)
    scale = np.maximum(left.std(0), 1e-6)
    normalized = (left-center)/scale
    target_center = y[train].mean()
    lhs = normalized.T @ normalized + strength*np.eye(x.shape[1])
    weight = np.linalg.solve(lhs, normalized.T @
                             (y[train]-target_center))
    return target_center + (x[test]-center)/scale @ weight


def prediction(arrays, y, train, test, mode, strength):
    if mode == 'constant':
        return np.full(len(test), y[train].mean())
    return fit_predict(arrays[mode], y, train, test, strength)


def choose_hyperparameters(arrays, y, groups, train):
    candidates = [('constant', None)] + [
        (mode, strength) for mode in MODES[1:] for strength in LAMBDAS]
    ranked = []
    for mode, strength in candidates:
        gain = 0.
        for group in np.unique(groups[train]):
            validation = train[groups[train] == group]
            fit = train[groups[train] != group]
            scores = prediction(arrays, y, fit, validation, mode, strength)
            gain += float(np.dot(y[validation], scores > 0))
        ranked.append((gain, mode, strength))
    # Stable order prefers constant and stronger regularization on equal gains.
    best = max(enumerate(ranked), key=lambda pair:
               (pair[1][0], -pair[0]))[1]
    return best[1], best[2], best[0]


def cross_validate(arrays, y, groups):
    predictions = np.empty(len(y), dtype='float64')
    selected = []
    for group in np.unique(groups):
        test = np.flatnonzero(groups == group)
        train = np.flatnonzero(groups != group)
        mode, strength, inner_gain = choose_hyperparameters(
            arrays, y, groups, train)
        predictions[test] = prediction(arrays, y, train, test,
                                       mode, strength)
        selected.append({'heldout_group': int(group), 'n': len(test),
            'selected_mode': mode, 'selected_lambda': strength,
            'inner_train_fold_gain': inner_gain,
            'heldout_gain_vs_freeze': float(np.dot(
                y[test], predictions[test] > 0))})
    decision = predictions > 0
    return {'heldout_pairs': len(y),
        'gain_vs_always_freeze': float(np.dot(y, decision)),
        'update_decisions': int(decision.sum()),
        'selected_modes': dict(Counter(x['selected_mode'] for x in selected)),
        'folds': selected,
        'rows': [{'index': i, 'predicted_reward_difference': float(score),
                  'update': bool(decision[i]),
                  'observed_reward_difference': float(y[i])}
                 for i, score in enumerate(predictions)]}


def run(args):
    if args.probe_output.exists():
        raise FileExistsError(args.probe_output)
    arrays, y, groups, report, audit = inputs(args)
    analyses = {name: cross_validate(arrays, y, group)
                for name, group in groups.items()}
    baseline = {'pairs': len(y),
        'always_freeze_success': sum(x['freeze']['reward']
                                     for x in report['rows']),
        'always_update_success': sum(x['update']['reward']
                                     for x in report['rows']),
        'hindsight_pair_oracle': sum(max(x['freeze']['reward'],
                                         x['update']['reward'])
                                     for x in report['rows']),
        'update_only': int((y > 0).sum()),
        'freeze_only': int((y < 0).sum())}
    value = {'protocol': 'Train-domain nested grouped CV of frozen-PCA ridge prediction for update-minus-freeze future reward; no family slots; neither CV axis is an independent official benchmark; PCA itself was fit on this old train bank',
        'raw_report_sha256': file_hash(args.output),
        'audit_sha256': file_hash(args.audit_output),
        'review_sha256': file_hash(args.update_review),
        'features_sha256': file_hash(args.features),
        'residual_sha256': file_hash(args.residual),
        'feature_modes': list(MODES), 'ridge_lambdas': list(LAMBDAS),
        'decision_threshold': 0.,
        'baseline': baseline, 'cross_validation': analyses}
    if (audit['summary']['freeze'] != baseline['always_freeze_success'] or
            audit['summary']['update'] != baseline['always_update_success']):
        raise ValueError('Probe labels differ from original-environment audit')
    args.probe_output.parent.mkdir(parents=True, exist_ok=True)
    args.probe_output.write_text(json.dumps(value, ensure_ascii=False,
                                            indent=2) + '\n')
    print(json.dumps({'baseline': baseline,
        'source_transition_cv_gain': analyses['source_transition'][
            'gain_vs_always_freeze'],
        'future_target_cv_gain': analyses['future_target'][
            'gain_vs_always_freeze']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--parent-review', type=Path, default=Path(
        'data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--update-review', type=Path, default=Path(
        'data/annotations/alf_incremental_update108_v9_reviewed_20261007.json'))
    parser.add_argument('--features', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_future_reward_train8x18_features_20261007.pt'))
    parser.add_argument('--residual', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_20261007.pt'))
    parser.add_argument('--output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_incremental_update108_v9_20261007.json'))
    parser.add_argument('--audit-output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_incremental_update108_v9_audited_20261007.json'))
    parser.add_argument('--probe-output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_incremental_gate_v10_nested_cv_20261007.json'))
    run(parser.parse_args())
