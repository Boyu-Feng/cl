"""Train a generic trajectory-LoRA source selector from paired future reward.

A frozen Qwen encodes full histories, action-feedback events and target
observations. Target-grouped reward differences train a regularized kernel
utility estimator. No task-family labels, action dictionaries or holdout
rewards are used for training or model selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash


def source_summaries(encoded):
    counts = encoded['event_counts'].long()
    events = encoded['events'].double()
    deltas = encoded['deltas'].double()
    mean_event = torch.stack([events[i, :count].mean(0)
        for i, count in enumerate(counts)])
    mean_delta = torch.stack([deltas[i, :count].mean(0)
        for i, count in enumerate(counts)])
    progression = torch.stack([events[i, count - 1] - events[i, 0]
        for i, count in enumerate(counts)])
    return (encoded['global'].double(), mean_event, mean_delta, progression)


def pair_features(encoded, target_center, variant):
    if variant not in ('global', 'event'):
        raise ValueError('Unknown utility feature variant')
    sources = source_summaries(encoded)
    targets = encoded['targets'].double()
    query = F.normalize(targets - target_center, dim=-1)
    groups = [query.unsqueeze(1).expand(-1, len(sources[0]), -1)]
    for group_index, source in enumerate(sources):
        if variant == 'global' and group_index > 0:
            break
        source = F.normalize(source - source.mean(0), dim=-1)
        standalone = source.unsqueeze(0).expand(len(targets), -1, -1)
        interaction = source.unsqueeze(0) * query.unsqueeze(1)
        groups.extend((F.normalize(standalone, dim=-1),
                       F.normalize(interaction, dim=-1)))
    return torch.cat(groups, dim=-1) / len(groups) ** .5


def target_reward_matrix(report, n_sources, n_targets):
    if len(report['pairs']) != n_sources * n_targets or report['failures']:
        raise ValueError('Incomplete source-target reward matrix')
    rewards = torch.zeros(n_targets, n_sources, dtype=torch.float64)
    for index, row in enumerate(report['pairs']):
        source_id, target_id = divmod(index, n_targets)
        if (row['source_id'] != source_id or row['target_id'] != target_id or
                row['episode']['status'] != 'complete'):
            raise ValueError('Source-target reward order changed')
        rewards[target_id, source_id] = row['episode']['reward']
    return rewards


def kernel_fit(x, y, regularization=1.0):
    kernel = x @ x.T
    alpha = torch.linalg.solve(kernel + regularization * torch.eye(
        len(kernel), dtype=kernel.dtype), y)
    return alpha


def best_static(rewards, base_rewards):
    source = int(torch.argmax(rewards.sum(0)).item())
    return source if rewards[:, source].sum() > base_rewards.sum() else -1


def score_fold(encoded, rewards, base_rewards, holdout_id, variant, mode):
    train_ids = [i for i in range(len(rewards)) if i != holdout_id]
    center = encoded['targets'][train_ids].double().mean(0)
    features = pair_features(encoded, center, variant)
    train_x = features[train_ids].reshape(-1, features.shape[-1])
    if mode == 'relative':
        train_y = (rewards[train_ids] - rewards[train_ids].mean(-1,
            keepdim=True)).reshape(-1)
    elif mode == 'advantage':
        train_y = (rewards[train_ids] - base_rewards[train_ids, None]).reshape(-1)
    else:
        raise ValueError('Unknown future utility objective')
    alpha = kernel_fit(train_x, train_y)
    heldout_x = features[holdout_id]
    scores = (heldout_x @ train_x.T) @ alpha
    predicted = int(torch.argmax(scores).item())
    if mode == 'advantage' and scores[predicted] <= 0:
        predicted = -1
    static = best_static(rewards[train_ids], base_rewards[train_ids])
    return {'target_id': holdout_id, 'selected_source': predicted,
        'selected_reward': float(base_rewards[holdout_id] if predicted == -1
                                 else rewards[holdout_id, predicted]),
        'static_source': static,
        'static_reward': float(base_rewards[holdout_id] if static == -1
                              else rewards[holdout_id, static]),
        'base_reward': float(base_rewards[holdout_id]),
        'oracle_reward': float(max(rewards[holdout_id].max(),
                                    base_rewards[holdout_id])),
        'random_expected': float(rewards[holdout_id].mean()),
        'scores': scores.tolist()}


def train(args):
    if args.output.exists() or args.save_model.exists():
        raise FileExistsError('Use fresh selector report and checkpoint')
    encoded = torch.load(args.features, map_location='cpu', weights_only=True)
    audit = json.loads(args.audit.read_text())
    report = json.loads(args.matrix.read_text())
    if (encoded['review_sha256'] != audit['review_sha256'] or
            encoded['checkpoint_sha256'] != audit['checkpoint_sha256'] or
            encoded['target_policy'] != 'train18' or
            audit['raw_report_sha256'] != file_hash(args.matrix) or
            report['review_sha256'] != audit['review_sha256'] or
            len(report['base']) != 18 or len(report['pairs']) != 144 or
            digest([row['episode']['initial_observation']
                    for row in report['base']]) !=
                    encoded['base_observations_sha256']):
        raise ValueError('Changed or unaudited training features/rewards')
    rewards = target_reward_matrix(report, 8, 18)
    base_rewards = torch.tensor([row['episode']['reward']
        for row in report['base']], dtype=torch.float64)
    cv = {}
    for variant in ('global', 'event'):
        for mode in ('relative', 'advantage'):
            key = f'{variant}_{mode}'
            folds = [score_fold(encoded, rewards, base_rewards,
                target_id, variant, mode) for target_id in range(18)]
            cv[key] = {'selected': sum(row['selected_reward'] for row in folds),
                'static_train_only': sum(row['static_reward'] for row in folds),
                'base': sum(row['base_reward'] for row in folds),
                'oracle': sum(row['oracle_reward'] for row in folds),
                'random_expected': sum(row['random_expected'] for row in folds),
                'folds': folds}
    selected_key = max(cv, key=lambda name: (cv[name]['selected'],
        name.endswith('relative'), name.startswith('global')))
    selected_variant, selected_mode = selected_key.split('_')
    center = encoded['targets'].double().mean(0)
    features = pair_features(encoded, center, selected_variant)
    x = features.reshape(-1, features.shape[-1])
    y = ((rewards - rewards.mean(-1, keepdim=True)) if
         selected_mode == 'relative' else
         (rewards - base_rewards[:, None])).reshape(-1)
    alpha = kernel_fit(x, y)
    static = best_static(rewards, base_rewards)
    saved = {'protocol': 'Frozen generic event-aware source utility selector trained on target-centered future official reward, grouped leave-one-target-out model selection, unit kernel ridge regularization',
        'variant': selected_variant, 'mode': selected_mode,
        'target_center': center,
        'training_features': x, 'alpha': alpha,
        'source_global': encoded['global'],
        'source_events': encoded['events'],
        'source_deltas': encoded['deltas'],
        'source_event_counts': encoded['event_counts'],
        'best_static_source': static,
        'train_review_sha256': audit['review_sha256'],
        'train_audit_sha256': file_hash(args.audit),
        'train_features_sha256': file_hash(args.features),
        'checkpoint_sha256': audit['checkpoint_sha256']}
    args.save_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, args.save_model)
    result = {'protocol': saved['protocol'],
        'training_matrix_audit_sha256': file_hash(args.audit),
        'features_sha256': file_hash(args.features),
        'checkpoint_sha256': audit['checkpoint_sha256'],
        'variant': selected_variant, 'mode': selected_mode,
        'best_static_source': static,
        'best_static_train_success': float(base_rewards.sum() if static == -1
                                           else rewards[:, static].sum()),
        'cross_validation': cv,
        'saved_model_sha256': file_hash(args.save_model)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'variant': selected_variant, 'mode': selected_mode,
        'cv': {key: {metric: row[metric] for metric in
            ('selected', 'static_train_only', 'base', 'oracle',
             'random_expected')}
            for key, row in cv.items()},
        'best_static_source': static}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=Path, default=Path('results/trajectory_hyperlora/alf_future_reward_train8x18_features_20261007.pt'))
    parser.add_argument('--matrix', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_20261007.json'))
    parser.add_argument('--audit', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_audited_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_future_utility_selector_train8x18_20261007.json'))
    parser.add_argument('--save-model', type=Path, default=Path('results/trajectory_hyperlora/alf_future_utility_selector_train8x18_20261007.pt'))
    train(parser.parse_args())

if __name__ == '__main__':
    main()
