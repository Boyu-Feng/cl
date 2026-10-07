"""Refit the frozen-feature source utility gate on two reviewed train matrices.

Features, objectives and regularization match v1. Model selection now holds
out entire plan sequences because the added games share sequence IDs with
old training games. This tests whether added games improve generalization
without treating same-sequence siblings as independent. It does not train
or modify the LoRA generator itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.train_alf_future_utility_selector_v1 import (
    best_static, kernel_fit, pair_features,
    target_reward_matrix,
)


def sequence_folds(encoded, rewards, base_rewards, sequence_ids,
                   variant, mode):
    rows = []
    for sequence_id in sorted(set(sequence_ids)):
        held_ids = [i for i, value in enumerate(sequence_ids)
                    if value == sequence_id]
        train_ids = [i for i, value in enumerate(sequence_ids)
                     if value != sequence_id]
        center = encoded['targets'][train_ids].double().mean(0)
        features = pair_features(encoded, center, variant)
        train_x = features[train_ids].reshape(-1, features.shape[-1])
        if mode == 'relative':
            train_y = (rewards[train_ids] -
                rewards[train_ids].mean(-1, keepdim=True)).reshape(-1)
        elif mode == 'advantage':
            train_y = (rewards[train_ids] -
                base_rewards[train_ids, None]).reshape(-1)
        else:
            raise ValueError('Unknown source utility objective')
        alpha = kernel_fit(train_x, train_y)
        static = best_static(rewards[train_ids],
                             base_rewards[train_ids])
        for target_id in held_ids:
            scores = (features[target_id] @ train_x.T) @ alpha
            selected = int(torch.argmax(scores).item())
            if mode == 'advantage' and scores[selected] <= 0:
                selected = -1
            rows.append({'target_id': target_id,
                'heldout_sequence_id': sequence_id,
                'heldout_sequence_games': len(held_ids),
                'selected_source': selected,
                'selected_reward': float(
                    base_rewards[target_id] if selected == -1 else
                    rewards[target_id, selected]),
                'static_source': static,
                'static_reward': float(
                    base_rewards[target_id] if static == -1 else
                    rewards[target_id, static]),
                'base_reward': float(base_rewards[target_id]),
                'oracle_reward': float(max(rewards[target_id].max(),
                                           base_rewards[target_id])),
                'random_expected': float(rewards[target_id].mean()),
                'scores': scores.tolist()})
    return sorted(rows, key=lambda row: row['target_id'])


def train(args):
    if args.output.exists() or args.save_model.exists():
        raise FileExistsError('Use fresh combined selector outputs')
    old_features = torch.load(args.old_features, map_location='cpu',
                              weights_only=True)
    new_features = torch.load(args.new_features, map_location='cpu',
                              weights_only=True)
    old_audit = json.loads(args.old_audit.read_text())
    new_audit = json.loads(args.new_audit.read_text())
    old_review = json.loads(args.old_review.read_text())
    new_review = json.loads(args.new_review.read_text())
    old_report = json.loads(args.old_matrix.read_text())
    new_report = json.loads(args.new_matrix.read_text())
    checkpoint_hash = old_audit['checkpoint_sha256']
    if (old_audit['review_sha256'] != file_hash(args.old_review) or
            new_audit['review_sha256'] != file_hash(args.new_review) or
            old_features['review_sha256'] != old_audit['review_sha256'] or
            old_features['checkpoint_sha256'] != checkpoint_hash or
            old_features['target_policy'] != 'train18' or
            old_audit['raw_report_sha256'] != file_hash(args.old_matrix) or
            new_features['review_sha256'] != new_audit['review_sha256'] or
            new_features['checkpoint_sha256'] != checkpoint_hash or
            new_features['target_policy'] != 'additional_train18_v3' or
            new_audit['raw_report_sha256'] != file_hash(args.new_matrix) or
            new_audit['checkpoint_sha256'] != checkpoint_hash or
            new_audit['summary']['failed_replays'] != 0 or
            new_audit['summary']['pairs'] != 144 or
            old_report['failures'] or new_report['failures'] or
            len(old_report['base']) != len(new_report['base']) or
            len(old_report['base']) != 18 or
            new_features['original_features_sha256'] !=
                file_hash(args.old_features) or
            any(not torch.allclose(old_features[k], new_features[k],
                rtol=0, atol=0) for k in
                ('global', 'events', 'deltas', 'event_counts'))):
        raise ValueError('Changed combined source, target or audit lineage')
    for features, report in ((old_features, old_report),
                             (new_features, new_report)):
        if (digest([row['episode']['initial_observation']
                    for row in report['base']]) !=
                features['base_observations_sha256']):
            raise ValueError('Changed target initial observations')
    encoded = {key: old_features[key] for key in
        ('global', 'events', 'deltas', 'event_counts')}
    encoded['targets'] = torch.cat((old_features['targets'],
                                    new_features['targets']), dim=0)
    rewards = torch.cat((target_reward_matrix(old_report, 8, 18),
                         target_reward_matrix(new_report, 8, 18)), dim=0)
    base = torch.tensor([row['episode']['reward'] for report in
        (old_report, new_report) for row in report['base']],
        dtype=torch.float64)
    sequence_ids = [row['sequence_index'] for review in
                    (old_review, new_review) for row in review['targets']]
    if (len(sequence_ids) != 36 or len(set(sequence_ids)) != 19 or
            any(old_review['targets'][i]['game'] !=
                old_report['base'][i]['game'] for i in range(18)) or
            any(new_review['targets'][i]['game'] !=
                new_report['base'][i]['game'] for i in range(18))):
        raise ValueError('Changed group-separated target sequence lineage')
    cv = {}
    for variant in ('global', 'event'):
        for mode in ('relative', 'advantage'):
            key = f'{variant}_{mode}'
            folds = sequence_folds(encoded, rewards, base, sequence_ids,
                                   variant, mode)
            cv[key] = {'selected': sum(x['selected_reward'] for x in folds),
                'static_train_only': sum(x['static_reward'] for x in folds),
                'base': sum(x['base_reward'] for x in folds),
                'oracle': sum(x['oracle_reward'] for x in folds),
                'random_expected': sum(x['random_expected'] for x in folds),
                'folds': folds}
    selected_key = max(cv, key=lambda name: (cv[name]['selected'],
        name.endswith('relative'), name.startswith('global')))
    variant, mode = selected_key.split('_')
    center = encoded['targets'].double().mean(0)
    features = pair_features(encoded, center, variant)
    x = features.reshape(-1, features.shape[-1])
    y = ((rewards-rewards.mean(-1, keepdim=True)) if
         mode == 'relative' else
         (rewards-base[:, None])).reshape(-1)
    alpha = kernel_fit(x, y)
    static = best_static(rewards, base)
    audit_binding = digest([file_hash(args.old_audit),
                            file_hash(args.new_audit)])
    feature_binding = digest([file_hash(args.old_features),
                              file_hash(args.new_features)])
    saved = {'protocol': 'Generic kernel future-reward source gate as v1 refit on 36 official train games; four prespecified variants selected by leave-one-plan-sequence-out CV; no family input or holdout labels',
        'variant': variant, 'mode': mode,
        'target_center': center,
        'training_features': x, 'alpha': alpha,
        'source_global': encoded['global'],
        'source_events': encoded['events'],
        'source_deltas': encoded['deltas'],
        'source_event_counts': encoded['event_counts'],
        'best_static_source': static,
        'train_review_sha256': digest([
            old_audit['review_sha256'], new_audit['review_sha256']]),
        'train_audit_sha256': audit_binding,
        'train_features_sha256': feature_binding,
        'cross_validation_unit': 'plan_sequence',
        'training_sequence_ids': sequence_ids,
        'checkpoint_sha256': checkpoint_hash}
    args.save_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, args.save_model)
    result = {'protocol': saved['protocol'],
        'old_audit_sha256': file_hash(args.old_audit),
        'new_audit_sha256': file_hash(args.new_audit),
        'old_features_sha256': file_hash(args.old_features),
        'new_features_sha256': file_hash(args.new_features),
        'training_matrix_audit_sha256': audit_binding,
        'features_sha256': feature_binding,
        'checkpoint_sha256': checkpoint_hash,
        'variant': variant, 'mode': mode,
        'cross_validation_unit': 'plan_sequence',
        'cross_validation_groups': len(set(sequence_ids)),
        'best_static_source': static,
        'best_static_train_success': float(
            base.sum() if static == -1 else rewards[:, static].sum()),
        'cross_validation': cv,
        'saved_model_sha256': file_hash(args.save_model)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False,
                                       indent=2)+'\n')
    print(json.dumps({'variant': variant, 'mode': mode,
        'cv': {key: {metric: row[metric] for metric in
            ('selected', 'static_train_only', 'base', 'oracle')}
            for key, row in cv.items()},
        'best_static_source': static}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--old-features', type=Path, default=Path('results/trajectory_hyperlora/alf_future_reward_train8x18_features_20261007.pt'))
    parser.add_argument('--new-features', type=Path, default=Path('results/trajectory_hyperlora/alf_future_reward_additional8x18_features_v3_20261007.pt'))
    parser.add_argument('--old-matrix', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_20261007.json'))
    parser.add_argument('--new-matrix', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_additional8x18_v3_20261007.json'))
    parser.add_argument('--old-audit', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_audited_20261007.json'))
    parser.add_argument('--new-audit', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_additional8x18_v3_audited_20261007.json'))
    parser.add_argument('--old-review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--new-review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_additional8x18_v3_reviewed_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_future_utility_selector_combined36_v2_20261007.json'))
    parser.add_argument('--save-model', type=Path, default=Path('results/trajectory_hyperlora/alf_future_utility_selector_combined36_v2_20261007.pt'))
    train(parser.parse_args())
