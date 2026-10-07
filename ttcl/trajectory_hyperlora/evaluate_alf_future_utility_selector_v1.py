"""Evaluate a frozen reward-trained source selector on disjoint targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.train_alf_future_utility_selector_v1 import (
    pair_features, target_reward_matrix,
)


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    model = torch.load(args.model, map_location='cpu', weights_only=True)
    encoded = torch.load(args.features, map_location='cpu', weights_only=True)
    audit = json.loads(args.audit.read_text())
    report = json.loads(args.matrix.read_text())
    training = json.loads(args.training_report.read_text())
    if (model['train_audit_sha256'] != training['training_matrix_audit_sha256'] or
            model['train_features_sha256'] != training['features_sha256'] or
            model['checkpoint_sha256'] != audit['checkpoint_sha256'] or
            encoded['checkpoint_sha256'] != audit['checkpoint_sha256'] or
            encoded['review_sha256'] != audit['review_sha256'] or
            encoded['target_policy'] != 'independent12' or
            audit['target_policy'] != 'independent12' or
            audit['raw_report_sha256'] != file_hash(args.matrix) or
            training['saved_model_sha256'] != file_hash(args.model) or
            report['review_sha256'] != audit['review_sha256'] or
            audit['summary']['checkpoint_training_candidate_overlap'] != 0 or
            audit['summary']['warmstart_training_target_overlap'] != 0 or
            audit['summary']['actual_checkpoint_label_target_overlap'] != 0 or
            audit['summary']['prior_training_game_overlap'] != 0 or
            audit['summary']['prior_training_sequence_overlap'] != 0 or
            digest([row['episode']['initial_observation']
                    for row in report['base']]) !=
                    encoded['base_observations_sha256']):
        raise ValueError('Changed model or non-independent holdout')
    for source_name, current_name in (
            ('source_global', 'global'), ('source_events', 'events'),
            ('source_deltas', 'deltas'),
            ('source_event_counts', 'event_counts')):
        if not torch.allclose(model[source_name], encoded[current_name],
                              rtol=1e-5, atol=1e-5):
            raise ValueError('Holdout source bank or interaction encoding changed')
    rewards = target_reward_matrix(report, 8, 12)
    features = pair_features(encoded, model['target_center'], model['variant'])
    if features.shape[-1] != model['training_features'].shape[-1]:
        raise ValueError('Feature width changed')
    query = features @ model['training_features'].T
    predictions = query @ model['alpha']
    rows = []
    for target_id, scores in enumerate(predictions):
        selected = int(torch.argmax(scores).item())
        if model['mode'] == 'advantage' and scores[selected] <= 0:
            selected = -1
        static = model['best_static_source']
        base_reward = report['base'][target_id]['episode']['reward']
        rows.append({'target_id': target_id, 'selected_source': selected,
            'selected_reward': float(base_reward if selected == -1
                                     else rewards[target_id, selected]),
            'static_source': static,
            'static_reward': float(base_reward if static == -1
                                   else rewards[target_id, static]),
            'base_reward': base_reward,
            'oracle_reward': float(max(rewards[target_id].max(),
                                       base_reward)),
            'random_expected': float(rewards[target_id].mean()),
            'scores': scores.tolist()})
    summary = {'targets': len(rows),
        'selected': sum(row['selected_reward'] for row in rows),
        'static_train_selected': sum(row['static_reward'] for row in rows),
        'base': sum(row['base_reward'] for row in rows),
        'oracle': sum(row['oracle_reward'] for row in rows),
        'random_expected': sum(row['random_expected'] for row in rows),
        'selected_only_vs_static': sum(
            row['selected_reward'] > row['static_reward'] for row in rows),
        'static_only_vs_selected': sum(
            row['selected_reward'] < row['static_reward'] for row in rows)}
    result = {'protocol': 'Frozen target-centered official-reward source selector evaluated by fully crossed, disjoint official ALFWorld train-domain holdout; no holdout labels used in model choice',
        'model_sha256': file_hash(args.model),
        'training_report_sha256': file_hash(args.training_report),
        'holdout_features_sha256': file_hash(args.features),
        'holdout_audit_sha256': file_hash(args.audit),
        'variant': model['variant'], 'mode': model['mode'],
        'summary': summary, 'targets': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('results/trajectory_hyperlora/alf_future_utility_selector_train8x18_20261007.pt'))
    parser.add_argument('--training-report', type=Path, default=Path('results/trajectory_hyperlora/alf_future_utility_selector_train8x18_20261007.json'))
    parser.add_argument('--features', type=Path, default=Path('results/trajectory_hyperlora/alf_future_reward_holdout8x12_v3c_features_20261007.pt'))
    parser.add_argument('--matrix', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_holdout_8x12_v3c_20261007.json'))
    parser.add_argument('--audit', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_holdout_8x12_v3c_audited_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_future_utility_selector_holdout8x12_20261007.json'))
    evaluate(parser.parse_args())

if __name__ == '__main__':
    main()
