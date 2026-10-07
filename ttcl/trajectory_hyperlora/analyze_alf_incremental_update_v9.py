"""Audit latent update geometry against paired future-reward effects."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.centered_feedback_residual_v3 import CenteredEvidenceResidual


def analyze(args):
    if args.analysis_output.exists():
        raise FileExistsError(args.analysis_output)
    review = json.loads(args.update_review.read_text())
    report = json.loads(args.output.read_text())
    audit = json.loads(args.audit_output.read_text())
    features = torch.load(args.features, map_location='cpu', weights_only=True)
    state = torch.load(args.residual, map_location='cpu', weights_only=True)
    if (audit['raw_report_sha256'] != file_hash(args.output) or
            audit['review_sha256'] != file_hash(args.update_review) or
            audit['summary']['pairs'] != 108 or
            audit['summary']['failed_replays'] != 0 or
            review['parent_review_sha256'] != file_hash(args.parent_review) or
            features['review_sha256'] != file_hash(args.parent_review) or
            features['checkpoint_sha256'] != file_hash(args.checkpoint) or
            report['review_sha256'] != file_hash(args.update_review) or
            report['residual_sha256'] != file_hash(args.residual) or
            report['failures'] or len(report['rows']) != 108 or
            features['global'].shape != (8, 2560) or
            features['targets'].shape != (18, 2560)):
        raise ValueError('Changed reviewed labels, features, or audited run')
    model = CenteredEvidenceResidual.from_state(state['correction'])
    model.load_state_dict(state['correction'])
    model.eval()
    source_delta = torch.stack([features['deltas'][i, :int(count)].mean(0)
        for i, count in enumerate(features['event_counts'])])
    target_groups = defaultdict(list)
    transition_groups = defaultdict(list)
    code_rows = []
    with torch.no_grad():
        for reviewed, row in zip(review['targets'], report['rows'], strict=True):
            if (reviewed['input_content_sha256'] !=
                    row['input_content_sha256'] or
                    reviewed['transition_index'] !=
                    row['transition_index'] or
                    reviewed['target_id'] != row['target_id']):
                raise ValueError('Changed paired result order')
            old, new, target = (row['first_source_id'],
                                row['next_source_id'], row['target_id'])
            first = model(features['global'][old], source_delta[old],
                          features['targets'][target])
            updated = model((features['global'][old] +
                features['global'][new]) / 2,
                (source_delta[old] + source_delta[new]) / 2,
                features['targets'][target])
            if (abs(first.norm().item() - row['freeze_code_norm']) > 1e-4 or
                    abs(updated.norm().item() -
                        row['update_code_norm']) > 1e-4):
                raise ValueError('Frozen encoded features do not reproduce LoRA code')
            delta = updated - first
            effect = row['update']['reward'] - row['freeze']['reward']
            item = {'transition_index': row['transition_index'],
                'target_id': target,
                'reward_difference': effect,
                'freeze_reward': row['freeze']['reward'],
                'update_reward': row['update']['reward'],
                'changed_trajectory': (row['freeze']['trajectory'] !=
                                       row['update']['trajectory']),
                'code_delta_norm': delta.norm().item(),
                'code_delta_over_old_norm': (delta.norm() /
                    first.norm().clamp_min(1e-8)).item(),
                'code_cosine': F.cosine_similarity(
                    first.unsqueeze(0), updated.unsqueeze(0)).item()}
            if not all(math.isfinite(item[k]) for k in (
                    'code_delta_norm', 'code_delta_over_old_norm',
                    'code_cosine')):
                raise ValueError('Nonfinite parameter geometry')
            code_rows.append(item)
            target_groups[target].append(item)
            transition_groups[row['transition_index']].append(item)
    def summary(rows):
        return {'pairs': len(rows),
            'freeze_success': sum(x['freeze_reward'] for x in rows),
            'update_success': sum(x['update_reward'] for x in rows),
            'hindsight_pair_oracle': sum(max(x['freeze_reward'],
                                             x['update_reward']) for x in rows),
            'update_only': sum(x['reward_difference'] > 0 for x in rows),
            'freeze_only': sum(x['reward_difference'] < 0 for x in rows),
            'changed_trajectories': sum(x['changed_trajectory'] for x in rows),
            'mean_code_delta_norm': sum(x['code_delta_norm']
                                        for x in rows) / len(rows),
            'mean_code_delta_over_old_norm': sum(
                x['code_delta_over_old_norm'] for x in rows) / len(rows),
            'mean_code_cosine': sum(x['code_cosine'] for x in rows) / len(rows)}
    result = {'protocol': 'Descriptive latent geometry and paired update credit, conditioned on independently replayed official train rollouts; hindsight oracle is not deployable and rows sharing sources/targets are not independent',
        'raw_report_sha256': file_hash(args.output),
        'audit_sha256': file_hash(args.audit_output),
        'review_sha256': file_hash(args.update_review),
        'features_sha256': file_hash(args.features),
        'residual_sha256': file_hash(args.residual),
        'overall': summary(code_rows),
        'by_transition': {str(key): summary(rows) for key, rows
                          in sorted(transition_groups.items())},
        'by_target': {str(key): summary(rows) for key, rows
                      in sorted(target_groups.items())},
        'rows': code_rows}
    if (len(result['by_transition']) != 6 or
            any(x['pairs'] != 18 for x in result['by_transition'].values()) or
            len(result['by_target']) != 18 or
            any(x['pairs'] != 6 for x in result['by_target'].values())):
        raise ValueError('Incomplete source/target cross design')
    args.analysis_output.parent.mkdir(parents=True, exist_ok=True)
    args.analysis_output.write_text(json.dumps(result, ensure_ascii=False,
                                                indent=2) + '\n')
    print(json.dumps({'overall': result['overall'],
        'by_transition': result['by_transition']}), flush=True)


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
    parser.add_argument('--analysis-output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_incremental_update108_v9_geometry_20261007.json'))
    analyze(parser.parse_args())
