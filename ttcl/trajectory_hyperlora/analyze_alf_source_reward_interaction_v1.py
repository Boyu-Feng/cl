"""Audit and decompose source-by-target ALFWorld terminal-reward matrices.

For a balanced matrix R, the additive least-squares decomposition is
R_st = mu + alpha_s + beta_t + gamma_st with zero row/column sums in gamma.
The orthogonal sums of squares are descriptive; they are not a generalization
test or an estimate of how well a selector could be learned from few games.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash


def analyze_matrix(report_path: Path, audit_path: Path, review_path: Path):
    report = json.loads(report_path.read_text())
    audit = json.loads(audit_path.read_text())
    review = json.loads(review_path.read_text())
    sources = review['sources']
    targets = review['targets']
    if (report['review_sha256'] != file_hash(review_path) or
            audit['review_sha256'] != file_hash(review_path) or
            audit['raw_report_sha256'] != file_hash(report_path) or
            report['checkpoint_sha256'] != audit['checkpoint_sha256'] or
            report['failures'] or
            len(report['pairs']) != len(sources)*len(targets) or
            audit['summary']['pairs'] != len(report['pairs'])):
        raise ValueError('Changed or incomplete audited reward matrix')
    n_source, n_target = len(sources), len(targets)
    values = np.full((n_source, n_target), np.nan, dtype=np.float64)
    bindings = {(x['source_id'], x['target_id']): x
                for x in review['pair_bindings']}
    for row in report['pairs']:
        i, j = row['source_id'], row['target_id']
        expected = bindings[i, j]
        if (not np.isnan(values[i, j]) or
                row['input_content_sha256'] !=
                    expected['input_content_sha256'] or
                row['episode']['status'] != 'complete' or
                row['episode']['reward'] not in (0., 1.)):
            raise ValueError('Duplicate, changed, or invalid reward pair')
        values[i, j] = row['episode']['reward']
    if np.isnan(values).any():
        raise ValueError('Missing source-target reward')
    mu = values.mean()
    source = values.mean(axis=1, keepdims=True)-mu
    target = values.mean(axis=0, keepdims=True)-mu
    interaction = values-mu-source-target
    total_ss = float(np.square(values-mu).sum())
    source_ss = float(n_target*np.square(source).sum())
    target_ss = float(n_source*np.square(target).sum())
    interaction_ss = float(np.square(interaction).sum())
    if (abs(total_ss-source_ss-target_ss-interaction_ss) > 1e-9 or
            abs(interaction.sum(axis=0)).max() > 1e-12 or
            abs(interaction.sum(axis=1)).max() > 1e-12):
        raise ValueError('Additive decomposition failed numerical audit')
    singular = np.linalg.svd(interaction, compute_uv=False)
    energy = np.cumsum(np.square(singular))/interaction_ss
    result = {'report_sha256': file_hash(report_path),
        'audit_sha256': file_hash(audit_path),
        'review_sha256': file_hash(review_path),
        'sources': n_source, 'targets': n_target,
        'mean_reward': float(mu),
        'total_centered_ss': total_ss,
        'source_main_ss': source_ss,
        'target_main_ss': target_ss,
        'source_target_interaction_ss': interaction_ss,
        'source_main_fraction': source_ss/total_ss,
        'target_main_fraction': target_ss/total_ss,
        'source_target_interaction_fraction': interaction_ss/total_ss,
        'interaction_top1_energy_fraction': float(energy[0]),
        'interaction_top2_energy_fraction': float(energy[1]),
        'source_dependent_targets': int(sum(
            len(set(values[:, j])) > 1 for j in range(n_target))),
        'best_fixed_source_wins': int(values.sum(axis=1).max()),
        'posthoc_target_oracle_wins': int(values.max(axis=0).sum())}
    if (result['source_dependent_targets'] !=
            audit['summary']['source_dependent_targets'] or
            result['best_fixed_source_wins'] !=
            audit['summary']['best_static_source'] or
            result['posthoc_target_oracle_wins'] !=
            audit['summary']['oracle']):
        raise ValueError('Audited reward-matrix summary changed')
    return result


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    matrices = {'train8x18': analyze_matrix(
        args.train_report, args.train_audit, args.train_review),
        'disjoint12': analyze_matrix(
            args.holdout_report, args.holdout_audit, args.holdout_review)}
    if matrices['train8x18']['sources'] != 8 or matrices['train8x18']['targets'] != 18 or matrices['disjoint12']['sources'] != 8 or matrices['disjoint12']['targets'] != 12:
        raise ValueError('Changed fixed matrix shapes')
    result = {'protocol': 'Exact balanced two-way least-squares decomposition of independently audited binary source-target terminal rewards; descriptive interaction, no inferential or learned-selector claim',
        'matrices': matrices}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({name: {key: value[key] for key in (
        'source_main_fraction', 'target_main_fraction',
        'source_target_interaction_fraction',
        'source_dependent_targets', 'best_fixed_source_wins',
        'posthoc_target_oracle_wins')} for name, value in matrices.items()}),
        flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--train-report', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_20261007.json'))
    parser.add_argument('--train-audit', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_audited_20261007.json'))
    parser.add_argument('--holdout-review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_holdout_8x12_v3c_reviewed_20261007.json'))
    parser.add_argument('--holdout-report', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_holdout_8x12_v3c_merged_20261007.json'))
    parser.add_argument('--holdout-audit', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_holdout_8x12_v3c_merged_audited_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_source_reward_interaction_v1_20261007.json'))
    main(parser.parse_args())
