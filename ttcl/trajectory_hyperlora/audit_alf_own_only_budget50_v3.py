"""Replay 50-step budget-sensitivity interventions against fixed train games."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.audit_alf_own_only_first_divergence_v2 import replay


def audit(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.report.read_text())
    prior = json.loads(args.prior_intervention.read_text())
    reviewed = json.loads(args.first_divergence_review.read_text())
    rollouts = json.loads(args.rollouts.read_text())
    if (report['prior_intervention_sha256'] !=
            file_hash(args.prior_intervention) or
            report['first_divergence_review_sha256'] !=
            file_hash(args.first_divergence_review) or
            report['rollouts_sha256'] != file_hash(args.rollouts) or
            report['checkpoint_sha256'] != rollouts['checkpoint_sha256'] or
            report['failures'] or prior['failures'] or
            len(prior['rows']) != 42):
        raise ValueError('Changed 50-step intervention lineage')
    original = {x['game']: x for x in rollouts['games']}
    by_binding = {x['input_content_sha256']: x
                  for x in reviewed['pairs']}
    expected = [x for x in prior['rows']
                if x['counterfactual_reward'] == 0.]
    if (len(report['rows']) != len(expected) or
            report['n_prior_losses'] != len(expected) or
            len({x['input_content_sha256'] for x in report['rows']}) !=
                len(expected)):
        raise ValueError('Missing 30-step losses or duplicate 50-step results')
    recovered = []
    for row, old in zip(report['rows'], expected, strict=True):
        key = row['input_content_sha256']
        pair = by_binding[key]
        if (key != old['input_content_sha256'] or
                row['game'] != pair['game'] or
                row['prior_30_step_reward'] != 0. or
                row['counterfactual_50_step_reward'] !=
                    row['episode']['reward'] or
                row['episode']['steps'] > 50 or
                file_hash(args.data_root / pair['game']) !=
                    pair['game_sha256']):
            raise ValueError('Changed budget-sensitivity target')
        replay(args.data_root / pair['game'],
            original[pair['game']]['arms']['own'],
            row['episode'], pair)
        if row['counterfactual_50_step_reward'] == 1.:
            recovered.append(key)
    summary = {'thirty_step_policy_conditional_losses': len(expected),
        'still_loss_at_fifty': len(expected)-len(recovered),
        'recovered_with_more_steps': len(recovered),
        'failed_50_step_replays': 0}
    result = {'protocol': 'Independent environment replay of every 30-step negative action intervention under 50-step budget',
        'report_sha256': file_hash(args.report),
        'prior_intervention_sha256': file_hash(args.prior_intervention),
        'fifty_step_recovery_bindings': recovered,
        'summary': summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, default=Path('results/trajectory_hyperlora/alf_own_only_first_divergence_budget50_v3_20261007.json'))
    parser.add_argument('--prior-intervention', type=Path, default=Path('results/trajectory_hyperlora/alf_own_only_first_divergence42_intervention_v2_20261007.json'))
    parser.add_argument('--first-divergence-review', type=Path, default=Path('data/annotations/alf_first_divergence42_reviewed_20261007.json'))
    parser.add_argument('--rollouts', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_train240_taskpair_current1000_20261006.json'))
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_own_only_first_divergence_budget50_v3_audited_20261007.json'))
    audit(parser.parse_args())
