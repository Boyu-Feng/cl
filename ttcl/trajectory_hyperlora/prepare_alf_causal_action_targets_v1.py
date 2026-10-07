"""Bind policy-conditional first-action intervention rewards as new targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash


def expected(args):
    weak = json.loads(args.first_divergence_review.read_text())
    thirty = json.loads(args.intervention30.read_text())
    audit30 = json.loads(args.audit30.read_text())
    fifty = json.loads(args.intervention50.read_text())
    audit50 = json.loads(args.audit50.read_text())
    if (len(weak['pairs']) != 42 or len(thirty['rows']) != 42 or
            thirty['failures'] or fifty['failures'] or
            audit30['intervention_report_sha256'] !=
                file_hash(args.intervention30) or
            audit50['report_sha256'] != file_hash(args.intervention50) or
            thirty['first_divergence_review_sha256'] !=
                file_hash(args.first_divergence_review) or
            fifty['prior_intervention_sha256'] !=
                file_hash(args.intervention30) or
            audit30['summary']['policy_conditional_action_loss'] !=
                fifty['n_prior_losses'] or
            audit50['summary']['thirty_step_policy_conditional_losses'] !=
                fifty['n_prior_losses']):
        raise ValueError('Changed or incomplete intervention audit')
    fifty_by = {row['input_content_sha256']: row for row in fifty['rows']}
    rows = []
    for pair, old in zip(weak['pairs'], thirty['rows'], strict=True):
        if (old['input_content_sha256'] != pair['input_content_sha256'] or
                old['game'] != pair['game'] or
                old['turn'] != pair['turn'] or
                old['positive_action'] != pair['positive_action'] or
                old['negative_action'] != pair['negative_action'] or
                old['original_reward'] != 1.0):
            raise ValueError('Changed reviewed branch point')
        key = pair['input_content_sha256']
        at30 = 1.0-old['counterfactual_reward']
        at50 = (1.0-fifty_by[key]['counterfactual_50_step_reward']
                if key in fifty_by else 0.0)
        if (at30 not in (0., 1.) or at50 not in (0., 1.) or
                at50 > at30):
            raise ValueError('Counterfactual policy changed unexpectedly')
        content = {'first_divergence_input_sha256': key,
            'intervention30_sha256': file_hash(args.intervention30),
            'audit30_sha256': file_hash(args.audit30),
            'intervention50_sha256': file_hash(args.intervention50),
            'audit50_sha256': file_hash(args.audit50),
            'game_sha256': pair['game_sha256'],
            'own_episode_sha256': pair['own_episode_sha256'],
            'base_episode_sha256': pair['base_episode_sha256'],
            'target_action_input_sha256':
                pair['target_action_input_sha256'],
            'turn': pair['turn'],
            'observation': pair['observation'],
            'admissible_commands': pair['admissible_commands'],
            'positive_action': pair['positive_action'],
            'negative_action': pair['negative_action'],
            'policy_advantage30': at30,
            'policy_advantage50': at50}
        rows.append({'game': pair['game'], 'family': pair['family'],
            **content, 'input_content_sha256': digest(content)})
    return rows


def prepare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = expected(args)
    result = {'protocol': 'New content-bound official train action targets, policy-conditional reward difference from fixed old-LoRA continuation at 30 and 50 steps; not a universal action value or independent test',
        'first_divergence_review_sha256':
            file_hash(args.first_divergence_review),
        'intervention30_sha256': file_hash(args.intervention30),
        'audit30_sha256': file_hash(args.audit30),
        'intervention50_sha256': file_hash(args.intervention50),
        'audit50_sha256': file_hash(args.audit50),
        'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'targets': len(rows),
        'positive_at_30': sum(x['policy_advantage30'] for x in rows),
        'positive_at_50': sum(x['policy_advantage50'] for x in rows)}),
        flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--first-divergence-review', type=Path, default=Path('data/annotations/alf_first_divergence42_reviewed_20261007.json'))
    parser.add_argument('--intervention30', type=Path, default=Path('results/trajectory_hyperlora/alf_own_only_first_divergence42_intervention_v2_20261007.json'))
    parser.add_argument('--audit30', type=Path, default=Path('results/trajectory_hyperlora/alf_own_only_first_divergence42_intervention_v2_audited_20261007.json'))
    parser.add_argument('--intervention50', type=Path, default=Path('results/trajectory_hyperlora/alf_own_only_first_divergence_budget50_v3_20261007.json'))
    parser.add_argument('--audit50', type=Path, default=Path('results/trajectory_hyperlora/alf_own_only_first_divergence_budget50_v3_audited_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path('data/annotations/alf_own_only_causal_action42_reviewed_20261007.json'))
    prepare(parser.parse_args())
