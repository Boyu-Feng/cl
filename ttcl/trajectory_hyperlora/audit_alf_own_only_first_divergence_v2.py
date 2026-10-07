"""Independently replay action-intervention episodes and audit content lineage."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ttcl.experience_evolution.environment import make_env
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_same_game_retry import digest as episode_digest
from ttcl.trajectory_hyperlora.prepare_alf_first_divergence_pairs_v1 import build


def replay(game: Path, reference: dict, alternate: dict, pair: dict):
    env = make_env(game)
    try:
        state = env.reset()
        if str(state['feedback']) != reference['initial_observation']:
            raise ValueError('Changed counterfactual initial state')
        steps = alternate['trajectory']
        if (not steps or len(steps) != alternate['steps'] or
                alternate['status'] != 'complete' or
                alternate['invalid_commands'] != 0 or
                pair['turn'] >= len(steps)):
            raise ValueError('Incomplete counterfactual episode')
        for index, step in enumerate(steps):
            admissible = list(state['admissible_commands'])
            if (step['turn'] != index or
                    step['command'] not in admissible or
                    step['valid'] is not True or
                    index < pair['turn'] and
                        step != reference['trajectory'][index] or
                    index == pair['turn'] and
                        step['command'] != pair['negative_action']):
                raise ValueError('Changed shared prefix or intervention action')
            state, _, done = env.step(step['command'])
            if (str(state['feedback']) != step['observation'] or
                    bool(state['won']) != step['won'] or
                    done and index != len(steps)-1):
                raise ValueError('Counterfactual environment transition changed')
        if bool(state['won']) != bool(alternate['reward']):
            raise ValueError('Counterfactual terminal reward changed')
    finally:
        env.close()


def audit(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    run = json.loads(args.report.read_text())
    reviewed = json.loads(args.first_divergence_review.read_text())
    bank = json.loads(args.own_bank.read_text())
    original = json.loads(args.rollouts.read_text())
    pairs = build(bank, original, file_hash(args.own_bank),
                  file_hash(args.rollouts))
    if (reviewed['pairs'] != pairs or
            run['first_divergence_review_sha256'] !=
                file_hash(args.first_divergence_review) or
            run['own_bank_sha256'] != file_hash(args.own_bank) or
            run['rollouts_sha256'] != file_hash(args.rollouts) or
            run['checkpoint_sha256'] != original['checkpoint_sha256'] or
            run['failures'] or run['max_cases'] != 42 or
            len(run['rows']) != 42):
        raise ValueError('Changed or incomplete first-divergence experiment')
    by_game = {item['game']: item for item in original['games']}
    counts = Counter()
    changed = []
    for pair, row in zip(pairs, run['rows'], strict=True):
        reference = by_game[pair['game']]['arms']['own']
        if (row['game'] != pair['game'] or
                row['input_content_sha256'] !=
                    pair['input_content_sha256'] or
                row['turn'] != pair['turn'] or
                row['positive_action'] != pair['positive_action'] or
                row['negative_action'] != pair['negative_action'] or
                row['original_reward'] != 1.0 or
                row['reference_trajectory_sha256'] !=
                    episode_digest(reference['trajectory']) or
                row['counterfactual_reward'] !=
                    row['counterfactual']['reward'] or
                file_hash(args.data_root / pair['game']) !=
                    pair['game_sha256']):
            raise ValueError('Changed intervention provenance')
        replay(args.data_root / pair['game'], reference,
               row['counterfactual'], pair)
        counts[pair['family']] += 1
        if row['counterfactual_reward'] < row['original_reward']:
            changed.append(pair['input_content_sha256'])
    summary = {'pairs': len(pairs),
        'policy_conditional_action_loss': len(changed),
        'policy_conditional_action_no_loss': len(pairs)-len(changed),
        'by_family': dict(sorted(counts.items())),
        'causal_scope': 'single forced action under fixed successful old-LoRA policy, not cross-policy or general task effect'}
    result = {'protocol': 'Independent game-file and transition replay of all forty-two first-divergence action interventions',
        'intervention_report_sha256': file_hash(args.report),
        'review_sha256': file_hash(args.first_divergence_review),
        'checkpoint_sha256': run['checkpoint_sha256'],
        'policy_conditional_loss_bindings': changed,
        'summary': summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, default=Path('results/trajectory_hyperlora/alf_own_only_first_divergence42_intervention_v2_20261007.json'))
    parser.add_argument('--first-divergence-review', type=Path, default=Path('data/annotations/alf_first_divergence42_reviewed_20261007.json'))
    parser.add_argument('--own-bank', type=Path, default=Path('data/annotations/alf_own_only42_replay_reviewed_20261007.json'))
    parser.add_argument('--rollouts', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_train240_taskpair_current1000_20261006.json'))
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_own_only_first_divergence42_intervention_v2_audited_20261007.json'))
    audit(parser.parse_args())
