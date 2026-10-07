"""Counterfactually test reviewed own-success/base-failure first action splits.

The successful own-source policy and environment reset remain fixed. Its
recorded trajectory is regenerated before forcing the base branch's first
alternative action and continuing with that same policy. A terminal-reward
change supports a policy-conditional action effect, not a universal label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest as content_digest, file_hash,
)
from ttcl.trajectory_hyperlora.alfworld_same_game_retry import digest as episode_digest
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_source_fields
from ttcl.trajectory_hyperlora.prepare_alf_first_divergence_pairs_v1 import build
from ttcl.trajectory_hyperlora.prepare_alf_onpolicy_supervision import checked_rollouts
from ttcl.trajectory_hyperlora.probe_alf_first_divergence import swapped_episode


def save(path: Path, report: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    reviewed = json.loads(args.first_divergence_review.read_text())
    own_bank = json.loads(args.own_bank.read_text())
    report, by_game = checked_rollouts(args)
    if (reviewed['own_bank_sha256'] != file_hash(args.own_bank) or
            reviewed['report_sha256'] != file_hash(args.rollouts) or
            reviewed['pairs'] != build(own_bank, report,
                file_hash(args.own_bank), file_hash(args.rollouts)) or
            len(reviewed['pairs']) != 42 or
            report['checkpoint_sha256'] != file_hash(args.checkpoint) or
            report['summary']['n'] != 240 or
            report['memory_text'] or
            report.get('actor_history_turns', 0) != 0):
        raise ValueError('Changed intervention labels, policy, or actor protocol')
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != 'contextual' or
            not agent.task_conditioned or
            agent.task_pair_pooling != 'mean'):
        raise ValueError('Expected frozen contextual task-conditioned actor')
    games = {game['game']: game for game in report['games']}
    result = {'protocol': 'Official train first-action counterfactual: replay exact successful old-LoRA actor, force paired failed-base action at shared prefix, continue identical frozen old-LoRA policy; one reset per arm, 30-step/64-token/constraint actor; policy-conditional effect, no general causal label',
        'first_divergence_review_sha256': file_hash(args.first_divergence_review),
        'own_bank_sha256': file_hash(args.own_bank),
        'rollouts_sha256': file_hash(args.rollouts),
        'source_review_sha256': file_hash(args.source_review),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'max_cases': args.max_cases, 'rows': [], 'failures': []}
    selected = reviewed['pairs'][:args.max_cases]
    save(args.output, result)
    for pair in selected:
        game = games[pair['game']]
        original = game['arms']['own']
        source_row, source_note = by_game[pair['game']]
        if (source_row['input_content_sha256'] != game['input_content_sha256'] or
                source_row['source_game'] != game['source_game'] or
                content_digest(game['arms']['base']) !=
                    pair['base_episode_sha256']):
            raise ValueError('Changed own-source or base branch')
        if file_hash(args.data_root / pair['game']) != pair['game_sha256']:
            raise ValueError('Changed original game')
        fields = contextual_source_fields(agent, tokenizer,
            source_note['source_records'], args.device,
            args.contextual_source_max_tokens, pooling='both')
        reference = run_episode(agent, tokenizer,
            args.data_root / pair['game'], fields, adapter=True,
            device=args.device, max_steps=30, max_new_tokens=64,
            constrain_actions=True)
        if (reference['status'] != 'complete' or
                reference['reward'] != original['reward'] or
                reference['steps'] != original['steps'] or
                episode_digest(reference['trajectory']) !=
                    episode_digest(original['trajectory'])):
            result['failures'].append({'game': pair['game'],
                'reason': 'successful frozen actor not reproduced',
                'reference': reference})
            save(args.output, result)
            raise ValueError('Own-source frozen policy changed')
        try:
            with torch.no_grad():
                alternate = swapped_episode(agent, tokenizer,
                    args.data_root / pair['game'], fields, original,
                    pair['turn'], pair['negative_action'],
                    args.device, args.contextual_source_max_tokens)
            if (alternate['status'] != 'complete' or
                    alternate['invalid_commands'] != 0 or
                    alternate['steps'] > 30):
                raise ValueError('Changed intervention actor budget')
            row = {'game': pair['game'], 'family': pair['family'],
                'input_content_sha256': pair['input_content_sha256'],
                'turn': pair['turn'],
                'positive_action': pair['positive_action'],
                'negative_action': pair['negative_action'],
                'original_reward': original['reward'],
                'counterfactual_reward': alternate['reward'],
                'reference_trajectory_sha256':
                    episode_digest(reference['trajectory']),
                'counterfactual': alternate}
            result['rows'].append(row)
            save(args.output, result)
            print(json.dumps({'n': len(result['rows']),
                'family': pair['family'], 'turn': pair['turn'],
                'positive_reward': original['reward'],
                'negative_action_reward': alternate['reward']}), flush=True)
        except Exception as exc:
            result['failures'].append({'game': pair['game'],
                'reason': repr(exc)})
            save(args.output, result)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_taskpair_current1000_20261006.pt'))
    parser.add_argument('--rollouts', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_train240_taskpair_current1000_20261006.json'))
    parser.add_argument('--candidates', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json'))
    parser.add_argument('--source-review', type=Path, default=Path('data/annotations/alf_sibling_train600_reviewed_20261006.json'))
    parser.add_argument('--retry-candidates', type=Path, default=Path('results/trajectory_hyperlora/alf_same_game_retry_candidates_20261006.json'))
    parser.add_argument('--retry-review', type=Path, default=Path('data/annotations/alf_same_game_retry_reviewed_20261006.json'))
    parser.add_argument('--all-source-review', type=Path, default=Path('data/annotations/alf_expert_first_attempts_reviewed_20261006.json'))
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--own-bank', type=Path, default=Path('data/annotations/alf_own_only42_replay_reviewed_20261007.json'))
    parser.add_argument('--first-divergence-review', type=Path, default=Path('data/annotations/alf_first_divergence42_reviewed_20261007.json'))
    parser.add_argument('--expected-games', type=int, default=240)
    parser.add_argument('--per-family-limit', type=int, default=40)
    parser.add_argument('--contextual-source-max-tokens', type=int, default=2048)
    parser.add_argument('--max-cases', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.7)
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_own_only_first_divergence42_intervention_v2_20261007.json'))
    args = parser.parse_args()
    if not 1 <= args.max_cases <= 42:
        parser.error('Intervention budget must be 1..42 reviewed cases')
    run(args)
