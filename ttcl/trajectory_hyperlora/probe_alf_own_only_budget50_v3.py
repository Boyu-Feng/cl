"""Recheck policy-conditional action losses with the 50-step ALFWorld budget.

Thirty-step terminal losses can be horizon artifacts. This keeps the same
success-conditioned frozen actor/source and the same single forced action,
then grants twenty extra continuation steps. The original 30-step successful
reference remains unchanged and is never used as a new training target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.experience_evolution.environment import ACTOR_SYSTEM, make_env
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_same_game_retry import digest as episode_digest
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import clean_command, generate, load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_source_fields, contextual_text_fields, task_context_text
from ttcl.trajectory_hyperlora.prepare_alf_onpolicy_supervision import checked_rollouts
from ttcl.trajectory_hyperlora.probe_alf_own_only_first_divergence_v2 import save


def intervene(agent, tokenizer, game: Path, fields: dict, reference: dict,
              split: int, alternate: str, device: str, source_tokens: int):
    env = make_env(game)
    trajectory = []
    try:
        state = env.reset()
        initial = str(state['feedback'])
        if initial != reference['initial_observation']:
            raise ValueError('Changed intervention reset')
        messages = [{'role': 'system', 'content': ACTOR_SYSTEM}]
        for turn in range(50):
            if agent.task_conditioned and (agent.task_context_scope == 'current' or turn == 0):
                target = contextual_text_fields(agent, tokenizer,
                    task_context_text(initial, str(state['feedback'])),
                    device, source_tokens,
                    pooling='both' if agent.task_pair_pooling == 'mean' else 'last')
                with torch.no_grad():
                    agent.set_source(fields, target_fields=target)
            available = list(state['admissible_commands'])
            messages.append({'role': 'user', 'content':
                str(state['feedback'])+'\nAvailable commands:\n'+
                '\n'.join(available)})
            if turn < split:
                original = reference['trajectory'][turn]
                response, command = original['response'], original['command']
                if command not in available:
                    raise ValueError('Counterfactual prefix became inadmissible')
            elif turn == split:
                response = command = alternate
                if command not in available:
                    raise ValueError('Counterfactual action became inadmissible')
            else:
                response = generate(agent, tokenizer, messages, device, 64,
                                    available)
                command = clean_command(response, available)
            state, _, done = env.step(command)
            messages.append({'role': 'assistant', 'content': response})
            step = {'turn': turn, 'response': response, 'command': command,
                'valid': command in available,
                'observation': str(state['feedback']),
                'won': bool(state['won'])}
            if turn < split and step != reference['trajectory'][turn]:
                raise ValueError('Counterfactual prefix transition changed')
            trajectory.append(step)
            if done or state['won']:
                break
        return {'status': 'complete', 'reward': float(bool(state['won'])),
            'steps': len(trajectory),
            'invalid_commands': sum(not x['valid'] for x in trajectory),
            'trajectory': trajectory}
    finally:
        env.close()
        agent.set_source(None)


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    prior = json.loads(args.prior_intervention.read_text())
    reviewed = json.loads(args.first_divergence_review.read_text())
    original, by_game = checked_rollouts(args)
    if (len(prior['rows']) != 42 or prior['failures'] or
            prior['max_cases'] != 42 or
            prior['first_divergence_review_sha256'] !=
                file_hash(args.first_divergence_review) or
            prior['rollouts_sha256'] != file_hash(args.rollouts) or
            original['checkpoint_sha256'] != file_hash(args.checkpoint) or
            [x['input_content_sha256'] for x in prior['rows']] !=
                [x['input_content_sha256'] for x in reviewed['pairs']]):
        raise ValueError('Changed 30-step intervention lineage')
    losses = [(row, pair) for row, pair in zip(prior['rows'],
        reviewed['pairs'], strict=True) if row['counterfactual_reward'] == 0.]
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    games = {x['game']: x for x in original['games']}
    result = {'protocol': 'Frozen success-conditioned old LoRA, same first forced base action and initial reset as reviewed 30-step intervention, 50-step continuation; tests budget sensitivity, no additional training',
        'prior_intervention_sha256': file_hash(args.prior_intervention),
        'first_divergence_review_sha256': file_hash(args.first_divergence_review),
        'rollouts_sha256': file_hash(args.rollouts),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'n_prior_losses': len(losses), 'rows': [], 'failures': []}
    save(args.output, result)
    for row, pair in losses:
        source, note = by_game[pair['game']]
        reference = games[pair['game']]['arms']['own']
        if (source['input_content_sha256'] !=
                games[pair['game']]['input_content_sha256'] or
                row['reference_trajectory_sha256'] !=
                    episode_digest(reference['trajectory']) or
                row['turn'] != pair['turn'] or
                row['negative_action'] != pair['negative_action']):
            raise ValueError('Changed frozen success-policy branch')
        fields = contextual_source_fields(agent, tokenizer,
            note['source_records'], args.device,
            args.contextual_source_max_tokens, pooling='both')
        try:
            with torch.no_grad():
                episode = intervene(agent, tokenizer,
                    args.data_root / pair['game'], fields, reference,
                    pair['turn'], pair['negative_action'], args.device,
                    args.contextual_source_max_tokens)
            if episode['invalid_commands'] != 0 or episode['steps'] > 50:
                raise ValueError('Invalid 50-step intervention')
            result['rows'].append({'game': pair['game'],
                'input_content_sha256': pair['input_content_sha256'],
                'prior_30_step_reward': row['counterfactual_reward'],
                'counterfactual_50_step_reward': episode['reward'],
                'episode': episode})
            save(args.output, result)
            print(json.dumps({'n': len(result['rows']), 'of': len(losses),
                'reward50': episode['reward'],
                'steps50': episode['steps']}), flush=True)
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
    parser.add_argument('--first-divergence-review', type=Path, default=Path('data/annotations/alf_first_divergence42_reviewed_20261007.json'))
    parser.add_argument('--prior-intervention', type=Path, default=Path('results/trajectory_hyperlora/alf_own_only_first_divergence42_intervention_v2_20261007.json'))
    parser.add_argument('--expected-games', type=int, default=240)
    parser.add_argument('--per-family-limit', type=int, default=40)
    parser.add_argument('--contextual-source-max-tokens', type=int, default=2048)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--gpu-fraction', type=float, default=.7)
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_own_only_first_divergence_budget50_v3_20261007.json'))
    run(parser.parse_args())
