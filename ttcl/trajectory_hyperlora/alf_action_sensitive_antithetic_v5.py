"""Compare equal-norm random and actor-sensitive LoRA exploration.

This is a frozen-generator, official ALFWorld train-domain mechanism probe.
Every new rollout gets a fresh source-target-perturbation content binding.
The score direction uses two commands already executed by reviewed actors
from the same reset, without reading their rewards or any family slot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from ttcl.experience_evolution.environment import ACTOR_SYSTEM, make_env
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import bounded_source_text
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v2 import checked, save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields, task_context_text
from ttcl.trajectory_hyperlora.train_alf_own_only_soft_distill_v2 import action_logits, prompt_ids
from ttcl.trajectory_hyperlora.train_alf_feedback_residual_v4_antithetic import ResidualInjection


def inputs(args):
    sources, targets, bindings = checked(args)
    matrix = json.loads(args.matrix.read_text())
    audit = json.loads(args.matrix_audit.read_text())
    prior = json.loads(args.prior_train.read_text())
    prior_audit = json.loads(args.prior_audit.read_text())
    saved = torch.load(args.prior_model, map_location='cpu', weights_only=True)
    if (matrix['review_sha256'] != file_hash(args.review) or
            audit['review_sha256'] != file_hash(args.review) or
            audit['raw_report_sha256'] != file_hash(args.matrix) or
            matrix['checkpoint_sha256'] != file_hash(args.checkpoint) or
            matrix['failures'] or len(matrix['base']) != 18 or
            len(matrix['pairs']) != 144 or
            prior_audit['report_sha256'] != file_hash(args.prior_train) or
            prior_audit['model_sha256'] != file_hash(args.prior_model) or
            prior['review_sha256'] != file_hash(args.review) or
            prior['checkpoint_sha256'] != file_hash(args.checkpoint) or
            prior['failures'] or len(prior['pair_schedule']) != 72 or
            saved['model_kind'] != 'centered_bilinear_antithetic' or
            saved['basis'].shape != (128, 8) or
            args.sigma != .5 or args.seed != 20261007):
        raise ValueError('Changed frozen pilot ancestry or budget')
    arms = {}
    commands = defaultdict(dict)
    for row, binding in zip(matrix['pairs'], bindings, strict=True):
        key = row['source_id'], row['target_id']
        if (row['input_content_sha256'] != binding['input_content_sha256'] or
                row['episode']['status'] != 'complete' or
                row['episode']['steps'] < 1 or key in arms):
            raise ValueError('Changed source-target arm or initial action')
        arms[key] = row
        commands[row['target_id']][row['source_id']] = row['episode']['trajectory'][0]['command']
    if len(arms) != 144:
        raise ValueError('Incomplete frozen source-target arms')
    return sources, targets, matrix, prior, saved, arms, commands


def expected(args):
    _, targets, matrix, prior, _, arms, commands = inputs(args)
    rows = []
    for scheduled in prior['pair_schedule']:
        si, ti = scheduled['source_id'], scheduled['target_id']
        first = commands[ti][si]
        alternative = next((commands[ti][j] for j in sorted(commands[ti])
                            if commands[ti][j] != first), None)
        if alternative is None:
            continue
        target = targets[ti]
        episode = arms[si, ti]['episode']
        env = make_env(args.data_root / target['game'])
        try:
            state = env.reset()
            available = list(state['admissible_commands'])
            if (str(state['feedback']) != episode['initial_observation'] or
                    hashlib.sha256(json.dumps(available).encode()).hexdigest() !=
                    episode['initial_commands_sha256'] or
                    first not in available or alternative not in available):
                raise ValueError('Changed paired first-decision state')
        finally:
            env.close()
        content = {'source_id': si, 'target_id': ti,
            'target_game_sha256': target['game_sha256'],
            'source_target_binding': arms[si, ti]['input_content_sha256'],
            'reference_episode_sha256': digest(episode),
            'initial_observation_sha256': digest(episode['initial_observation']),
            'initial_commands_sha256': episode['initial_commands_sha256'],
            'action_one': first, 'action_two': alternative,
            'matrix_sha256': file_hash(args.matrix),
            'prior_model_sha256': file_hash(args.prior_model),
            'sigma': args.sigma, 'seed': args.seed,
            'max_steps': 50, 'max_new_tokens': 64,
            'actor_history_turns': 2, 'loop_guard_max': 2}
        rows.append({'game': target['game'], **content,
                     'input_content_sha256': digest(content)})
        if len(rows) == args.max_pairs:
            break
    if len(rows) != args.max_pairs or args.max_pairs > 50:
        raise ValueError('Not enough reviewed pairs with two first actions')
    return rows


def prepare(args):
    if args.pilot_review.exists():
        raise FileExistsError(args.pilot_review)
    rows = expected(args)
    review = {'protocol': 'Twenty-four preordered reviewed official train source-target pairs with two different observed admissible first commands; no new-perturbation reward read during selection; four equal-norm rollout arms per pair',
        'matrix_sha256': file_hash(args.matrix),
        'matrix_audit_sha256': file_hash(args.matrix_audit),
        'prior_train_sha256': file_hash(args.prior_train),
        'prior_audit_sha256': file_hash(args.prior_audit),
        'prior_model_sha256': file_hash(args.prior_model),
        'review_sha256': file_hash(args.review),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'sigma': args.sigma, 'seed': args.seed,
        'rows': rows}
    args.pilot_review.parent.mkdir(parents=True, exist_ok=True)
    args.pilot_review.write_text(json.dumps(review, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'reviewed_pairs': len(rows),
        'different_targets': len({row['target_id'] for row in rows})}), flush=True)


def score(agent, tokenizer, prefix, action, device):
    answer = tokenizer(action + tokenizer.eos_token,
                       add_special_tokens=False).input_ids
    logits = action_logits(agent, prefix, answer, device)
    answer_ids = torch.tensor(answer, device=device)[:, None]
    return F.log_softmax(logits, -1).gather(-1, answer_ids).mean()


def sensitive_direction(agent, tokenizer, fields, initial, available,
                        actions, basis, device):
    with torch.no_grad():
        query = contextual_text_fields(agent, tokenizer,
            task_context_text(initial, initial), device, 2048,
            pooling='both')
        agent.set_source(fields, target_fields=query)
    code = torch.zeros(8, device=device, requires_grad=True)
    prior = [adapter.b for adapter in agent.adapters]
    try:
        shift = code @ basis.T
        for adapter, head, old in zip(agent.adapters, agent.b_heads,
                                      prior, strict=True):
            adapter.b = old + F.linear(shift.to(head.weight.dtype),
                                     head.weight).reshape_as(old)
        messages = [{'role': 'system', 'content': ACTOR_SYSTEM},
                    {'role': 'user', 'content': initial+
                     '\nAvailable commands:\n'+'\n'.join(available)}]
        prefix = prompt_ids(tokenizer, messages, 4096)
        margin = (score(agent, tokenizer, prefix, actions[0], device)-
                  score(agent, tokenizer, prefix, actions[1], device))
        gradient, = torch.autograd.grad(margin, code)
        gradient = gradient.detach().float()
        if not torch.isfinite(gradient).all() or gradient.norm() < 1e-8:
            raise ValueError('No finite first-decision latent sensitivity')
        return gradient, float(margin.detach())
    finally:
        agent.set_source(None)


def collect(args):
    sources, targets, _, _, saved, _, _ = inputs(args)
    review = json.loads(args.pilot_review.read_text())
    expected_rows = expected(args)
    if (review['rows'] != expected_rows or
            review['matrix_sha256'] != file_hash(args.matrix) or
            review['matrix_audit_sha256'] != file_hash(args.matrix_audit) or
            review['prior_train_sha256'] != file_hash(args.prior_train) or
            review['prior_audit_sha256'] != file_hash(args.prior_audit) or
            review['prior_model_sha256'] != file_hash(args.prior_model) or
            review['review_sha256'] != file_hash(args.review) or
            review['checkpoint_sha256'] != file_hash(args.checkpoint) or
            review['sigma'] != args.sigma or review['seed'] != args.seed):
        raise ValueError('Changed reviewed perturbation input')
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        result = json.loads(args.output.read_text())
        if (result['pilot_review_sha256'] != file_hash(args.pilot_review) or
                result['checkpoint_sha256'] != file_hash(args.checkpoint) or
                result['prior_model_sha256'] != file_hash(args.prior_model) or
                result['failures'] or len(result['rows']) > len(expected_rows) or
                any(row['input_content_sha256'] !=
                    expected_rows[i]['input_content_sha256']
                    for i, row in enumerate(result['rows']))):
            raise ValueError('Changed or failed saved pilot prefix')
    else:
        result = {'protocol': 'Frozen official train equal-norm four-arm antithetic LoRA exploration: isotropic versus actor action-sensitive latent direction; no generator update or validation claim',
            'pilot_review_sha256': file_hash(args.pilot_review),
            'checkpoint_sha256': file_hash(args.checkpoint),
            'prior_model_sha256': file_hash(args.prior_model),
            'sigma': args.sigma, 'seed': args.seed,
            'max_steps': 50, 'max_new_tokens': 64,
            'actor_history_turns': 2, 'loop_guard_max': 2,
            'rows': [], 'failures': []}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save(args.output, result)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != 'contextual' or
            agent.task_context_scope != 'current' or
            agent.task_pair_pooling != 'mean'):
        raise ValueError('Changed frozen actor')
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    basis = saved['basis'].to(args.device).float()
    fields_cache = {}
    for index in range(len(result['rows']), len(expected_rows)):
        row = expected_rows[index]
        si, ti = row['source_id'], row['target_id']
        if si not in fields_cache:
            bounded, original, retained = bounded_source_text(
                tokenizer, sources[si]['records'], 2048)
            with torch.no_grad():
                fields_cache[si] = (contextual_text_fields(agent, tokenizer,
                    bounded, args.device, 2048, pooling='both'),
                    original, retained)
        fields, original, retained = fields_cache[si]
        game = args.data_root / targets[ti]['game']
        env = make_env(game)
        try:
            state = env.reset()
            initial = str(state['feedback'])
            available = list(state['admissible_commands'])
        finally:
            env.close()
        try:
            gradient, margin = sensitive_direction(agent, tokenizer,
                fields, initial, available,
                (row['action_one'], row['action_two']), basis, args.device)
            direction = gradient/gradient.norm()
            generator = torch.Generator(device='cpu').manual_seed(args.seed+index)
            noise = torch.randn(8, generator=generator).to(args.device)
            random_direction = noise/noise.norm()
            scale = args.sigma*math.sqrt(8)
            perturbations = {'random': scale*random_direction,
                             'sensitive': scale*direction}
            episodes = {}
            for method, delta in perturbations.items():
                for sign in (1, -1):
                    with ResidualInjection(agent, sign*(delta @ basis.T)):
                        with torch.no_grad():
                            episode = run_episode(agent, tokenizer, game,
                                fields, adapter=True, device=args.device,
                                max_steps=50, max_new_tokens=64,
                                constrain_actions=True,
                                actor_history_turns=2, loop_guard_max=2)
                    if (episode['status'] != 'complete' or
                            episode['initial_observation'] != initial or
                            episode['invalid_commands'] != 0):
                        raise ValueError('Changed or incomplete pilot rollout')
                    episodes[f'{method}_{"positive" if sign == 1 else "negative"}'] = episode
            result['rows'].append({'input_content_sha256':
                row['input_content_sha256'],
                'source_id': si, 'target_id': ti,
                'source_tokens_original': original,
                'source_tokens_retained': retained,
                'gradient': gradient.cpu().tolist(),
                'initial_action_margin': margin,
                'random_delta': perturbations['random'].cpu().tolist(),
                'sensitive_delta': perturbations['sensitive'].cpu().tolist(),
                'episodes': episodes})
            save(args.output, result)
            print(json.dumps({'done': len(result['rows']),
                'target_id': ti,
                'random': [episodes[f'random_{x}']['reward']
                           for x in ('positive', 'negative')],
                'sensitive': [episodes[f'sensitive_{x}']['reward']
                              for x in ('positive', 'negative')],
                'failures': 0}), flush=True)
        except Exception as exc:
            result['failures'].append({'index': index,
                'input_content_sha256': row['input_content_sha256'],
                'reason': repr(exc)})
            save(args.output, result)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'collect'))
    parser.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--source-report-seq0', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq0_6_20261007.json'))
    parser.add_argument('--source-report-seq6', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq6_6_20261007.json'))
    parser.add_argument('--plan', type=Path, default=Path('ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json'))
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--target-policy', choices=('train18',), default='train18')
    parser.add_argument('--checkpoint-training-candidates', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json'))
    parser.add_argument('--warmstart-candidates', type=Path, default=Path('results/trajectory_hyperlora/alf_next_task_candidates_20261005.json'))
    parser.add_argument('--prior-train-review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--matrix', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_20261007.json'))
    parser.add_argument('--matrix-audit', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_audited_20261007.json'))
    parser.add_argument('--prior-train', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_antithetic_v4_seed42_20261007.json'))
    parser.add_argument('--prior-audit', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_antithetic_v4_seed42_audited_20261007.json'))
    parser.add_argument('--prior-model', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_antithetic_v4_seed42_20261007.pt'))
    parser.add_argument('--pilot-review', type=Path, default=Path('data/annotations/alf_action_sensitive_antithetic24_reviewed_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_action_sensitive_antithetic24_v5_20261007.json'))
    parser.add_argument('--max-pairs', type=int, default=24)
    parser.add_argument('--sigma', type=float, default=.5)
    parser.add_argument('--seed', type=int, default=20261007)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.7)
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else collect)(args)
