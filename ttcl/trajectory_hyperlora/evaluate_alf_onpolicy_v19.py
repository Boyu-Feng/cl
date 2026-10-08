"""Run each frozen ALFWorld memory arm on its own from-empty history."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ttcl.experience_evolution.environment import make_env
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)
from ttcl.trajectory_hyperlora.alfworld_online_context_vector_mean_v1 import (
    update_fields, vector_hash,
)
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v2 import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.centered_feedback_residual_v3 import CenteredEvidenceResidual
from ttcl.trajectory_hyperlora.evaluate_alf_fresh_seen_online_v6 import (
    bounded_history_text, encode_source, target_feature,
)
from ttcl.trajectory_hyperlora.freeze_alf_onpolicy_v19 import METHODS, expected
from ttcl.trajectory_hyperlora.train_alf_feedback_residual_v4_antithetic import ResidualInjection


def checked(args):
    review = json.loads(args.review.read_text())
    if (review != expected(args) or args.method not in METHODS or
            not 0 <= args.order < 3):
        raise ValueError('Changed method-on-policy review or arm')
    targets = [row for row in review['targets']
               if row['order_id'] == args.order and
                  row['method'] == args.method]
    if len(targets) != 30 or [row['position'] for row in targets] != list(range(30)):
        raise ValueError('Changed official arm order')
    state = torch.load(args.residual, map_location='cpu', weights_only=True)
    if (state['model_kind'] != 'centered_bilinear_action_sensitive_v5' or
            tuple(state['basis'].shape) != (128, 8)):
        raise ValueError('Changed frozen LoRA correction')
    return targets, state


def run(args):
    targets, state = checked(args)
    if args.validate_only:
        print(json.dumps({'order': args.order, 'method': args.method,
            'targets': len(targets),
            'review_sha256': file_hash(args.review)}), flush=True)
        return
    output = args.output or Path(
        f'results/trajectory_hyperlora/alf_onpolicy_v19_order{args.order}_{args.method}_20261008.json')
    if output.exists():
        raise FileExistsError(output)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != 'contextual' or
            agent.task_context_scope != 'current' or
            agent.task_pair_pooling != 'mean'):
        raise ValueError('Changed frozen actor')
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    correction = CenteredEvidenceResidual.from_state(
        state['correction']).to(args.device)
    correction.load_state_dict(state['correction'])
    correction.eval()
    basis = state['basis'].to(args.device).float()
    accepted = []
    prior_success_hashes = []
    fields = None
    global_vector = delta_vector = None
    report = {'protocol': 'Frozen official valid_unseen method-on-policy arm: start empty, own official-won trajectories only, one 50-step constrained greedy attempt per task, two-turn history, loop guard2; previously exposed games, exploratory',
        'plan_sha256': file_hash(args.plan),
        'review_sha256': file_hash(args.review),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'residual_sha256': file_hash(args.residual),
        'order': args.order, 'method': args.method,
        'max_steps': 50, 'max_new_tokens': 64,
        'actor_history_turns': 2, 'loop_guard_max': 2,
        'source_token_limit': 2048,
        'rows': [], 'failures': []}
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, report)
    for binding in targets:
        game = args.data_root / binding['game']
        entry = {'position': binding['position'],
            'family': binding['family'], 'game': binding['game'],
            'game_sha256': binding['game_sha256'],
            'target_binding_sha256': binding['input_content_sha256'],
            'prior_success_records_sha256': prior_success_hashes.copy(),
            'prior_success_chain_sha256': digest(prior_success_hashes),
            'accepted_memory_count_before': len(accepted),
            'source_vector_sha256_before': vector_hash(fields)}
        entry['input_content_sha256'] = digest({
            'target_binding_sha256': binding['input_content_sha256'],
            'prior_success_records_sha256': prior_success_hashes,
            'accepted_memory_count': len(accepted),
            'source_vector_sha256': entry['source_vector_sha256_before']})
        try:
            if file_hash(game) != binding['game_sha256']:
                raise ValueError('Changed official target game')
            env = make_env(game)
            try:
                initial = str(env.reset()['feedback'])
            finally:
                env.close()
            memory_text = None
            entry['raw_text_original_tokens'] = 0
            entry['raw_text_retained_tokens'] = 0
            if args.method == 'own_success_raw_text' and accepted:
                memory_text, original, retained = bounded_history_text(
                    tokenizer, accepted, 2048)
                entry['raw_text_original_tokens'] = original
                entry['raw_text_retained_tokens'] = retained
            entry['raw_text'] = memory_text or ''
            entry['raw_text_sha256'] = digest(memory_text or '')
            entry['prompt_memory_sha256'] = hashlib.sha256(
                (memory_text or '').encode()).hexdigest()
            use_lora = (args.method in ('continuous_success_mean',
                        'fixed_first_success') and fields is not None)
            entry['adapter_mounted'] = use_lora
            if use_lora:
                q = target_feature(agent, tokenizer, initial, args.device)
                with torch.no_grad():
                    code = correction(global_vector, delta_vector, q)
                    shift = code @ basis.T
                entry['code_norm'] = float(code.norm())
                with ResidualInjection(agent, shift):
                    episode = run_episode(agent, tokenizer, game, fields,
                        adapter=True, device=args.device, max_steps=50,
                        max_new_tokens=64, constrain_actions=True,
                        actor_history_turns=2, loop_guard_max=2)
            else:
                entry['code_norm'] = 0.
                episode = run_episode(agent, tokenizer, game, {},
                    adapter=False, device=args.device, max_steps=50,
                    max_new_tokens=64, constrain_actions=True,
                    actor_history_turns=2, loop_guard_max=2,
                    memory_text=memory_text)
            if (episode['status'] != 'complete' or
                    episode['initial_observation'] != initial or
                    episode['invalid_commands'] != 0):
                raise ValueError('Incomplete or invalid official episode')
            entry['episode'] = episode
            entry['memory_write'] = False
            if episode['reward']:
                records = records_from_episode(episode)
                record_hash = digest(records)
                prior_success_hashes.append(record_hash)
                entry['own_success_records'] = records
                entry['own_success_records_sha256'] = record_hash
                if args.method == 'own_success_raw_text':
                    accepted.append({'records': records})
                    entry['memory_write'] = True
                elif args.method in ('continuous_success_mean',
                                     'fixed_first_success') and (
                        args.method == 'continuous_success_mean' or
                        not accepted):
                    fresh, fresh_global, fresh_delta, original, retained = (
                        encode_source(agent, tokenizer, records, args.device))
                    count = len(accepted)
                    fields = update_fields(fields, fresh, count)
                    if count:
                        global_vector += (fresh_global-global_vector)/(count+1)
                        delta_vector += (fresh_delta-delta_vector)/(count+1)
                    else:
                        global_vector = fresh_global.clone()
                        delta_vector = fresh_delta.clone()
                    accepted.append({'records': records})
                    entry['source_tokens_original'] = original
                    entry['source_tokens_retained'] = retained
                    entry['memory_write'] = True
            entry['source_vector_sha256_after'] = vector_hash(fields)
            entry['accepted_memory_count_after'] = len(accepted)
            entry['prior_success_records_sha256_after'] = prior_success_hashes.copy()
            report['rows'].append(entry)
            save(output, report)
            print(json.dumps({'order': args.order,
                'method': args.method, 'done': len(report['rows']),
                'success': sum(row['episode']['reward'] for row in report['rows']),
                'writes': sum(row['memory_write'] for row in report['rows'])}),
                flush=True)
        except Exception as exc:
            report['failures'].append({'position': binding['position'],
                'reason': repr(exc), 'partial': entry})
            save(output, report)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path(
        'current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--residual', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_20261007.pt'))
    parser.add_argument('--plan', type=Path, default=Path(
        'ttcl/trajectory_hyperlora/alf_valid_unseen_multiorder_v10_plan.json'))
    parser.add_argument('--review', type=Path, default=Path(
        'data/annotations/alf_onpolicy_multiorder_v19_reviewed_20261008.json'))
    parser.add_argument('--data-root', type=Path, default=Path(
        'ttcl/data/alfworld_delta'))
    parser.add_argument('--order', type=int, required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.7)
    run(parser.parse_args())
