"""Train-only first-action interventions for audited incremental LoRA pairs.

For each terminal-discordant freeze/update pair, reproduce both original
episodes, then replace the first different command under each *fixed* arm's
policy. The arm's own response prefix, LoRA and continuation policy stay fixed.
This estimates a local action intervention, not the whole LoRA update effect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ttcl.experience_evolution.environment import ACTOR_SYSTEM, make_env
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v2 import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import (
    clean_command, generate, guarded_commands, load_agent, run_episode,
)
from ttcl.trajectory_hyperlora.alfworld_online_context_vector_mean_v1 import update_fields
from ttcl.trajectory_hyperlora.centered_feedback_residual_v3 import CenteredEvidenceResidual
from ttcl.trajectory_hyperlora.collect_alf_incremental_update_v9 import checked
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields, task_context_text
from ttcl.trajectory_hyperlora.evaluate_alf_fresh_seen_online_v6 import encode_source, target_feature
from ttcl.trajectory_hyperlora.train_alf_feedback_residual_v4_antithetic import ResidualInjection
from ttcl.trajectory_hyperlora.train_alf_next_task import compact_messages


def candidates(args):
    sources, targets, model, reviewed = checked(args)
    report = json.loads(args.output.read_text())
    audit = json.loads(args.audit_output.read_text())
    if (audit['raw_report_sha256'] != file_hash(args.output) or
            audit['summary']['failed_replays'] != 0 or
            report['review_sha256'] != file_hash(args.update_review) or
            report['checkpoint_sha256'] != file_hash(args.checkpoint) or
            report['residual_sha256'] != file_hash(args.residual) or
            len(report['rows']) != 108 or len(reviewed) != 108 or
            report['failures'] or report['max_steps'] != 50 or
            report['actor_history_turns'] != 2 or report['loop_guard_max'] != 2):
        raise ValueError('Changed audited v9 rollout')
    rows = []
    for index, (target, pair) in enumerate(zip(reviewed, report['rows'], strict=True)):
        if pair['input_content_sha256'] != target['input_content_sha256']:
            raise ValueError('Changed reviewed target binding')
        freeze, update = pair['freeze'], pair['update']
        if freeze['reward'] == update['reward']:
            continue
        split = next((j for j, (a, b) in enumerate(zip(
            freeze['trajectory'], update['trajectory'], strict=False))
            if a['command'] != b['command']), None)
        if split is None:
            raise ValueError('Discordant reward without a shared action split')
        if (any(freeze['trajectory'][j]['observation'] !=
                update['trajectory'][j]['observation'] for j in range(split)) or
                freeze['initial_observation'] != update['initial_observation']):
            raise ValueError('First action split has different environment state')
        content = {'method': 'incremental_first_action_v12',
            'parent_report_sha256': file_hash(args.output),
            'parent_audit_sha256': file_hash(args.audit_output),
            'parent_input_content_sha256': target['input_content_sha256'],
            'target_game_sha256': target['target_game_sha256'],
            'first_records_sha256': target['first_records_sha256'],
            'next_records_sha256': target['next_records_sha256'],
            'checkpoint_sha256': file_hash(args.checkpoint),
            'residual_sha256': file_hash(args.residual),
            'pair_index': index, 'split': split,
            'freeze_trajectory_sha256': digest(freeze['trajectory']),
            'update_trajectory_sha256': digest(update['trajectory']),
            'freeze_action': freeze['trajectory'][split]['command'],
            'update_action': update['trajectory'][split]['command'],
            'max_steps': 50, 'history_turns': 2, 'loop_guard_max': 2}
        rows.append({**content, 'input_content_sha256': digest(content)})
    if len(rows) != 16:
        raise ValueError('Changed count of terminal-discordant pairs')
    return sources, targets, model, report, rows


def prepare(args):
    if args.intervention_review.exists():
        raise FileExistsError(args.intervention_review)
    _, _, _, _, rows = candidates(args)
    value = {'protocol': 'Reviewed train-domain action intervention targets, content-bound to audited v9 pair and exact first different command; no intervention outcomes used in selection',
        'parent_report_sha256': file_hash(args.output),
        'parent_audit_sha256': file_hash(args.audit_output),
        'targets': rows}
    args.intervention_review.parent.mkdir(parents=True, exist_ok=True)
    save(args.intervention_review, value)
    print(json.dumps({'targets': len(rows),
                      'review_sha256': file_hash(args.intervention_review)}))


def intervene(agent, tokenizer, game, fields, reference, split, alternate, device):
    env = make_env(game)
    trajectory = []
    seen = {}
    try:
        state = env.reset()
        initial = str(state['feedback'])
        initial_commands = list(state['admissible_commands'])
        if initial != reference['initial_observation']:
            raise ValueError('Changed initial state')
        messages = [{'role': 'system', 'content': ACTOR_SYSTEM}]
        for turn in range(50):
            current_target = contextual_text_fields(agent, tokenizer,
                task_context_text(initial, str(state['feedback'])), device,
                agent.contextual_source_max_tokens, pooling='both')
            with torch.no_grad():
                agent.set_source(fields, target_fields=current_target)
            available = list(state['admissible_commands'])
            feedback = str(state['feedback'])
            allowed = guarded_commands(feedback, available, seen, 2)
            messages.append({'role': 'user', 'content': feedback +
                '\nAvailable commands:\n' + '\n'.join(available)})
            if turn < split:
                original = reference['trajectory'][turn]
                response, command = original['response'], original['command']
                if command not in available:
                    raise ValueError('Reference command inadmissible at prefix')
            elif turn == split:
                response = command = alternate
                if command not in available:
                    raise ValueError('Alternate action inadmissible at split')
            else:
                response = generate(agent, tokenizer,
                    compact_messages(messages, 2), device, 64, allowed)
                command = clean_command(response, available)
            valid = command in available
            key = (feedback, command)
            seen[key] = seen.get(key, 0) + 1
            state, _, done = env.step(command)
            messages.append({'role': 'assistant', 'content': response})
            step = {'turn': turn, 'response': response, 'command': command,
                'valid': valid, 'observation': str(state['feedback']),
                'won': bool(state['won'])}
            if turn < split and step != reference['trajectory'][turn]:
                raise ValueError('Shared-prefix transition changed')
            trajectory.append(step)
            if done or state['won']:
                break
        return {'status': 'complete', 'reward': float(bool(state['won'])),
            'steps': len(trajectory), 'initial_observation': initial,
            'memory_sha256': hashlib.sha256(b'').hexdigest(),
            'initial_commands_sha256': hashlib.sha256(json.dumps(
                initial_commands).encode()).hexdigest(),
            'invalid_commands': sum(not x['valid'] for x in trajectory),
            'trajectory': trajectory,
            'termination': 'success' if state['won'] else 'budget_or_environment_done'}
    finally:
        env.close()
        agent.set_source(None)


def evaluate(args):
    sources, _, model, report, rows = candidates(args)
    review = json.loads(args.intervention_review.read_text())
    if (review['targets'] != rows or
            review['parent_report_sha256'] != file_hash(args.output) or
            review['parent_audit_sha256'] != file_hash(args.audit_output)):
        raise ValueError('Changed action-intervention review')
    if args.intervention_output.exists():
        if not args.resume:
            raise FileExistsError(args.intervention_output)
        result = json.loads(args.intervention_output.read_text())
        if (result['review_sha256'] != file_hash(args.intervention_review) or
                result['failures'] or len(result['rows']) > len(rows) or
                any(x['input_content_sha256'] != y['input_content_sha256']
                    for x, y in zip(result['rows'], rows))):
            raise ValueError('Cannot resume altered intervention')
    else:
        result = {'protocol': 'Audited train-domain first-action swap under fixed freeze/update LoRA; 50 steps, greedy constrained actor, two-turn history and loop guard; descriptive local action effect only',
            'review_sha256': file_hash(args.intervention_review),
            'checkpoint_sha256': file_hash(args.checkpoint),
            'residual_sha256': file_hash(args.residual),
            'rows': [], 'failures': []}
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != 'contextual' or
            agent.task_context_scope != 'current' or
            agent.task_pair_pooling != 'mean'):
        raise ValueError('Changed frozen actor')
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    correction = CenteredEvidenceResidual.from_state(model['correction']).to(args.device)
    correction.load_state_dict(model['correction'])
    correction.eval()
    basis = model['basis'].to(args.device).float()
    cache = {}
    args.intervention_output.parent.mkdir(parents=True, exist_ok=True)
    save(args.intervention_output, result)
    for position in range(len(result['rows']), len(rows)):
        item = rows[position]
        pair = report['rows'][item['pair_index']]
        out = {'input_content_sha256': item['input_content_sha256'],
            'pair_index': item['pair_index'], 'split': item['split'],
            'arms': {}}
        try:
            for source_id in (pair['first_source_id'], pair['next_source_id']):
                if source_id not in cache:
                    cache[source_id] = encode_source(agent, tokenizer,
                        sources[source_id]['records'], args.device)
            first_f, first_g, first_d, _, _ = cache[pair['first_source_id']]
            next_f, next_g, next_d, _, _ = cache[pair['next_source_id']]
            configurations = (
                ('freeze', first_f, first_g, first_d, item['update_action']),
                ('update', update_fields(first_f, next_f, 1),
                 (first_g + next_g) / 2, (first_d + next_d) / 2,
                 item['freeze_action']))
            game = args.data_root / pair['game']
            if file_hash(game) != item['target_game_sha256']:
                raise ValueError('Changed official game')
            env = make_env(game)
            try:
                initial = str(env.reset()['feedback'])
            finally:
                env.close()
            q = target_feature(agent, tokenizer, initial, args.device)
            for arm, fields, global_v, delta_v, alternate in configurations:
                with torch.no_grad():
                    shift = correction(global_v, delta_v, q) @ basis.T
                with ResidualInjection(agent, shift):
                    reproduced = run_episode(agent, tokenizer, game, fields,
                        adapter=True, device=args.device, max_steps=50,
                        max_new_tokens=64, constrain_actions=True,
                        actor_history_turns=2, loop_guard_max=2)
                    reference = pair[arm]
                    if (reproduced['status'] != 'complete' or
                            reproduced['trajectory'] != reference['trajectory'] or
                            reproduced['reward'] != reference['reward']):
                        raise ValueError(f'{arm} reference did not reproduce')
                    changed = intervene(agent, tokenizer, game, fields,
                        reference, item['split'], alternate, args.device)
                if changed['invalid_commands'] or changed['status'] != 'complete':
                    raise ValueError(f'{arm} intervention invalid')
                out['arms'][arm] = {'original_reward': reference['reward'],
                    'swapped_reward': changed['reward'],
                    'alternate_action': alternate,
                    'swapped_episode': changed}
            result['rows'].append(out)
            save(args.intervention_output, result)
            print(json.dumps({'done': len(result['rows']),
                'pair_index': item['pair_index'],
                'freeze': [out['arms']['freeze']['original_reward'],
                           out['arms']['freeze']['swapped_reward']],
                'update': [out['arms']['update']['original_reward'],
                           out['arms']['update']['swapped_reward']]}), flush=True)
        except Exception as exc:
            result['failures'].append({'position': position,
                'reason': repr(exc), 'partial': out})
            save(args.intervention_output, result)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'evaluate'))
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
    parser.add_argument('--residual', type=Path, default=Path('results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_20261007.pt'))
    parser.add_argument('--training-report', type=Path, default=Path('results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_20261007.json'))
    parser.add_argument('--training-audit', type=Path, default=Path('results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_audited_20261007.json'))
    parser.add_argument('--update-review', type=Path, default=Path('data/annotations/alf_incremental_update108_v9_reviewed_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_incremental_update108_v9_20261007.json'))
    parser.add_argument('--audit-output', type=Path, default=Path('results/trajectory_hyperlora/alf_incremental_update108_v9_audited_20261007.json'))
    parser.add_argument('--intervention-review', type=Path, default=Path('data/annotations/alf_incremental_action16_v12_reviewed_20261007.json'))
    parser.add_argument('--intervention-output', type=Path, default=Path('results/trajectory_hyperlora/alf_incremental_action16_v12_20261007.json'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.7)
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else evaluate)(args)


if __name__ == '__main__':
    main()
