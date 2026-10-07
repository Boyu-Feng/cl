"""Replay one ALFWorld decision state under three online LoRA memories.

The state, actor prompt, admissible commands and checkpoint are identical;
only the prior self-trajectory source vector changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from ttcl.experience_evolution.environment import ACTOR_SYSTEM, clean_command, make_env
from ttcl.trajectory_hyperlora.alfworld_online_context_vector_mean_v1 import update_fields, vector_hash
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash, records_from_episode
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import bounded_source_text
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import guarded_commands, generate, load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields, task_context_text
from ttcl.trajectory_hyperlora.train_alf_next_task import compact_messages


def reconstruct(agent, tokenizer, run, index, policy, args, verify_state=True):
    fields = None
    successes = 0
    for row in run['games'][:index]:
        episode = row['episode']
        records = records_from_episode(episode)
        if row['new_records_sha256'] != digest(records):
            raise ValueError('Changed own historical trajectory')
        if episode['reward'] and (policy != 'first' or fields is None):
            text, original, retained = bounded_source_text(tokenizer, records, 2048)
            if (row['source_tokens_original'] != original or
                    row['source_tokens_retained'] != retained):
                raise ValueError('Changed historical source truncation')
            fresh = contextual_text_fields(agent, tokenizer, text,
                args.device, 2048, pooling='both')
            fields = update_fields(fields, fresh, successes) if policy == 'mean' else fresh
            successes += 1
    if (verify_state and vector_hash(fields) !=
            run['games'][index]['source_vector_sha256_before']):
        raise ValueError(f'{policy} memory state does not reproduce')
    return fields


def replay_target(run, index, step, args):
    row = run['games'][index]
    episode = row['episode']
    if (episode['status'] != 'complete' or episode['reward'] != 1 or
            step >= episode['steps']):
        raise ValueError('Need completed winning target and in-range action')
    env = make_env(args.data_root / row['game'])
    try:
        state = env.reset()
        initial = str(state['feedback'])
        if initial != episode['initial_observation']:
            raise ValueError('Target reset changed')
        messages = [{'role': 'system', 'content': ACTOR_SYSTEM}]
        seen = {}
        for turn in range(step):
            available = list(state['admissible_commands'])
            feedback = str(state['feedback'])
            messages.append({'role': 'user', 'content': feedback +
                '\nAvailable commands:\n' + '\n'.join(available)})
            saved = episode['trajectory'][turn]
            if saved['command'] not in guarded_commands(feedback, available, seen, 2):
                raise ValueError('Winning trajectory escaped guarded commands')
            key = (feedback, saved['command'])
            seen[key] = seen.get(key, 0) + 1
            state, _, done = env.step(saved['command'])
            if (done or str(state['feedback']) != saved['observation'] or
                    saved['won'] != bool(state['won'])):
                raise ValueError('Target replay diverged before intervention')
            messages.append({'role': 'assistant', 'content': saved['response']})
        feedback = str(state['feedback'])
        available = list(state['admissible_commands'])
        allowed = guarded_commands(feedback, available, seen, 2)
        messages.append({'role': 'user', 'content': feedback +
            '\nAvailable commands:\n' + '\n'.join(available)})
        return {'initial_observation': initial,
            'feedback': feedback, 'available': available,
            'allowed': allowed, 'messages': compact_messages(messages, 2),
            'recorded_next_command': episode['trajectory'][step]['command']}
    finally:
        env.close()


def analyze(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    runs = {name: json.loads(path.read_text()) for name, path in (
        ('latest', args.latest), ('mean', args.mean), ('first', args.first))}
    if any(run['checkpoint_sha256'] != file_hash(args.checkpoint) or
           run['failures'] or len(run['games']) != 18 for run in runs.values()):
        raise ValueError('Changed paired frozen run')
    game = runs['latest']['games'][args.index]['game']
    if any(run['games'][args.index]['game'] != game for run in runs.values()):
        raise ValueError('Different target problem across memories')
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    fields = {name: reconstruct(agent, tokenizer, run, args.index, name, args)
              for name, run in runs.items()}
    fields['mean_same_history'] = reconstruct(agent, tokenizer,
        runs['latest'], args.index, 'mean', args, verify_state=False)
    target = replay_target(runs['latest'], args.index, args.step, args)
    if (runs['mean']['games'][args.index]['episode']['trajectory'][:args.step] !=
            runs['latest']['games'][args.index]['episode']['trajectory'][:args.step]):
        raise ValueError('Mean and latest histories differ before chosen decision')
    target_fields = contextual_text_fields(agent, tokenizer,
        task_context_text(target['initial_observation'], target['feedback']),
        args.device, 2048, pooling='both')
    interventions = {}
    for name in ('base', 'first', 'mean', 'mean_same_history', 'latest'):
        with torch.no_grad():
            agent.set_source(fields[name] if name != 'base' else None,
                target_fields=target_fields if name != 'base' else None)
            b = torch.cat([layer.b.detach().float().flatten().cpu()
                for layer in agent.adapters]) if name != 'base' else None
            answer = generate(agent, tokenizer, target['messages'], args.device,
                64, target['allowed'])
            command = clean_command(answer, target['available'])
            agent.set_source(None)
        interventions[name] = {'answer': answer, 'command': command,
            'factor_norm': float(b.norm()) if b is not None else 0.0,
            'factor': b}
    if (interventions['latest']['command'] != target['recorded_next_command'] or
            interventions['mean']['command'] !=
            runs['mean']['games'][args.index]['episode']['trajectory'][args.step]['command']):
        raise ValueError('Matched replay did not reproduce original action')
    comparisons = {}
    for left, right in (('latest', 'mean'), ('latest', 'mean_same_history'),
                        ('latest', 'first'), ('mean', 'first')):
        a, b = interventions[left]['factor'], interventions[right]['factor']
        comparisons[f'{left}_vs_{right}'] = {'cosine': float(F.cosine_similarity(
            a[None], b[None]).item()),
            'relative_difference': float((a - b).norm() / b.norm())}
    for row in interventions.values():
        del row['factor']
    output = {'protocol': 'Same target state, actor prompt, available commands and frozen checkpoint; intervention changes only source trajectory vector generating B; target replay and source hashes checked',
        'checkpoint_sha256': file_hash(args.checkpoint),
        'run_sha256': {name: file_hash(path) for name, path in (
            ('latest', args.latest), ('mean', args.mean), ('first', args.first))},
        'game': game, 'index': args.index, 'step': args.step,
        'target': {key: value for key, value in target.items()
                   if key != 'messages'},
        'interventions': interventions,
        'matrix_comparisons': comparisons}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'commands': {k: v['command'] for k, v in interventions.items()},
        'matrix_comparisons': comparisons}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--latest', type=Path, default=Path('results/trajectory_hyperlora/alf_online_latest_success_train_seq6_6_20261007.json'))
    parser.add_argument('--mean', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq6_6_20261007.json'))
    parser.add_argument('--first', type=Path, default=Path('results/trajectory_hyperlora/alf_online_first_success_train_seq6_6_20261007.json'))
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--index', type=int, default=14)
    parser.add_argument('--step', type=int, default=13)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.65)
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_online_decision_counterfactual_seq6_index14_step13_20261007.json'))
    analyze(parser.parse_args())

if __name__ == '__main__':
    main()
