"""Audit paired suffixes from identical reviewed ALFWorld evidence prefixes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.experience_evolution.core import digest, seed
from ttcl.experience_evolution.environment import clean_command
from ttcl.icl_mem0_comparison.protocol import read, sha
from .probe_memory_at_evidence_prefix import ARMS, design_for, replay_prefix


def _suffix(plan: dict, case: dict, repeat: int, arm: str,
            path: Path) -> dict:
    episode = read(path)
    memory = case['memory_context'] if arm == 'with_memory' else ''
    actor_seed = seed(repeat, case['game'], 0, 'actor')
    if (episode['status'] != 'complete' or episode['game'] != case['game'] or
            episode['game_sha256'] != case['game_sha256'] or
            episode['arm'] != arm or episode['repeat'] != repeat or
            episode['seed'] != actor_seed or
            episode['prefix_length'] != case['prefix_length'] or
            episode['source_stage_episode_sha256'] !=
            case['source_stage_episode_sha256'] or
            episode['memory'] != memory or
            episode['memory_sha256'] != hashlib.sha256(memory.encode()).hexdigest() or
            episode['reward'] not in (0, 1) or
            not case['prefix_length'] < episode['total_steps'] <= 50 or
            episode['total_steps'] - case['prefix_length'] !=
            len(episode['trajectory']) or
            len(episode['trajectory']) != len(episode['generations'])):
        raise ValueError('Invalid evidence-prefix continuation')
    env, state, messages, initial = replay_prefix(plan, case)
    try:
        if (initial != (episode['initial_observation'],
                        episode['initial_commands_sha256']) or
                episode['suffix_start_observation'] != str(state['feedback']) or
                episode['suffix_start_observation_sha256'] !=
                case['last_prefix_observation_sha256']):
            raise ValueError('Suffix did not start from frozen public state')
        if memory:
            messages[0]['content'] += '\n\nPast experience:\n' + memory
        for offset, (step, generation) in enumerate(zip(
                episode['trajectory'], episode['generations'])):
            turn = case['prefix_length'] + offset
            available = list(state['admissible_commands'])
            messages.append({'role':'user', 'content':str(state['feedback']) +
                             '\nAvailable commands:\n' + '\n'.join(available)})
            if (generation['seed'] != seed(actor_seed, turn) or
                    generation['prompt_sha256'] != digest(messages) or
                    clean_command(generation['text'], available) != step['action'] or
                    (step['action'] in available) is not step['valid_command']):
                raise ValueError('Continuation prompt, seed or command changed')
            state, _, done = env.step(step['action'])
            if str(state['feedback']) != step['observation']:
                raise ValueError('Continuation public feedback changed')
            messages.append({'role':'assistant', 'content':generation['text']})
            terminal = done or state['won'] or turn + 1 == 50
            if terminal and turn + 1 != episode['total_steps']:
                raise ValueError('Continuation ran after terminal state')
        if (episode['reward'] != float(bool(state['won'])) or
                episode['termination'] != ('success' if state['won'] else
                                           'budget_or_environment_done')):
            raise ValueError('Official continuation reward changed')
    finally:
        env.close()
    return dict(reward=episode['reward'], total_steps=episode['total_steps'],
                actor_calls=len(episode['generations']),
                actions=[step['action'] for step in episode['trajectory']],
                invalid_commands=sum(step['valid_command'] is False
                                     for step in episode['trajectory']),
                initial=initial,
                start_observation=episode['suffix_start_observation'])


def audit(stage: Path, reviews: Path, output: Path) -> dict:
    design_path = output / 'design.json'
    design = read(design_path)
    if (design['schema'] != 'alf_memory_at_evidence_prefix_v1' or
            sha(Path(__file__).with_name('probe_memory_at_evidence_prefix.py')) !=
            design['runner_sha256'] or
            design != design_for(stage, reviews, output, design['url'])):
        raise ValueError('Frozen prefix fork design or source changed')
    units, missing = [], []
    for case in design['cases']:
        plan = read(Path(case['origin']) / 'plan.json')
        if sha(Path(case['origin']) / 'plan.json') != case['source_plan_sha256']:
            raise ValueError('Source actor plan changed')
        for repeat in design['repeats']:
            paths = {arm:output / f"case_{case['index']}" /
                     f'repeat_{repeat}' / arm / 'episode.json' for arm in ARMS}
            if any(not path.exists() for path in paths.values()):
                missing.append(dict(case=case['index'], repeat=repeat))
                continue
            arms = {arm:_suffix(plan, case, repeat, arm, path)
                    for arm, path in paths.items()}
            if (len({tuple(row['initial']) for row in arms.values()}) != 1 or
                    len({row['start_observation'] for row in arms.values()}) != 1):
                raise ValueError('Arms start from different public states')
            units.append(dict(case=case['index'], public_task=case['public_task'],
                              repeat=repeat,
                              rewards={name:row['reward'] for name,row in arms.items()},
                              total_steps={name:row['total_steps']
                                           for name,row in arms.items()},
                              invalid_commands={name:row['invalid_commands']
                                                for name,row in arms.items()},
                              actor_calls=sum(row['actor_calls']
                                              for row in arms.values()),
                              without_repeat_reward_equal=(
                                  arms['without_memory']['reward'] ==
                                  arms['without_repeat']['reward']),
                              without_repeat_actions_equal=(
                                  arms['without_memory']['actions'] ==
                                  arms['without_repeat']['actions']),
                              with_without_actions_equal=(
                                  arms['with_memory']['actions'] ==
                                  arms['without_memory']['actions'])))
    stable = [unit for unit in units if unit['without_repeat_reward_equal']]
    return dict(schema='alf_memory_at_evidence_prefix_audit_v1',
                design_sha256=sha(design_path),
                expected=len(design['cases']) * len(design['repeats']),
                audited=len(units), missing=missing, units=units,
                totals=dict(stable=len(stable),
                            wins=sum(u['rewards']['with_memory'] >
                                     u['rewards']['without_memory'] for u in stable),
                            losses=sum(u['rewards']['with_memory'] <
                                       u['rewards']['without_memory'] for u in stable),
                            ties=sum(u['rewards']['with_memory'] ==
                                     u['rewards']['without_memory'] for u in stable),
                            without_repeat_actions_equal=sum(
                                u['without_repeat_actions_equal'] for u in units),
                            with_without_actions_equal=sum(
                                u['with_without_actions_equal'] for u in units),
                            actor_calls=sum(u['actor_calls'] for u in units)),
                caveat='Two posthoc selected reviewed train prefixes; same public state, chat and budget; no online memory updates, unseen games or CLBench')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.stage.resolve(), args.reviews.resolve(),
                   args.output.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(expected=result['expected'], audited=result['audited'],
                          missing=result['missing'], totals=result['totals'])))


if __name__ == '__main__':
    main()
