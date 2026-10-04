"""Replay public ALFWorld state and actor prompts for delayed-memory arms."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from ttcl.alfworld_comparison.environment import make_env
from ttcl.experience_evolution.core import digest, seed
from ttcl.experience_evolution.environment import ACTOR_SYSTEM, clean_command
from ttcl.icl_mem0_comparison.protocol import read, sha
from .probe_delayed_memory_activation import ARMS, design_for


def _episode(plan: dict, case: dict, repeat: int, arm: str,
             path: Path) -> dict:
    episode = read(path)
    actor_seed = seed(repeat, case['game'], 0, 'actor')
    memory = case['memory_context']
    if (episode['status'] != 'complete' or episode['game'] != case['game'] or
            episode['game_sha256'] != case['game_sha256'] or
            episode['case'] != case['case'] or episode['arm'] != arm or
            episode['seed'] != actor_seed or episode['repeat'] != repeat or
            episode['eligible_memory'] != memory or
            episode['eligible_memory_sha256'] !=
            hashlib.sha256(memory.encode()).hexdigest() or
            episode['goal_object'] != case['goal_object'] or
            episode['reward'] not in (0, 1) or
            not 1 <= episode['steps'] <= plan['alf']['max_steps'] or
            episode['steps'] != len(episode['trajectory']) or
            episode['steps'] != len(episode['generations'])):
        raise ValueError('Invalid official stage episode')
    game = Path(plan['alf']['data_root']) / case['game']
    if sha(game) != case['game_sha256']:
        raise ValueError('Official game changed')
    target = re.compile(r'\b(?:a|an) ' + re.escape(case['goal_object']) +
                        r' \d+\b', re.I)
    active = arm in ('always', 'always_repeat')
    activation = 1 if active else None
    messages = [{'role':'system', 'content':ACTOR_SYSTEM +
                 ('\n\nPast experience:\n' + memory if active else '')}]
    env = make_env(game)
    try:
        state = env.reset()
        initial = (str(state['feedback']),
                   digest(json.dumps(list(state['admissible_commands']))))
        if (initial != (episode['initial_observation'],
                        episode['initial_commands_sha256']) or
                case['public_task'] not in initial[0]):
            raise ValueError('Public reset state changed')
        for turn, (step, generation) in enumerate(zip(
                episode['trajectory'], episode['generations'])):
            feedback = str(state['feedback'])
            available = list(state['admissible_commands'])
            messages.append({'role':'user', 'content':feedback +
                             '\nAvailable commands:\n' + '\n'.join(available)})
            if (generation['seed'] != seed(actor_seed, turn) or
                    generation['prompt_sha256'] != digest(messages) or
                    step['memory_active'] is not active or
                    clean_command(generation['text'], available) != step['action'] or
                    (step['action'] in available) is not step['valid_command']):
                raise ValueError('Actor seed, prompt or command differs')
            state, _, done = env.step(step['action'])
            observed = str(state['feedback'])
            triggered = bool(target.search(observed))
            if (step['observation'] != observed or
                    step['trigger_after_action'] is not triggered):
                raise ValueError('Public feedback or activation trigger differs')
            messages.append({'role':'assistant', 'content':generation['text']})
            terminal = done or state['won'] or turn + 1 == plan['alf']['max_steps']
            if terminal and turn + 1 != episode['steps']:
                raise ValueError('Actor continued after official terminal state')
            if not terminal and arm == 'delayed' and not active and triggered:
                active = True
                activation = turn + 2
                messages[0]['content'] += '\n\nPast experience:\n' + memory
        if (episode['activation_turn'] != activation or
                episode['reward'] != float(bool(state['won'])) or
                episode['termination'] != ('success' if state['won'] else
                                           'budget_or_environment_done')):
            raise ValueError('Official reward or activation differs')
    finally:
        env.close()
    return dict(reward=episode['reward'], steps=episode['steps'],
                activation_turn=episode['activation_turn'],
                invalid_commands=sum(step['valid_command'] is False
                                     for step in episode['trajectory']),
                actor_calls=len(episode['generations']),
                actions=[step['action'] for step in episode['trajectory']],
                initial=initial)


def audit(sign_report: Path, output: Path) -> dict:
    design_path = output / 'design.json'
    design = read(design_path)
    if (design['schema'] != 'alf_delayed_memory_activation_v1' or
            sha(Path(__file__).with_name('probe_delayed_memory_activation.py')) !=
            design['runner_sha256'] or
            design != design_for(sign_report, output, design['url'])):
        raise ValueError('Frozen stage design or source changed')
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
            arms = {arm:_episode(plan, case, repeat, arm, path)
                    for arm, path in paths.items()}
            if len({tuple(arm['initial']) for arm in arms.values()}) != 1:
                raise ValueError('Paired arms start from different public states')
            delayed_path = read(paths['delayed'])
            none_path = read(paths['none'])
            delayed_actions = arms['delayed']['actions']
            none_actions = arms['none']['actions']
            first_action_divergence = next(
                (i for i,(left,right) in enumerate(zip(delayed_actions,
                                                       none_actions),1)
                 if left != right), None)
            if first_action_divergence is None and \
                    len(delayed_actions) != len(none_actions):
                first_action_divergence = min(len(delayed_actions),
                                              len(none_actions)) + 1
            first_prompt_divergence = next(
                (i for i,(left,right) in enumerate(zip(
                    delayed_path['generations'], none_path['generations']),1)
                 if left['prompt_sha256'] != right['prompt_sha256']), None)
            activation = arms['delayed']['activation_turn']
            preactivation_action_divergence = bool(
                first_action_divergence is not None and
                (activation is None or first_action_divergence < activation))
            units.append(dict(case=case['index'], public_task=case['public_task'],
                              prior_shapley_sign=case['local_shapley_sign'],
                              repeat=repeat,
                              rewards={arm:row['reward'] for arm,row in arms.items()},
                              steps={arm:row['steps'] for arm,row in arms.items()},
                              activation_turn=activation,
                              first_delayed_none_prompt_divergence=first_prompt_divergence,
                              first_delayed_none_action_divergence=first_action_divergence,
                              preactivation_action_divergence=preactivation_action_divergence,
                              delayed_none_actions_equal=(delayed_actions == none_actions),
                              invalid_commands={arm:row['invalid_commands']
                                                for arm,row in arms.items()},
                              actor_calls=sum(row['actor_calls']
                                              for row in arms.values()),
                              always_repeat_equal=(arms['always']['reward'] ==
                                                   arms['always_repeat']['reward']),
                              always_repeat_actions_equal=(arms['always']['actions'] ==
                                                            arms['always_repeat']['actions'])))
    stable = [unit for unit in units if unit['always_repeat_equal']]
    def comparison(baseline: str) -> dict:
        return dict(wins=sum(unit['rewards']['delayed'] >
                             unit['rewards'][baseline] for unit in stable),
                    losses=sum(unit['rewards']['delayed'] <
                               unit['rewards'][baseline] for unit in stable),
                    ties=sum(unit['rewards']['delayed'] ==
                             unit['rewards'][baseline] for unit in stable))
    return dict(schema='alf_delayed_memory_activation_audit_v1',
                design_sha256=sha(design_path), expected=(len(design['cases']) *
                                                         len(design['repeats'])),
                audited=len(units), missing=missing, units=units,
                totals=dict(stable=len(stable),
                            repeat_actions_equal=sum(
                                unit['always_repeat_actions_equal'] for unit in units),
                            delayed_none_actions_equal=sum(
                                unit['delayed_none_actions_equal'] for unit in units),
                            preactivation_action_divergences=sum(
                                unit['preactivation_action_divergence'] for unit in units),
                            delayed_vs_always=comparison('always'),
                            delayed_vs_none=comparison('none'),
                            actor_calls=sum(unit['actor_calls'] for unit in units)),
                caveat='Posthoc selected two reviewed train games, three new actor seeds each; fixed single-memory context and no online writer/Q evolution or other ALFWorld families/CLBench')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sign-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.sign_report.resolve(), args.output.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(expected=result['expected'], audited=result['audited'],
                          missing=result['missing'], totals=result['totals'])))


if __name__ == '__main__':
    main()
