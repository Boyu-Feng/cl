"""Audit reviewed ALFWorld source-entity rebinding with repeat controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from ttcl.experience_evolution.core import seed
from ttcl.icl_mem0_comparison.protocol import read, sha
from .probe_source_entity_rebinding import ARMS, design_for


def audit(sign_report: Path, output: Path) -> dict:
    path = output / 'design.json'
    design = read(path)
    if (design['schema'] != 'alf_source_entity_rebinding_v1' or
            sha(Path(__file__).with_name('probe_source_entity_rebinding.py')) !=
            design['runner_sha256'] or
            design != design_for(sign_report, output, design['url'])):
        raise ValueError('Frozen entity-rebinding design or source changed')
    units, missing = [], []
    for case in design['cases']:
        origin = Path(case['origin'])
        plan = read(origin / 'plan.json')
        source_game = Path(plan['alf']['data_root']) / case['game']
        if (sha(origin / 'plan.json') != case['source_plan_sha256'] or
                sha(source_game) != case['game_sha256']):
            raise ValueError('Official source plan or game changed')
        for repeat in design['repeats']:
            target = output / f"case_{case['index']}" / f'repeat_{repeat}'
            summary_path = target / 'summary.json'
            if not summary_path.exists():
                missing.append(dict(case=case['index'], repeat=repeat))
                continue
            summary = read(summary_path)
            if (summary['case'] != case['index'] or
                    summary['repeat'] != repeat or
                    summary['design_sha256'] != sha(path) or
                    set(summary['replay']) != set(ARMS)):
                raise ValueError('Missing or changed actor summary')
            contexts = dict(native=case['native_context'],
                            rebound=case['rebound_context'], none='',
                            native_repeat=case['native_context'],
                            rebound_repeat=case['rebound_context'])
            initial, rewards, steps, actions, invalid = None, {}, {}, {}, {}
            calls = 0
            for arm in ARMS:
                episode = read(target / arm / 'episode.json')
                context = contexts[arm]
                current_initial = (episode['initial_observation'],
                                   episode['initial_commands_sha256'])
                if initial is None:
                    initial = current_initial
                elif initial != current_initial:
                    raise ValueError('Actor arms start at different public states')
                if (episode['status'] != 'complete' or
                        episode['game'] != case['game'] or
                        episode['seed'] != seed(repeat, case['game'], 0, 'actor') or
                        episode['memory'] != context or
                        episode['memory_sha256'] !=
                        hashlib.sha256(context.encode()).hexdigest() or
                        not 1 <= episode['steps'] <= 50 or
                        episode['steps'] != len(episode['trajectory']) or
                        episode['steps'] != len(episode['generations']) or
                        episode['reward'] not in (0, 1)):
                    raise ValueError('Invalid official ALFWorld episode')
                actual = dict(reward=episode['reward'],
                              steps=episode['steps'],
                              first_action=episode['trajectory'][0]['action'])
                if actual != summary['replay'][arm]:
                    raise ValueError('Summary disagrees with official episode')
                rewards[arm] = episode['reward']
                steps[arm] = episode['steps']
                actions[arm] = [row['action'] for row in episode['trajectory']]
                invalid[arm] = sum(row['valid_command'] is False
                                   for row in episode['trajectory'])
                calls += len(episode['generations'])
            source_obj = case['entity_binding']['source_object']
            goal_obj = case['entity_binding']['current_object']
            wrong_takes = {}
            goal_takes = {}
            for arm in ARMS:
                wrong_takes[arm] = sum(bool(re.search(
                    r'^take ' + re.escape(source_obj) + r' \d+\b',
                    action, re.I)) for action in actions[arm])
                goal_takes[arm] = sum(bool(re.search(
                    r'^take ' + re.escape(goal_obj) + r' \d+\b',
                    action, re.I)) for action in actions[arm])
            units.append(dict(case=case['index'],
                              public_task=case['public_task'],
                              repeat=repeat, rewards=rewards, steps=steps,
                              invalid_commands=invalid,
                              source_object_takes=wrong_takes,
                              goal_object_takes=goal_takes,
                              actor_calls=calls,
                              native_repeat_reward_equal=(
                                  rewards['native'] == rewards['native_repeat']),
                              rebound_repeat_reward_equal=(
                                  rewards['rebound'] == rewards['rebound_repeat']),
                              native_repeat_actions_equal=(
                                  actions['native'] == actions['native_repeat']),
                              rebound_repeat_actions_equal=(
                                  actions['rebound'] == actions['rebound_repeat'])))
    stable = [unit for unit in units
              if unit['native_repeat_reward_equal'] and
              unit['rebound_repeat_reward_equal']]
    def pair(baseline: str) -> dict:
        return dict(wins=sum(unit['rewards']['rebound'] >
                             unit['rewards'][baseline] for unit in stable),
                    losses=sum(unit['rewards']['rebound'] <
                               unit['rewards'][baseline] for unit in stable),
                    ties=sum(unit['rewards']['rebound'] ==
                             unit['rewards'][baseline] for unit in stable))
    return dict(schema='alf_source_entity_rebinding_audit_v1',
                design_sha256=sha(path),
                expected=len(design['cases']) * len(design['repeats']),
                audited=len(units), missing=missing, units=units,
                totals=dict(stable_both_repeats=len(stable),
                            native_repeat_reward_equal=sum(
                                u['native_repeat_reward_equal'] for u in units),
                            rebound_repeat_reward_equal=sum(
                                u['rebound_repeat_reward_equal'] for u in units),
                            native_repeat_actions_equal=sum(
                                u['native_repeat_actions_equal'] for u in units),
                            rebound_repeat_actions_equal=sum(
                                u['rebound_repeat_actions_equal'] for u in units),
                            rebound_vs_native_on_stable=pair('native'),
                            rebound_vs_none_on_stable=pair('none'),
                            actor_calls=sum(u['actor_calls'] for u in units)),
                caveat='Posthoc two-game train mechanism test; only units with both repeat rewards stable enter paired win/loss counts; fixed single-memory context is not online MemRL or CLBench')


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
