"""Independently audit the two additional fixed-snapshot oracle-Q replays."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.experience_evolution.core import seed
from ttcl.icl_mem0_comparison.protocol import read, sha
from .probe_oracle_q_additional import ARMS, design_for


def audit(output: Path) -> dict:
    path = output / 'design.json'
    design = read(path)
    if (design['schema'] != 'alf_oracle_q_additional_v1' or
            sha(Path(__file__).with_name('probe_oracle_q_additional.py')) !=
            design['runner_sha256'] or
            sha(Path(design['audit_path'])) != design['audit_sha256'] or
            design != design_for(Path(design['audit_path']), output,
                                 design['url'])):
        raise ValueError('Design or source retrieval changed')
    units, missing = [], []
    for case in design['cases']:
        origin = Path(case['origin'])
        source_row = origin / 'runs' / case['later_case'] / 'row.json'
        row = read(source_row)
        plan = read(origin / 'plan.json')
        game = row['game']
        if (sha(source_row) != case['row_sha256'] or
                sha(origin / 'plan.json') != case['source_plan_sha256'] or
                sha(Path(plan['alf']['data_root']) / game) !=
                case['later_input_sha256']):
            raise ValueError('Official task or source row changed')
        for repeat in design['repeats']:
            target = output / f"case_{case['case_id']}" / f'actor_repeat_{repeat}'
            summary_path = target / 'summary.json'
            if not summary_path.exists():
                missing.append(dict(case_id=case['case_id'], repeat=repeat))
                continue
            summary = read(summary_path)
            if (summary['case_id'] != case['case_id'] or
                    summary['repeat'] != repeat or
                    summary['design_sha256'] != sha(path) or
                    set(summary['replay']) != set(ARMS)):
                raise ValueError('Incomplete or changed summary')
            rewards, actions, invalid, calls = {}, {}, {}, 0
            initial = None
            for arm in ARMS:
                episode = read(target / arm / 'episode.json')
                context = (case['oracle_context'] if arm == 'oracle_q'
                           else case['native_context'])
                current_initial = (episode['initial_observation'],
                                   episode['initial_commands_sha256'])
                if initial is None:
                    initial = current_initial
                elif initial != current_initial:
                    raise ValueError('Arms differ in starting state')
                if (episode['status'] != 'complete' or
                        episode['game'] != game or
                        episode['seed'] != seed(repeat, game, 0, 'actor') or
                        episode['memory'] != context or
                        episode['memory_sha256'] != hashlib.sha256(
                            context.encode()).hexdigest() or
                        not 1 <= episode['steps'] <= 50 or
                        episode['steps'] != len(episode['trajectory']) or
                        episode['steps'] != len(episode['generations']) or
                        episode['reward'] not in (0, 1)):
                    raise ValueError('Invalid official episode')
                if dict(reward=episode['reward'], steps=episode['steps'],
                        first_action=episode['trajectory'][0]['action']) != \
                        summary['replay'][arm]:
                    raise ValueError('Summary differs from official episode')
                rewards[arm] = episode['reward']
                actions[arm] = [step['action'] for step in episode['trajectory']]
                invalid[arm] = sum(step['valid_command'] is False
                                   for step in episode['trajectory'])
                calls += len(episode['generations'])
            units.append(dict(case_id=case['case_id'], repeat=repeat,
                              rewards=rewards, invalid_commands=invalid,
                              actor_calls=calls,
                              native_repeat_reward_equal=(
                                  rewards['native'] == rewards['native_repeat']),
                              native_repeat_actions_equal=(
                                  actions['native'] == actions['native_repeat'])))
    stable = [unit for unit in units if unit['native_repeat_reward_equal']]
    return dict(schema='alf_oracle_q_additional_audit_v1',
                design_sha256=sha(path), expected=(len(design['cases']) *
                                                   len(design['repeats'])),
                audited=len(units), missing=missing, units=units,
                totals=dict(stable=len(stable),
                            wins=sum(u['rewards']['oracle_q'] > u['rewards']['native']
                                     for u in stable),
                            losses=sum(u['rewards']['oracle_q'] < u['rewards']['native']
                                       for u in stable),
                            ties=sum(u['rewards']['oracle_q'] == u['rewards']['native']
                                     for u in stable),
                            actor_calls=sum(u['actor_calls'] for u in units),
                            invalid_native=sum(u['invalid_commands']['native'] for u in units),
                            invalid_oracle_q=sum(u['invalid_commands']['oracle_q'] for u in units)),
                caveat='Development tasks, posthoc earlier-task credit, fixed snapshots, no online memory evolution or learned policy')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.output.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(expected=result['expected'], audited=result['audited'],
                          missing=result['missing'], totals=result['totals'])))


if __name__ == '__main__':
    main()
