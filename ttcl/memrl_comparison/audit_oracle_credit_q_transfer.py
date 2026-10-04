"""Independently audit fixed-snapshot oracle-Q transfer episodes and reward."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.experience_evolution.core import seed
from ttcl.icl_mem0_comparison.protocol import read, sha
from .probe_oracle_credit_q_transfer import design_for


ARMS = ('native', 'oracle_q', 'native_repeat')


def audit(output: Path) -> dict:
    path = output / 'design.json'
    design = read(path)
    if design['schema'] != 'alf_oracle_credit_q_transfer_v1' or \
            sha(Path(__file__).with_name('probe_oracle_credit_q_transfer.py')) != \
            design['runner_sha256']:
        raise ValueError('Wrong or changed oracle-Q transfer design')
    expected = design_for(Path(design['origin']), Path(design['prior_probe']),
                          output, design['url'])
    if design != expected:
        raise ValueError('Oracle-Q retrieval or lineage changed')
    origin = Path(design['origin'])
    source = read(origin / 'runs' / design['next_case'] / 'row.json')
    game = source['game']
    plan = read(origin / 'plan.json')
    if sha(Path(plan['alf']['data_root']) / game) != design['next_input_sha256']:
        raise ValueError('Official ALFWorld input changed')
    units, missing = [], []
    for repeat in design['repeats']:
        root = output / f'actor_repeat_{repeat}'
        summary_path = root / 'summary.json'
        if not summary_path.exists():
            missing.append(repeat)
            continue
        summary = read(summary_path)
        if summary['repeat'] != repeat or summary['design_sha256'] != sha(path) or \
                set(summary['replay']) != set(ARMS):
            raise ValueError('Changed or incomplete transfer summary')
        reward, actions, invalid, calls = {}, {}, {}, 0
        initial = None
        for arm in ARMS:
            episode = read(root / arm / 'episode.json')
            context = (design['changed_context'] if arm == 'oracle_q'
                       else design['original_context'])
            current_initial = (episode['initial_observation'],
                               episode['initial_commands_sha256'])
            if initial is None:
                initial = current_initial
            elif current_initial != initial:
                raise ValueError('Arms did not start at the same ALFWorld state')
            if (episode['status'] != 'complete' or episode['game'] != game or
                    episode['seed'] != seed(repeat, game, 0, 'actor') or
                    episode['memory'] != context or
                    episode['memory_sha256'] != hashlib.sha256(context.encode()).hexdigest() or
                    not 1 <= episode['steps'] <= 50 or
                    episode['steps'] != len(episode['trajectory']) or
                    episode['steps'] != len(episode['generations']) or
                    episode['reward'] not in (0, 1)):
                raise ValueError(f'Invalid official transfer episode: {repeat}/{arm}')
            actual = dict(reward=episode['reward'], steps=episode['steps'],
                          first_action=episode['trajectory'][0]['action'])
            if actual != summary['replay'][arm]:
                raise ValueError('Summary differs from official episode')
            reward[arm] = episode['reward']
            actions[arm] = [row['action'] for row in episode['trajectory']]
            invalid[arm] = sum(row['valid_command'] is False
                               for row in episode['trajectory'])
            calls += len(episode['generations'])
        units.append(dict(repeat=repeat, rewards=reward,
                          invalid_commands=invalid, actor_calls=calls,
                          native_repeat_reward_equal=(reward['native'] ==
                                                      reward['native_repeat']),
                          native_repeat_actions_equal=(actions['native'] ==
                                                       actions['native_repeat'])))
    stable = [u for u in units if u['native_repeat_reward_equal']]
    return dict(schema='alf_oracle_credit_q_transfer_audit_v1',
                design_sha256=sha(path), expected=len(design['repeats']),
                audited=len(units), missing=missing, units=units,
                totals=dict(stable=len(stable),
                            wins=sum(u['rewards']['oracle_q'] > u['rewards']['native']
                                     for u in stable),
                            losses=sum(u['rewards']['oracle_q'] < u['rewards']['native']
                                       for u in stable),
                            ties=sum(u['rewards']['oracle_q'] == u['rewards']['native']
                                     for u in stable),
                            invalid_native=sum(u['invalid_commands']['native'] for u in units),
                            invalid_oracle_q=sum(u['invalid_commands']['oracle_q'] for u in units),
                            actor_calls=sum(u['actor_calls'] for u in units)),
                caveat='Fixed-snapshot oracle local credit directly overwrites one Q; not learned, not native EMA, not an online chain')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.output.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(audited=result['audited'], expected=result['expected'],
                          missing=result['missing'], totals=result['totals'])))


if __name__ == '__main__':
    main()
