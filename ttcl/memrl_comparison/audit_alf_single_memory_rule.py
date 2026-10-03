"""Audit the frozen only-third-memory ALFWorld validation and repeat controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.experience_evolution.core import seed
from ttcl.icl_mem0_comparison.protocol import read, sha
from .audit_credit_train_extension import audit as audit_source
from .credit_probe import memory_arms
from .validate_alf_single_memory_rule import ORDERS, selected_cases


def audit(output: Path) -> dict:
    design_path = output / 'design.json'
    design = read(design_path)
    if design['schema'] != 'alf_single_memory_rule_validation_v1':
        raise ValueError('Wrong validation schema')
    origin, exploration = Path(design['origin']), Path(design['exploration'])
    source = audit_source(origin)
    if not source['complete'] or source['missing'] or source['audited_pairs'] != 18:
        raise ValueError('Validation source audit failed')
    for path, expected in ((origin / 'plan.json', design['source_plan_sha256']),
                           (origin / 'design.json', design['source_design_sha256']),
                           (origin / 'complete.json', design['source_complete_sha256']),
                           (exploration / 'design.json', design['exploration_design_sha256']),
                           (exploration / 'summary.json', design['exploration_summary_sha256']),
                           (Path(__file__).with_name('validate_alf_single_memory_rule.py'),
                            design['runner_sha256']),
                           (Path(__file__).with_name('credit_probe.py'),
                            design['credit_probe_sha256'])):
        if sha(path) != expected:
            raise ValueError(f'Frozen source changed: {path}')
    exploration_origin = Path(read(exploration / 'design.json')['origin'])
    prior_hashes = {game['sha256'] for games in
                    read(exploration_origin / 'design.json')['selected_games'].values()
                    for game in games}
    if [item['case'] for item in design['cases']] != selected_cases(origin):
        raise ValueError('Validation selection changed')
    plan = read(origin / 'plan.json')
    units, missing = [], []
    for item in design['cases']:
        case = item['case']
        spec, arms = memory_arms(origin, case)
        if (len(spec['ids']) != 3 or item['ids'] != spec['ids'] or
                item['input_sha256'] != spec['source_input_sha256'] or
                item['input_sha256'] in prior_hashes or
                item['snapshot_sha256'] != spec['snapshot_sha256'] or
                item['retrieval_sha256'] != spec['retrieval_sha256'] or
                item['original_row_sha256'] != spec['original_memrl_row_sha256'] or
                item['full_sha256'] != spec['arm_context_sha256']['full'] or
                item['only_2_sha256'] != spec['arm_context_sha256']['only_2']):
            raise ValueError('Case source or content binding changed')
        game = spec['original_memrl']['game']
        if sha(Path(plan['alf']['data_root']) / game) != item['input_sha256']:
            raise ValueError('Official game content changed')
        for repeat in design['repeats']:
            target = output / case / f'actor_repeat_{repeat}'
            summary_path = target / 'summary.json'
            if not summary_path.exists():
                missing.append(dict(case=case, repeat=repeat))
                continue
            summary, order = read(summary_path), read(target / 'order.json')
            index = int(hashlib.sha256(f'{case}/{repeat}'.encode()).hexdigest(), 16) % 2
            names = list(ORDERS[index])
            if (order != dict(names=names, seed=repeat) or
                    summary['case'] != case or summary['repeat'] != repeat or
                    summary['design_sha256'] != sha(design_path) or
                    summary['order'] != names or set(summary['replay']) != set(names)):
                raise ValueError('Balanced order or summary changed')
            rewards, actions, calls, initial = {}, {}, 0, None
            for name in names:
                episode = read(target / name / 'episode.json')
                context = arms['full' if name.startswith('full') else 'only_2']
                start = (episode['initial_observation'],
                         episode['initial_commands_sha256'])
                if initial is None:
                    initial = start
                elif start != initial:
                    raise ValueError('Paired arms started in different states')
                if (episode['status'] != 'complete' or episode['game'] != game or
                        episode['seed'] != seed(repeat, game, 0, 'actor') or
                        episode['memory'] != context or
                        episode['memory_sha256'] != hashlib.sha256(context.encode()).hexdigest() or
                        episode['reward'] not in (0, 1) or
                        not 1 <= episode['steps'] <= 50 or
                        episode['steps'] != len(episode['trajectory']) or
                        episode['steps'] != len(episode['generations'])):
                    raise ValueError('Official episode differs from frozen arm')
                actual = dict(reward=episode['reward'], steps=episode['steps'],
                              first_action=episode['trajectory'][0]['action'])
                if actual != summary['replay'][name]:
                    raise ValueError('Reported reward differs from official episode')
                rewards[name] = episode['reward']
                actions[name] = [step['action'] for step in episode['trajectory']]
                calls += len(episode['generations'])
            candidate = [rewards['only_2'], rewards['only_2_repeat']]
            baseline = [rewards['full'], rewards['full_repeat']]
            effect = (sum(candidate) - sum(baseline)) / 2
            radius = (abs(candidate[0] - candidate[1]) +
                      abs(baseline[0] - baseline[1])) / 2
            units.append(dict(case=case, repeat=repeat, candidate=candidate,
                              baseline=baseline, primary_delta=candidate[0] - baseline[0],
                              corrected_effect=effect, repeat_noise_radius=radius,
                              decisive_positive=effect > radius + 1e-12,
                              decisive_negative=effect < -radius - 1e-12,
                              full_actions_equal=actions['full'] == actions['full_repeat'],
                              candidate_actions_equal=
                              actions['only_2'] == actions['only_2_repeat'],
                              actor_calls=calls))
    return dict(schema='alf_single_memory_rule_audit_v1',
                design_sha256=sha(design_path),
                expected=len(design['cases']) * len(design['repeats']),
                audited=len(units), missing=missing, units=units,
                totals=dict(primary_wins=sum(u['primary_delta'] > 0 for u in units),
                            primary_losses=sum(u['primary_delta'] < 0 for u in units),
                            decisive_positive=sum(u['decisive_positive'] for u in units),
                            decisive_negative=sum(u['decisive_negative'] for u in units),
                            full_actions_equal=sum(u['full_actions_equal'] for u in units),
                            candidate_actions_equal=sum(u['candidate_actions_equal'] for u in units),
                            actor_calls=sum(u['actor_calls'] for u in units)),
                caveat='Fixed-snapshot official train validation of one exploratory rule; not an online MemRL score')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = audit(args.output.resolve())
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
