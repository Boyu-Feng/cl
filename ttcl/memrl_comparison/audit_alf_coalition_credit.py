"""Independently check frozen coalition arms and compute exact three-player credit."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from ttcl.experience_evolution.core import seed
from ttcl.icl_mem0_comparison.protocol import read, sha
from .audit_credit_train_extension import audit as audit_source
from .credit_probe import memory_arms


def shapley_three(values: dict[frozenset[int], float]) -> list[float]:
    full = frozenset(range(3))
    if set(values) != {frozenset(i for i in range(3) if mask & (1 << i))
                       for mask in range(8)}:
        raise ValueError('Missing coalition')
    return [
        (values[frozenset({i})] - values[frozenset()]) / 3
        + sum((values[frozenset({i, j})] - values[frozenset({j})]) / 6
              for j in range(3) if j != i)
        + (values[full] - values[full - {i}]) / 3
        for i in range(3)
    ]


def audit(output: Path) -> dict:
    design = read(output / 'design.json')
    if design['schema'] not in ('alf_three_memory_coalition_v1',
                                'alf_three_memory_coalition_holdout_v1'):
        raise ValueError('Wrong design schema')
    holdout = design['schema'] == 'alf_three_memory_coalition_holdout_v1'
    origin = Path(design['origin'])
    report = audit_source(origin)
    if not report['complete'] or report['missing']:
        raise ValueError('Source audit incomplete')
    for name, expected in (('plan.json', design['plan_sha256']),
                           ('design.json', design['source_design_sha256']),
                           ('complete.json', design['source_complete_sha256'])):
        if sha(origin / name) != expected:
            raise ValueError(f'Source {name} changed')
    root = Path(__file__).parent
    runner = 'probe_alf_coalition_holdout.py' if holdout else 'probe_alf_coalition_credit.py'
    if (sha(root / runner) != design['runner_sha256'] or
            sha(root / 'credit_probe.py') != design['credit_probe_sha256']):
        raise ValueError('Frozen runner source changed')
    if holdout:
        from .probe_alf_coalition_credit import CASES as previous_cases
        from .probe_alf_coalition_holdout import selected_cases
        if (design['previous_cases'] != list(previous_cases) or
                sha(root / 'probe_alf_coalition_credit.py') != design['previous_runner_sha256'] or
                [item['case'] for item in design['cases']] != selected_cases(origin)):
            raise ValueError('Outcome-blind selection changed')
    plan = read(origin / 'plan.json')
    units, missing = [], []
    for item in design['cases']:
        case = item['case']
        spec, arms = memory_arms(origin, case)
        if (spec['source_input_sha256'] != item['input_sha256'] or
                spec['ids'] != item['ids'] or
                spec['snapshot_sha256'] != item['snapshot_sha256'] or
                spec['retrieval_sha256'] != item['retrieval_sha256'] or
                spec['original_memrl_row_sha256'] != item['original_row_sha256'] or
                spec['arm_context_sha256'] != item['context_sha256']):
            raise ValueError(f'Case source changed: {case}')
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
            names = order['names']
            interior = [name for name in arms if name != 'full']
            random.Random(int(hashlib.sha256(f'{case}/{repeat}'.encode()).hexdigest(), 16)).shuffle(interior)
            if (summary['case'] != case or summary['repeat'] != repeat or
                    summary['design_sha256'] != sha(output / 'design.json') or
                    summary['order'] != names or order['seed'] != repeat or
                    names != ['full', *interior, 'full_repeat'] or
                    set(names) != set(arms) | {'full_repeat'} or
                    set(summary['replay']) != set(names)):
                raise ValueError('Missing, duplicate or changed arm/order')
            rewards, actions, calls = {}, {}, 0
            initial = None
            for name in names:
                episode = read(target / name / 'episode.json')
                context = arms['full'] if name == 'full_repeat' else arms[name]
                current_initial = (episode['initial_observation'],
                                   episode['initial_commands_sha256'])
                if initial is None:
                    initial = current_initial
                elif current_initial != initial:
                    raise ValueError('Coalition arms did not start in the same state')
                if (episode['status'] != 'complete' or episode['game'] != game or
                        episode['seed'] != seed(repeat, game, 0, 'actor') or
                        episode['memory'] != context or
                        episode['memory_sha256'] != hashlib.sha256(context.encode()).hexdigest() or
                        not 1 <= episode['steps'] <= 50 or
                        episode['steps'] != len(episode['trajectory']) or
                        episode['steps'] != len(episode['generations']) or
                        episode['reward'] not in (0, 1)):
                    raise ValueError(f'Invalid official episode: {case}/{repeat}/{name}')
                actual = dict(reward=episode['reward'], steps=episode['steps'],
                              first_action=episode['trajectory'][0]['action'])
                if actual != summary['replay'][name]:
                    raise ValueError('Summary differs from episode')
                rewards[name] = episode['reward']
                actions[name] = [row['action'] for row in episode['trajectory']]
                calls += len(episode['generations'])
            values = {frozenset(): rewards['none'], frozenset(range(3)): rewards['full']}
            for i in range(3):
                values[frozenset({i})] = rewards[f'only_{i}']
                values[frozenset(range(3)) - {i}] = rewards[f'drop_{i}']
            phi = shapley_three(values)
            if abs(sum(phi) - (rewards['full'] - rewards['none'])) > 1e-9:
                raise ValueError('Shapley efficiency failed')
            units.append(dict(case=case, repeat=repeat, rewards=rewards,
                              loo=[rewards['full'] - rewards[f'drop_{i}'] for i in range(3)],
                              shapley=phi, full_repeat_reward_equal=
                              rewards['full'] == rewards['full_repeat'],
                              full_repeat_actions_equal=
                              actions['full'] == actions['full_repeat'], actor_calls=calls))
    return dict(schema='alf_three_memory_coalition_audit_v1',
                design_sha256=sha(output / 'design.json'),
                expected=len(design['cases']) * len(design['repeats']),
                audited=len(units), missing=missing, units=units,
                caveat='Posthoc fixed-snapshot official train probe; no online Q or writer update')


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
