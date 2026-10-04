"""Audit task-conditioned sign changes for exactly identical memory text.

Only previously reviewed ALFWorld train coalition probes are read. The
robust Shapley sign requires two stable actor seeds of one sign and none of
the opposite sign. A repeated memory can have different co-retrieved peers.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import statistics

from ttcl.icl_mem0_comparison.protocol import read, sha
from .audit_alf_coalition_credit import audit as audit_coalitions
from .audit_credit_q_opportunity import audit as audit_opportunities
from .credit_probe import memory_arms


def audit(outputs: list[Path]) -> dict:
    opportunities = audit_opportunities(outputs)
    groups = defaultdict(list)
    for source in opportunities['outputs']:
        output = Path(source['output'])
        origin = Path(read(output / 'design.json')['origin'])
        audited = audit_coalitions(output)
        units = defaultdict(list)
        for unit in audited['units']:
            units[unit['case']].append(unit)
        for case in source['cases']:
            spec, _ = memory_arms(origin, case['case'])
            if spec['source_input_sha256'] != case['input_sha256']:
                raise ValueError('Input binding changed')
            source_row = origin / 'runs' / case['case'] / 'row.json'
            row = read(source_row)
            if row['input_sha256'] != case['input_sha256']:
                raise ValueError('Official source input changed')
            initial = (origin / 'runs' / case['case'] / 'attempt_1' /
                       'episode.json')
            first_observation = read(initial)['initial_observation']
            if 'Your task is to: ' not in first_observation:
                raise ValueError('Public ALFWorld goal missing')
            game_sha256 = sha(Path(read(origin / 'plan.json')['alf']['data_root']) /
                              row['game'])
            if game_sha256 != case['input_sha256']:
                raise ValueError('Public game content changed')
            for item in case['memories']:
                i = item['memory_index']
                mid = item['memory_id']
                if (spec['ids'][i] != mid or
                        spec['memory_text_sha256'][mid] !=
                        item['memory_text_sha256']):
                    raise ValueError('Memory identity or text changed')
                stable_units = [unit for unit in units[case['case']]
                                if unit['full_repeat_reward_equal']]
                stable = [unit['shapley'][i] for unit in stable_units]
                solo_effect = [unit['rewards'][f'only_{i}'] -
                               unit['rewards']['none']
                               for unit in stable_units]
                full_marginal = [unit['rewards']['full'] -
                                 unit['rewards'][f'drop_{i}']
                                 for unit in stable_units]
                sign = item['robust_shapley_sign']
                if sign and (sum(value * sign > 0 for value in stable) < 2 or
                             any(value * sign < 0 for value in stable)):
                    raise ValueError('Robust sign lacks seed support')
                groups[item['memory_text_sha256']].append(dict(
                    output=str(output), coalition_design_sha256=sha(output / 'design.json'),
                    source_design_sha256=sha(origin / 'design.json'),
                    case=case['case'], input_sha256=case['input_sha256'],
                    game=row['game'], game_sha256=game_sha256,
                    initial_observation_sha256=hashlib.sha256(
                        first_observation.encode()).hexdigest(),
                    public_task=first_observation.split('Your task is to: ', 1)[-1]
                                .split('\n', 1)[0],
                    memory_id=mid, memory_index=i,
                    companion_text_sha256=[spec['memory_text_sha256'][other]
                                           for other in spec['ids'] if other != mid],
                    stable_shapley=stable,
                    stable_solo_effect=solo_effect,
                    stable_full_marginal=full_marginal,
                    stable_shapley_mean=(statistics.fmean(stable)
                                         if stable else None),
                    robust_shapley_sign=sign))
    repeated = []
    for text_hash, instances in sorted(groups.items()):
        if len({row['input_sha256'] for row in instances}) < 2:
            continue
        signs = {row['robust_shapley_sign'] for row in instances}
        repeated.append(dict(memory_text_sha256=text_hash,
                             instances=instances,
                             has_two_nonzero=sum(sign != 0 for sign in
                                                 (row['robust_shapley_sign']
                                                  for row in instances)) >= 2,
                             opposite_signs=(-1 in signs and 1 in signs)))
    nonzero_pairs = [(a, b) for item in repeated
                     for a, b in itertools.combinations(item['instances'], 2)
                     if (a['input_sha256'] != b['input_sha256'] and
                         a['robust_shapley_sign'] and b['robust_shapley_sign'])]
    return dict(schema='alf_credit_sign_transfer_audit_v1',
                script_sha256=sha(Path(__file__)),
                opportunity_script_sha256=opportunities['script_sha256'],
                outputs=[dict(path=str(output), design_sha256=sha(output / 'design.json'))
                         for output in outputs],
                repeated_memories=repeated,
                totals=dict(cases=opportunities['totals']['cases'],
                            repeated_exact_text=len(repeated),
                            repeated_with_two_nonzero=sum(item['has_two_nonzero']
                                                          for item in repeated),
                            repeated_with_opposite_signs=sum(item['opposite_signs']
                                                             for item in repeated),
                            nonzero_cross_input_pairs=len(nonzero_pairs),
                            opposite_nonzero_pairs=sum(a['robust_shapley_sign'] !=
                                                       b['robust_shapley_sign']
                                                       for a, b in nonzero_pairs)),
                caveat='Selected development inputs; same text can be paired with different tasks and companion memories; sign flips refute a universal sign for these instances but do not isolate the cause or estimate a population rate')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, nargs='+', required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = audit([path.resolve() for path in args.outputs])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['totals']))


if __name__ == '__main__':
    main()
