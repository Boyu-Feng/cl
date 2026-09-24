from __future__ import annotations

from collections import defaultdict
import random
import statistics
from pathlib import Path

from ttcl.experience_evolution.core import read, save


def paired_interval(rows, arm, baseline='retry_none', repeats=2000, metric='success'):
    """Bootstrap task clusters while preserving their observed decoding repeats."""
    if repeats < 1:
        raise ValueError('Bootstrap repeats must be positive')
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row['game'], row['repeat'])][row['arm']] = row
    tasks = defaultdict(list)
    for (game, _), pair in grouped.items():
        if arm in pair and baseline in pair:
            tasks[game].append(pair[arm][metric] - pair[baseline][metric])
    clusters = [values for _, values in sorted(tasks.items())]
    if not clusters:
        return None
    rng = random.Random(926)
    samples = []
    for _ in range(repeats):
        chosen = rng.choices(clusters, k=len(clusters))
        samples.append(sum(sum(values) for values in chosen) / sum(map(len, chosen)))
    samples.sort()
    values = [value for cluster in clusters for value in cluster]
    return {'mean_difference': statistics.mean(values), 'task_clusters': len(clusters),
            'paired_count': len(values), 'bootstrap_repeats': repeats,
            'interval_95': [samples[int(.025*repeats)], samples[min(repeats-1, int(.975*repeats))]],
            'unit': 'task, with sampling seeds kept together; not independent environments'}


def summarize(rows, arms, final=False):
    if not arms or len(set(arms)) != len(arms):
        raise ValueError('Expected a nonempty set of distinct methods')
    groups = {arm: {} for arm in arms}
    for row in rows:
        if row['arm'] not in groups:
            raise ValueError(f"Unknown result method: {row['arm']}")
        if row['status'] == 'complete':
            key = (row['game'], row['repeat'])
            if key in groups[row['arm']]:
                raise ValueError(f"Duplicate result cell: {row['arm']} {key}")
            groups[row['arm']][key] = row
    common = set.intersection(*(set(g) for g in groups.values()))
    result = {'common_task_seed_pairs': len(common), 'arms': {}}
    if not common:
        return result
    paired_rows = [group[key] for group in groups.values() for key in common]
    cost_fields = ['actor_calls', 'physical_actor_calls', 'actor_input_tokens',
                   'actor_output_tokens', 'physical_actor_input_tokens',
                   'physical_actor_output_tokens', 'actor_seconds', 'physical_actor_seconds',
                   'writer_calls', 'writer_input_tokens', 'writer_output_tokens',
                   'writer_seconds', 'invalid_commands']
    for arm, group in groups.items():
        selected = [group[k] for k in sorted(common)]
        value = {'n': len(selected), 'successes': sum(r['success'] for r in selected),
                 'success_rate': statistics.mean(r['success'] for r in selected),
                 'first_successes': sum(r['first_success'] for r in selected),
                 'first_success_rate': statistics.mean(r['first_success'] for r in selected),
                 'mean_attempts': statistics.mean(r['attempts'] for r in selected),
                 'mean_steps': statistics.mean(r['actor_calls'] for r in selected)}
        value.update({field: sum(r.get(field, 0) for r in selected) for field in cost_fields})
        value['by_seed'] = {str(s): {'n': len(part), 'successes': sum(r['success'] for r in part),
                                    'first_successes': sum(r['first_success'] for r in part)}
                            for s in sorted({r['repeat'] for r in selected})
                            if (part := [r for r in selected if r['repeat'] == s])}
        value['by_family'] = {family: {'n': len(part),
                                      'success_rate': statistics.mean(r['success'] for r in part),
                                      'first_success_rate': statistics.mean(r['first_success'] for r in part)}
                              for family in sorted({r['family'] for r in selected})
                              if (part := [r for r in selected if r['family'] == family])}
        value['macro_family_success'] = statistics.mean(x['success_rate'] for x in value['by_family'].values())
        if final:
            value['paired_vs_retry'] = paired_interval(paired_rows, arm)
            value['first_paired_vs_none'] = paired_interval(paired_rows, arm, metric='first_success')
        result['arms'][arm] = value
    baseline = result['arms'].get('retry_none')
    if baseline is not None:
        result['single_attempt_none'] = {'n': baseline['n'],
            'successes': baseline['first_successes'], 'success_rate': baseline['first_success_rate'],
            'source': 'retry_none attempt 1; no additional physical execution'}
    return result


def load_rows(root, plan):
    """Read the declared grid only, validating every result's task identity."""
    expected, rows = set(), []
    for sequence in plan['sequences']:
        for position, task in enumerate(sequence['tasks']):
            for arm in plan['arms']:
                path = root/'evaluation'/sequence['family']/str(sequence['repeat'])/f'task_{position:03}'/arm/'result.json'
                if path in expected:
                    raise ValueError('Duplicate planned evaluation cell')
                expected.add(path)
                if not path.exists():
                    continue
                row = read(path)
                identity = {'arm': arm, 'family': sequence['family'], 'repeat': sequence['repeat'],
                            'position': position, 'game': task['path'],
                            'never_evaluated': task['never_evaluated']}
                if any(row.get(key) != value for key, value in identity.items()):
                    raise ValueError(f'Result does not match the planned evaluation cell: {path}')
                if row.get('status') != 'complete':
                    raise ValueError(f'Unfinished row cannot be counted as a completed cell: {path}')
                if row['first_success'] not in (0, 1) or row['success'] not in (0, 1):
                    raise ValueError(f'Nonbinary success in {path}')
                if not 1 <= row['attempts'] <= plan['max_attempts']:
                    raise ValueError(f'Invalid attempt count in {path}')
                rows.append(row)
    if len(expected) != plan['expected_cells']:
        raise ValueError('Expected cell count does not match the declared grid')
    observed = set((root/'evaluation').glob('*/*/*/*/result.json'))
    if observed - expected:
        raise ValueError('Unplanned result files found in the evaluation tree')
    return rows


def report(root, final=False):
    root = Path(root)
    plan = read(root/'plan.json')
    rows = load_rows(root, plan)
    if final and len(rows) != plan['expected_cells']:
        raise RuntimeError('Cannot finalize an incomplete evaluation grid')
    summary = {'completed_cells': len(rows), 'expected_cells': plan['expected_cells'],
               'all': summarize(rows, plan['arms'], final),
               'never_previously_evaluated': summarize([r for r in rows if r['never_evaluated']], plan['arms'], final),
               'after_first_in_family_chain': summarize([r for r in rows if r['position'] > 0], plan['arms'], final),
               'interpretation': 'First attempt belongs to the same online three-attempt protocol; previous tasks may have used retries. Logical costs include documented shared executions; physical actor costs count their executing arm only.'}
    bank_path = root/'expel_bank/state.json'
    if bank_path.exists():
        summary['offline_expel_development'] = read(bank_path).get('development_cost', {})
    save(root/'summary.json', summary)
    save(root/'evaluation_status.json', {'phase': 'complete' if final else 'running',
                                        'completed': len(rows), 'expected': plan['expected_cells']})
    lines = ['# Matched ALFWorld comparison', '',
             f"Completed method/task/seed cells: {len(rows)}/{plan['expected_cells']}.", '',
             'All methods use an identical frozen actor, task resets, action limits, sampling seeds, and the same maximum attempts. Offline development and memory mechanisms differ and are disclosed in PROTOCOL.md.',
             'The no-experience single-attempt control is retry_none in the first-success column.', '']
    for name in ['all', 'never_previously_evaluated', 'after_first_in_family_chain']:
        value = summary[name]
        lines += [f"## {name}: {value['common_task_seed_pairs']} common pairs", '',
                  f"| method | first success | success within {plan['max_attempts']} | mean attempts | mean actions |",
                  '|---|---:|---:|---:|---:|']
        for arm, item in value['arms'].items():
            lines.append(f"| {arm} | {item['first_successes']}/{item['n']} | {item['successes']}/{item['n']} | {item['mean_attempts']:.3f} | {item['mean_steps']:.2f} |")
        lines.append('')
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return summary
