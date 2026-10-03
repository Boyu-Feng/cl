"""Audit CLBench prefix memory probes and report paired marginal effects."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import read, save, seed, sha
from ttcl.memrl_comparison.credit_probe import memory_arms


MARKER = '\n\nPast experience:\n'


def analyze(output: Path) -> dict:
    design = read(output / 'design.json')
    if 'cl_sampling_repeats' not in design:
        raise ValueError('Expected a CLBench sampling-repeat probe')
    origin = Path(design['origin'])
    if sha(origin / 'plan.json') != design['origin_plan_sha256']:
        raise ValueError('Origin plan changed')
    plan = read(origin / 'plan.json')
    from ttcl.structured_memory import run_benchmark as base
    selection = None
    if 'selection_path' in design:
        freeze_path = output / 'execution_freeze.json'
        if sha(freeze_path) != design['execution_freeze_sha256']:
            raise ValueError('CL probe execution freeze changed')
        for name, expected in read(freeze_path)['files'].items():
            if sha(output / 'source' / name) != expected:
                raise ValueError(f'Frozen CL probe execution source changed: {name}')
        selection_path = Path(design['selection_path'])
        if sha(selection_path) != design['selection_sha256']:
            raise ValueError('Frozen CL prefix selection changed')
        selection = read(selection_path)
        if (selection['origin'] != str(origin) or
                selection['origin_plan_sha256'] != sha(origin / 'plan.json') or
                selection['actor_repeats'] != design['cl_sampling_repeats'] or
                [item['case'] for item in selection['cases']] != design['cases']):
            raise ValueError('Probe design differs from frozen CL prefix selection')
    rows, missing = [], []
    for case in design['cases']:
        spec, arms = memory_arms(origin, case)
        if spec['benchmark'] != 'clbench' or not spec['ids']:
            raise ValueError(f'Expected retrieved CLBench memories: {case}')
        if spec['index'] >= int(.2 * plan['tasks'][spec['task']]):
            raise ValueError(f'Case is outside the training/calibration prefix: {case}')
        if selection is not None:
            item = next(item for item in selection['cases'] if item['case'] == case)
            if (item['memory_ids'] != spec['ids'] or
                    item['source_input_sha256'] != spec['source_input_sha256'] or
                    item['snapshot_sha256'] != spec['snapshot_sha256'] or
                    item['retrieval_sha256'] != spec['retrieval_sha256'] or
                    item['source_row_sha256'] != spec['original_memrl_row_sha256']):
                raise ValueError(f'Frozen CL source changed: {case}')
        source = {key:value for key,value in spec.items()
                  if key not in {'original_memrl', 'original_none', 'original_update'}}
        folder = output / 'clbench' / spec['task'] / str(spec['repeat']) / f"episode_{spec['index']+1:03d}"
        if read(folder / 'source.json') != source:
            raise ValueError(f'Source binding changed: {case}')
        for actor_repeat in design['cl_sampling_repeats']:
            target = folder / f'actor_repeat_{actor_repeat}'
            if not (target / 'summary.json').exists():
                missing.append(dict(case=case, actor_repeat=actor_repeat))
                continue
            summary = read(target / 'summary.json')
            if summary['ids'] != spec['ids'] or summary['actor_repeat'] != actor_repeat:
                raise ValueError(f'Summary identity changed: {target}')
            expected_arms = ['full', 'none']
            if len(spec['ids']) > 1:
                expected_arms.extend(f'drop_{i}' for i in range(len(spec['ids'])))
            if set(summary['replay']) != set(expected_arms):
                raise ValueError(f'Unexpected arms: {target}')
            rewards, seeds = {}, set()
            for arm in expected_arms:
                result = read(target / arm / 'result.json')
                if result != summary['replay'][arm]:
                    raise ValueError(f'Summary differs from official result: {target / arm}')
                response_path = target / arm / 'responses.jsonl'
                lines = response_path.read_text().splitlines()
                if not lines or len(lines) != result['actor_calls']:
                    raise ValueError(f'Actor response record incomplete: {response_path}')
                first = json.loads(lines[0])
                expected_seed = seed(base.generation_seed(
                    plan['task_seed'], first['instance_id'], 1), actor_repeat) % 2**32
                if first['actual_generation_seed'] != expected_seed:
                    raise ValueError(f'Actor seed differs from frozen repeat: {response_path}')
                if hashlib.sha256(first['query'].encode()).hexdigest() != result['query_sha256']:
                    raise ValueError(f'Public query changed: {response_path}')
                system = first['messages'][0]['content']
                if arm == 'none':
                    if MARKER in system:
                        raise ValueError(f'No-memory arm includes context: {response_path}')
                elif MARKER not in system or system.split(MARKER, 1)[1] != arms[arm]:
                    raise ValueError(f'Actual actor context differs: {response_path}')
                seeds.add(first['actual_generation_seed'])
                rewards[arm] = float(result['reward'])
            if len(seeds) != 1:
                raise ValueError(f'Arms did not use the same actor seed: {target}')
            deltas = [rewards['full'] - rewards['none' if len(spec['ids']) == 1
                                                   else f'drop_{i}']
                      for i in range(len(spec['ids']))]
            rows.append(dict(case=case, task=spec['task'], actor_repeat=actor_repeat,
                             memory_ids=spec['ids'], rewards=rewards,
                             delta_by_memory=deltas,
                             delta_0=deltas[0],
                             delta_1=deltas[1] if len(deltas) > 1 else None,
                             interaction=(deltas[0] + deltas[1]
                                          - rewards['full'] + rewards['none'])
                                          if len(deltas) == 2 else None,
                             memory_features=source['memory_features'],
                             memory_text_sha256=source['memory_text_sha256'],
                             source_input_sha256=source['source_input_sha256'],
                             retrieval_sha256=source['retrieval_sha256'],
                             snapshot_sha256=source['snapshot_sha256']))
    second = [row['delta_1'] for row in rows if row['delta_1'] is not None]
    return dict(design_sha256=sha(output / 'design.json'),
                selection_sha256=design.get('selection_sha256'),
                expected=len(design['cases'])*len(design['cl_sampling_repeats']),
                completed=len(rows), missing=missing, rows=rows,
                mean_delta_0=sum(row['delta_0'] for row in rows)/len(rows) if rows else None,
                mean_delta_1=sum(second)/len(second) if second else None,
                caveat='Reward-based diagnostic on selected CLBench calibration-prefix instances; added rollouts and heavy-tailed outcomes limit inference.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.output.resolve())
    save(args.output / 'analysis.json', report)
    print(f"Audited {report['completed']}/{report['expected']} paired comparisons")


if __name__ == '__main__':
    main()
