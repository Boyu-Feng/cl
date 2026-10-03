"""Paired online pilot of vanilla MemRL and typed source-grounded public evidence."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .credit_online_ablation import ControlledActorClient
from .grounded_evidence_v19 import TypedGroundedMemory
from .memory import Memory
from .worker import alf_cell, cl_cell


def evaluate(origin, output, benchmark, task, repeat, url, limit, temperature):
    plan = read(origin / 'plan.json')
    if benchmark == 'alfworld':
        matches = [s for s in plan['alf']['sequences']
                   if s['family'] == task and s['repeat'] == repeat]
        if len(matches) != 1:
            raise ValueError('Unknown ALFWorld task or repeat')
        items = matches[0]['tasks']
        bindings = [x['sha256'] for x in items]
    else:
        if task not in plan['tasks'] or repeat not in plan['repeats']:
            raise ValueError('Unknown CLBench task or repeat')
        items = list(range(plan['tasks'][task]))
        bindings = [read(origin / 'runs' / 'clbench' / task / str(repeat) / 'none' /
                         f'episode_{i+1:03d}' / 'row.json')['initial_query_sha256']
                    for i in items]
        os.environ.setdefault('TTCL_BENCH', str(origin / 'source' / 'bench'))
        os.chdir(os.environ['TTCL_BENCH'])
    if limit is not None:
        if limit < 1 or limit > len(items):
            raise ValueError('Invalid limit')
        items, bindings = items[:limit], bindings[:limit]
    plan['url'] = url
    plan['alf']['actor_url'] = url
    plan['alf']['actor_temperature'] = temperature
    plan['q_min_threshold'] = plan['q_min_thresholds'][benchmark]
    implementation = Path(__file__).with_name('grounded_evidence_v19.py')
    design = dict(origin=str(origin), origin_plan_sha256=sha(origin / 'plan.json'),
                  benchmark=benchmark, task=task, repeat=repeat, limit=len(items),
                  url=url, temperature=temperature, bindings=bindings,
                  implementation_sha256=sha(implementation), runner_sha256=sha(Path(__file__)),
                  policies=['vanilla', 'typed_grounded'],
                  note='Independent empty-to-online chains; paired actor seed; identical completions cached')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'design.json').exists() and read(output / 'design.json') != design:
        raise ValueError('Frozen design differs; use another output directory')
    save(output / 'design.json', design)
    source = output / 'source'
    source.mkdir(exist_ok=True)
    for name in ('grounded_evidence_v13_base.py', 'grounded_evidence_v19.py',
                 'typed_projection.py', 'typed_executor.py',
                 'evaluate_typed_grounded_v19.py',
                 'credit_online_ablation.py', 'worker.py', 'memory.py'):
        source_file = Path(__file__).with_name(name)
        frozen = source / name
        if frozen.exists():
            if sha(frozen) != sha(source_file):
                raise ValueError(f'Frozen source changed: {name}')
        else:
            shutil.copy2(source_file, frozen)
    client = ControlledActorClient(plan, repeat, temperature)
    cal = plan['calibration']['alfworld' if benchmark == 'alfworld' else task]
    memories = {'vanilla': Memory(plan, client, output / 'vanilla' / 'memory', cal)}
    memories['typed_grounded'] = TypedGroundedMemory(
        plan, client, output / 'typed_grounded' / 'memory', cal,
        embedder=memories['vanilla'].service.embedding_provider)
    rows = []
    for index, item in enumerate(items):
        for policy, memory in memories.items():
            target = output / policy / f'episode_{index+1:03d}'
            if benchmark == 'alfworld' and hasattr(memory, 'begin_task'):
                memory.begin_task('retryable_text_command', bindings[index])
            if (target / 'row.json').exists():
                row = read(target / 'row.json')
                if sha(target / 'memory_after.json') != row['memory_after_sha256']:
                    raise ValueError(f'Completed cell changed: {target}')
                memory.restore(target / 'memory_after.json')
            else:
                row = (alf_cell(plan, client, memory, item, repeat, 'memrl', target)
                       if benchmark == 'alfworld' else
                       cl_cell(plan, client, memory, task, item, 'memrl', target))
            if benchmark == 'alfworld':
                if row['input_sha256'] != bindings[index]:
                    raise ValueError('ALFWorld input changed')
            elif row['initial_query_sha256'] != bindings[index]:
                raise ValueError('CLBench input changed')
            if row['status'] != 'complete' and not (benchmark == 'clbench' and
                any(x in row.get('error', '') for x in
                    ('no schema-valid JSON action', 'Safety cap exceeded', 'Context overflow'))):
                raise RuntimeError(row.get('error', 'Unscored infrastructure failure'))
            row_summary = dict(policy=policy, index=index, status=row['status'],
                               reward=row.get('reward'), first_attempt=row.get('first_attempt'),
                               within_three=row.get('within_three'),
                               game=item['path'] if benchmark == 'alfworld' else None)
            rows.append(row_summary)
            save(output / 'progress.json', dict(completed=len(rows), expected=2 * len(items),
                                                cache_hits=client.response_cache_hits))
            print(json.dumps(row_summary), flush=True)
    save(output / 'rows.json', rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--origin', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--benchmark', choices=['alfworld', 'clbench'], required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--repeat', type=int, required=True)
    p.add_argument('--url', required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--temperature', type=float, default=.7)
    a = p.parse_args()
    evaluate(a.origin.resolve(), a.output.resolve(), a.benchmark, a.task,
             a.repeat, a.url, a.limit, a.temperature)


if __name__ == '__main__':
    main()
