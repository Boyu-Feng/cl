"""Paired online pilot of vanilla MemRL and contextual utility gated evidence memory."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .credit_online_ablation import ControlledActorClient
from .contextual_utility import ContextualUtilityMemory
from .memory import Memory
from .worker import alf_cell, cl_cell


def evaluate(origin, output, gate_path, benchmark, task, repeat, url, limit,
             temperature, alf_split=None, task_seed=None):
    plan = read(origin / 'plan.json')
    if benchmark == 'alfworld':
        if alf_split is None:
            matches = [s for s in plan['alf']['sequences']
                       if s['family'] == task and s['repeat'] == repeat]
            if len(matches) != 1:
                raise ValueError('Unknown ALFWorld task or repeat')
            items = matches[0]['tasks']
        else:
            if alf_split != 'valid_seen':
                raise ValueError('Unsupported ALFWorld split')
            root = Path(plan['alf']['data_root'])
            games = sorted((root / 'json_2.1.1' / alf_split).glob(
                f'{task}-*/trial_*/game.tw-pddl'))
            if not games:
                raise ValueError('No games for ALFWorld task/split')
            prior = {x['sha256'] for sequence in plan['alf']['sequences']
                     for x in sequence['tasks']}
            items = [dict(path=str(game.relative_to(root)), sha256=sha(game),
                          family=task, split=alf_split) for game in games]
            if any(item['sha256'] in prior for item in items):
                raise ValueError('Selected split overlaps the training plan')
        bindings = [x['sha256'] for x in items]
    else:
        if task not in plan['tasks'] or repeat not in plan['repeats']:
            raise ValueError('Unknown CLBench task or repeat')
        items = list(range(plan['tasks'][task]))
        os.environ.setdefault('TTCL_BENCH', str(origin / 'source' / 'bench'))
        os.chdir(os.environ['TTCL_BENCH'])
        if task_seed is None:
            bindings = [read(origin / 'runs' / 'clbench' / task / str(repeat) / 'none' /
                             f'episode_{i+1:03d}' / 'row.json')['initial_query_sha256']
                        for i in items]
        else:
            from ttcl.icl_mem0_comparison.worker import make_task
            import hashlib
            plan['task_seed'] = task_seed
            bindings = []
            for index in items:
                instance = make_task(task, task_seed)
                try:
                    query = instance.reset_baseline_instance(index)
                    bindings.append(hashlib.sha256(query.prompt.encode()).hexdigest())
                finally:
                    connection = getattr(instance, '_conn', None)
                    if connection is not None:
                        connection.close()
    if limit is not None:
        if limit < 1 or limit > len(items):
            raise ValueError('Invalid limit')
        items, bindings = items[:limit], bindings[:limit]
    plan['url'] = url
    plan['alf']['actor_url'] = url
    plan['alf']['actor_temperature'] = temperature
    plan['q_min_threshold'] = plan['q_min_thresholds'][benchmark]
    implementation = Path(__file__).with_name('contextual_utility.py')
    design = dict(origin=str(origin), origin_plan_sha256=sha(origin / 'plan.json'),
                  benchmark=benchmark, task=task, repeat=repeat, limit=len(items),
                  url=url, temperature=temperature, bindings=bindings,
                  alf_split=alf_split, task_seed=task_seed,
                  implementation_sha256=sha(implementation), base_implementation_sha256=sha(Path(__file__).with_name('general_evidence.py')),
                  gate_sha256=sha(gate_path), runner_sha256=sha(Path(__file__)),
                  policies=['vanilla', 'contextual_utility'],
                  note='Independent empty-to-online chains; paired actor seed; identical completions cached')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'design.json').exists() and read(output / 'design.json') != design:
        raise ValueError('Frozen design differs; use another output directory')
    save(output / 'design.json', design)
    source = output / 'source'
    source.mkdir(exist_ok=True)
    for name in ('contextual_utility.py', 'general_evidence.py', 'evaluate_contextual_utility.py',
                 'credit_online_ablation.py', 'worker.py', 'memory.py'):
        source_file = Path(__file__).with_name(name)
        frozen = source / name
        if frozen.exists():
            if sha(frozen) != sha(source_file):
                raise ValueError(f'Frozen source changed: {name}')
        else:
            shutil.copy2(source_file, frozen)
    frozen_gate = source / 'gate.json'
    if frozen_gate.exists():
        if sha(frozen_gate) != sha(gate_path):
            raise ValueError('Frozen gate changed')
    else:
        shutil.copy2(gate_path, frozen_gate)
    client = ControlledActorClient(plan, repeat, temperature)
    cal = plan['calibration']['alfworld' if benchmark == 'alfworld' else task]
    memories = {'vanilla': Memory(plan, client, output / 'vanilla' / 'memory', cal)}
    memories['contextual_utility'] = ContextualUtilityMemory(
        plan, client, output / 'contextual_utility' / 'memory', cal, frozen_gate,
        embedder=memories['vanilla'].service.embedding_provider)
    rows = []
    for index, item in enumerate(items):
        for policy, memory in memories.items():
            target = output / policy / f'episode_{index+1:03d}'
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
    p.add_argument('--gate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--benchmark', choices=['alfworld', 'clbench'], required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--repeat', type=int, required=True)
    p.add_argument('--url', required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--temperature', type=float, default=.7)
    p.add_argument('--alf-split', choices=['valid_seen'],
                   help='Run a disjoint ALFWorld split instead of the frozen plan sequence')
    p.add_argument('--task-seed', type=int,
                   help='Alternate CLBench seed; frozen task corpora may remain identical')
    a = p.parse_args()
    evaluate(a.origin.resolve(), a.output.resolve(), a.gate.resolve(), a.benchmark, a.task,
             a.repeat, a.url, a.limit, a.temperature, a.alf_split, a.task_seed)


if __name__ == '__main__':
    main()
