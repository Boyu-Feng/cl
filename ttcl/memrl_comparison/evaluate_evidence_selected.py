"""Paired native MemRL versus online model-selected raw evidence memory."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .credit_online_ablation import ControlledActorClient
from .evidence_selected_memory import EvidenceSelectedMemory
from .memory import Memory
from .worker import alf_cell, cl_cell


def run(origin, output, benchmark, task, repeat, url, limit, alf_source, temperature):
    plan = read(origin / 'plan.json')
    if benchmark == 'alfworld':
        if alf_source:
            source = read(alf_source)
            if source['repeat'] != repeat or task not in source['selected_games']:
                raise ValueError('ALF source selection does not match family or repeat')
            items = source['selected_games'][task]
        else:
            seq = [s for s in plan['alf']['sequences']
                   if s['family'] == task and s['repeat'] == repeat]
            if len(seq) != 1:
                raise ValueError('Unknown ALF sequence')
            items = seq[0]['tasks']
        if limit:
            items = items[:limit]
        bindings = [item['sha256'] for item in items]
    else:
        if alf_source:
            raise ValueError('ALF source selection is ALF-only')
        if task not in plan['tasks'] or repeat not in plan['repeats']:
            raise ValueError('Unknown CL sequence')
        items = list(range(plan['tasks'][task]))
        if limit:
            items = items[:limit]
        bindings = [read(origin / 'runs' / 'clbench' / task / str(repeat) / 'none' /
                         f'episode_{i+1:03d}' / 'row.json')['initial_query_sha256']
                    for i in items]
        os.environ.setdefault('TTCL_BENCH', str(origin / 'source' / 'bench'))
        os.chdir(os.environ['TTCL_BENCH'])
    plan['url'] = url.rstrip('/')
    plan['alf']['actor_url'] = url.rstrip('/')
    plan['alf']['actor_temperature'] = temperature
    plan['q_min_threshold'] = plan['q_min_thresholds'][benchmark]
    design = dict(schema='online_raw_evidence_pair_v1',
        origin=str(origin), plan_sha256=sha(origin / 'plan.json'),
        benchmark=benchmark, task=task, repeat=repeat, input_bindings=bindings,
        source_design=str(alf_source) if alf_source else None,
        source_design_sha256=sha(alf_source) if alf_source else None,
        url=url.rstrip('/'), temperature=temperature,
        arms=['memrl', 'evidence_selected'],
        runner_sha256=sha(Path(__file__)),
        memory_sha256=sha(Path(__file__).with_name('evidence_selected_memory.py')),
        budget='Original task and actor budget; evidence selection at most one 256-token model call per positive-reward source attempt')
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / 'design.json'
    if frozen.is_file() and read(frozen) != design:
        raise ValueError('Frozen pair design changed')
    save(frozen, design)
    source_dir = output / 'source'
    source_dir.mkdir(exist_ok=True)
    for name in ('evidence_selected_memory.py', 'evaluate_evidence_selected.py',
                 'memory.py', 'worker.py', 'credit_online_ablation.py'):
        original = Path(__file__).with_name(name)
        target = source_dir / name
        if target.exists() and sha(target) != sha(original):
            raise ValueError(f'Frozen source changed: {name}')
        if not target.exists():
            shutil.copy2(original, target)
    client = ControlledActorClient(plan, repeat, temperature)
    calibration = plan['calibration']['alfworld' if benchmark == 'alfworld' else task]
    native = Memory(plan, client, output / 'memrl' / 'memory', calibration)
    evidence = EvidenceSelectedMemory(plan, client, output / 'evidence_selected' / 'memory',
                                      calibration, embedder=native.service.embedding_provider)
    arms = {'memrl': native, 'evidence_selected': evidence}
    rows = []
    for index, item in enumerate(items):
        for arm, memory in arms.items():
            target = output / arm / f'episode_{index+1:03d}'
            if (target / 'row.json').is_file():
                record = read(target / 'row.json')
                if sha(target / 'memory_after.json') != record['memory_after_sha256']:
                    raise ValueError('Completed cell snapshot changed')
                memory.restore(target / 'memory_after.json')
            else:
                record = (alf_cell(plan, client, memory, item, repeat, arm, target)
                          if benchmark == 'alfworld' else
                          cl_cell(plan, client, memory, task, item, arm, target))
            if record['status'] != 'complete' or record.get('memory_update_status') == 'failed':
                raise RuntimeError(record.get('error', 'Unscored benchmark cell'))
            if record['input_sha256' if benchmark == 'alfworld' else 'initial_query_sha256'] != bindings[index]:
                raise ValueError('Input content changed')
            row = dict(index=index, arm=arm, input_sha256=bindings[index],
                reward=record['reward'], first_attempt=record.get('first_attempt'),
                attempts=record.get('attempts'), actor_calls=record['actor_calls'],
                selection_calls=evidence.selection_calls if arm == 'evidence_selected' else 0)
            rows.append(row)
            save(output / 'progress.json', dict(completed=len(rows), expected=len(items)*2,
                response_cache_hits=client.response_cache_hits))
            print(json.dumps(row), flush=True)
    save(output / 'rows.json', rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--benchmark', choices=['alfworld', 'clbench'], required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--repeat', type=int, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18559')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--alf-source', type=Path)
    parser.add_argument('--temperature', type=float, default=.7)
    args = parser.parse_args()
    run(args.origin.resolve(), args.output.resolve(), args.benchmark, args.task,
        args.repeat, args.url, args.limit,
        args.alf_source.resolve() if args.alf_source else None, args.temperature)


if __name__ == '__main__':
    main()
