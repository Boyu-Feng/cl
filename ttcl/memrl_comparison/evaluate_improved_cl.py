"""Paired online CLBench evaluation of native MemRL and public-evidence MemRL.

Each policy has its own empty-to-online memory chain. Identical actor requests
share a cached response; the extension can change context and final actions.
Results are development diagnostics on the existing CLBench task sequence.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil

from ttcl.icl_mem0_comparison.protocol import Client, read, save, seed, sha
from .improved_cl import ImprovedCLMemory
from .memory import Memory
from .worker import cl_cell


class PairedClient(Client):
    def __init__(self, plan, repeat, temperature):
        super().__init__(plan, repeat)
        self.temperature = temperature
        self.cache = {}
        self.cache_hits = 0

    def complete(self, messages, random_seed, *, tokens=4096, temperature=.7, top_p=.9):
        key = (json.dumps(messages, ensure_ascii=False, sort_keys=True), int(random_seed),
               tokens, temperature, top_p)
        if key in self.cache:
            self.cache_hits += 1
            return copy.deepcopy(self.cache[key])
        result = super().complete(messages, random_seed, tokens=tokens,
                                  temperature=temperature, top_p=top_p)
        self.cache[key] = copy.deepcopy(result)
        return result

    def generate(self, messages, random_seed):
        return self.complete(messages, seed(random_seed, self.repeat),
                             temperature=self.temperature,
                             top_p=.9 if self.temperature else 1.)


def evaluate(origin, output, task, repeat, url, limit, temperature):
    plan = read(origin / 'plan.json')
    if task not in {'blind_spectrum_monitoring', 'cohort_studies'}:
        raise ValueError(task)
    if repeat not in plan['repeats']:
        raise ValueError(repeat)
    count = plan['tasks'][task]
    if limit is None:
        limit = count
    if limit < 1 or limit > count:
        raise ValueError(limit)
    os.environ.setdefault('TTCL_BENCH', str(origin / 'source' / 'bench'))
    os.chdir(os.environ['TTCL_BENCH'])
    input_hashes = [read(origin / 'runs' / 'clbench' / task / str(repeat) / 'none' /
                         f'episode_{i+1:03d}' / 'row.json')['initial_query_sha256']
                    for i in range(limit)]
    design = dict(origin=str(origin), origin_plan_sha256=sha(origin / 'plan.json'),
                  origin_memory_sha256=sha(origin / 'source' / 'ttcl' / 'memrl_comparison' / 'memory.py'),
                  implementation_sha256=sha(Path(__file__).with_name('improved_cl.py')),
                  runner_sha256=sha(Path(__file__)), task=task, repeat=repeat, limit=limit,
                  temperature=temperature, url=url, input_hashes=input_hashes,
                  policies=['vanilla', 'public_evidence'],
                  note='Online empty-memory chains; original CL task seed and score; identical actor requests share completions')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'design.json').exists() and read(output / 'design.json') != design:
        raise ValueError('Existing design differs; use a new output directory')
    save(output / 'design.json', design)
    source_dir = output / 'source'
    source_dir.mkdir(exist_ok=True)
    for name in ('improved_cl.py', 'evaluate_improved_cl.py', 'worker.py', 'memory.py'):
        original = Path(__file__).with_name(name)
        frozen = source_dir / name
        if frozen.exists():
            if sha(frozen) != sha(original):
                raise ValueError(f'Frozen evaluator source changed: {name}')
        else:
            shutil.copy2(original, frozen)
    plan['url'] = url
    plan['q_min_threshold'] = plan['q_min_thresholds']['clbench']
    client = PairedClient(plan, repeat, temperature)
    cal = plan['calibration'][task]
    memories = {}
    memories['vanilla'] = Memory(plan, client, output / 'vanilla' / 'memory', cal)
    memories['public_evidence'] = ImprovedCLMemory(
        plan, client, output / 'public_evidence' / 'memory', cal, task,
        embedder=memories['vanilla'].service.embedding_provider)
    rows = []
    for i in range(limit):
        for policy, memory in memories.items():
            target = output / policy / f'episode_{i+1:03d}'
            if (target / 'row.json').exists():
                row = read(target / 'row.json')
                if row['status'] != 'complete' or sha(target / 'memory_after.json') != row['memory_after_sha256']:
                    raise ValueError(f'Cannot resume failed or changed cell: {target}')
                memory.restore(target / 'memory_after.json')
            else:
                row = cl_cell(plan, client, memory, task, i, 'memrl', target)
            if row['initial_query_sha256'] != input_hashes[i]:
                raise ValueError(f'Paired input changed: {task} {repeat} {i+1}')
            rows.append(dict(policy=policy, index=i, status=row['status'], reward=row['reward'],
                             writer_token_limit_hits=row['writer_token_limit_hits']))
            save(output / 'progress.json', dict(completed=len(rows), expected=2 * limit,
                                                cache_hits=client.cache_hits,
                                                mean_rewards={p:sum(r['reward'] for r in rows
                                                                    if r['policy'] == p and r['reward'] is not None) /
                                                              max(1, sum(r['policy'] == p and r['reward'] is not None
                                                                         for r in rows)) for p in memories}))
            print(json.dumps(rows[-1]), flush=True)
    save(output / 'rows.json', rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--origin', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--task', choices=['blind_spectrum_monitoring', 'cohort_studies'], required=True)
    p.add_argument('--repeat', type=int, required=True)
    p.add_argument('--url', required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--temperature', type=float, default=.7)
    a = p.parse_args()
    evaluate(a.origin.resolve(), a.output.resolve(), a.task, a.repeat, a.url, a.limit, a.temperature)


if __name__ == '__main__':
    main()
