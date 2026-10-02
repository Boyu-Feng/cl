"""Online ALFWorld credit-policy ablation with the original frozen MemRL run plan.

Two independent memory chains see the same games and actor seeds. Both inject
the original top-three context and build memories normally. Only the Q reward
assignment differs: native all-selected versus first-selected only.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import Client, read, save, seed, sha
from ttcl.memrl_comparison.memory import Memory
from ttcl.memrl_comparison.worker import alf_cell, cl_cell


class FirstOnlyCreditMemory(Memory):
    def update(self, query, public_trace, reward, success, retrieval, binding):
        ids = retrieval['ids']
        if len(ids) < 2:
            result = super().update(query, public_trace, reward, success, retrieval, binding)
            result['credited_ids'] = ids[:]
            return result
        original = self.service.update_value
        remaining = len(ids)
        first = ids[0]

        def selective_update(mid, value):
            nonlocal remaining
            remaining -= 1
            if remaining == 0:
                self.service.update_value = original
            if mid == first:
                return original(mid, value)
            return float(self.service._q_cache.get(mid, self.store.get(mid).metadata.q_value))

        self.service.update_value = selective_update
        try:
            result = super().update(query, public_trace, reward, success, retrieval, binding)
        finally:
            self.service.update_value = original
        if remaining != 0:
            raise RuntimeError('Unexpected number of Q update calls')
        result['credited_ids'] = [first]
        return result


class ControlledActorClient(Client):
    def __init__(self, plan, repeat, actor_temperature):
        super().__init__(plan, repeat)
        self.actor_temperature = actor_temperature
        self.response_cache = {}
        self.response_cache_hits = 0

    def complete(self, messages, random_seed, *, tokens=4096, temperature=.7, top_p=.9):
        key = (json.dumps(messages, ensure_ascii=False, sort_keys=True),
               int(random_seed), tokens, temperature, top_p)
        if key in self.response_cache:
            self.response_cache_hits += 1
            return copy.deepcopy(self.response_cache[key])
        result = super().complete(messages, random_seed, tokens=tokens,
                                  temperature=temperature, top_p=top_p)
        self.response_cache[key] = copy.deepcopy(result)
        return result

    def generate(self, messages, random_seed):
        return self.complete(messages, seed(random_seed, self.repeat),
                             temperature=self.actor_temperature,
                             top_p=.9 if self.actor_temperature else 1.)


def run(origin: Path, output: Path, benchmark: str, family: str, repeat: int,
        url: str, limit: int | None, actor_temperature: float):
    plan = read(origin / 'plan.json')
    if benchmark == 'alfworld':
        sequences = [s for s in plan['alf']['sequences'] if s['family'] == family and s['repeat'] == repeat]
        if len(sequences) != 1:
            raise ValueError('Expected exactly one family/seed sequence')
        tasks = sequences[0]['tasks'][:limit]
    else:
        if family not in plan['tasks'] or repeat not in plan['repeats']:
            raise ValueError('Unknown CLBench domain/repeat')
        tasks = list(range(plan['tasks'][family]))[:limit]
        os.chdir(os.environ['TTCL_BENCH'])
    plan['url'] = url
    plan['alf']['actor_url'] = url
    plan['alf']['actor_temperature'] = actor_temperature
    plan['q_min_threshold'] = plan['q_min_thresholds'][benchmark]
    design = dict(origin=str(origin), origin_plan_sha256=sha(origin / 'plan.json'),
                  benchmark=benchmark, family=family, repeat=repeat, limit=limit,
                  input_bindings=[t['sha256'] for t in tasks] if benchmark == 'alfworld' else
                                 [read(origin / 'runs' / 'clbench' / family / str(repeat) / 'none' /
                                       f'episode_{i+1:03d}' / 'row.json')['initial_query_sha256'] for i in tasks],
                  url=url, policies=['all_selected','first_selected'],
                  actor_temperature=actor_temperature,
                  policy='Only direct Q reward updates differ; both arms retrieve and write all selected memories. Identical completions share one sampled response across arms')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'design.json').exists() and read(output / 'design.json') != design:
        raise ValueError('Existing online-ablation design differs')
    save(output / 'design.json', design)
    client = ControlledActorClient(plan, repeat, actor_temperature)
    memories = {}
    calibration = plan['calibration']['alfworld' if benchmark == 'alfworld' else family]
    for name, cls in [('all_selected', Memory), ('first_selected', FirstOnlyCreditMemory)]:
        memories[name] = cls(plan, client, output / name / 'memory', calibration,
                             embedder=memories['all_selected'].service.embedding_provider if memories else None)
    rows = []
    for i, task in enumerate(tasks):
        for name, memory in memories.items():
            target = output / name / f'episode_{i+1:03d}'
            if (target / 'row.json').exists():
                row = read(target / 'row.json')
                if row['status'] != 'complete' or sha(target / 'memory_after.json') != row['memory_after_sha256']:
                    raise ValueError('Cannot resume a failed or changed cell')
                memory.restore(target / 'memory_after.json')
            else:
                row = (alf_cell(plan, client, memory, task, repeat, 'memrl', target) if benchmark == 'alfworld'
                       else cl_cell(plan, client, memory, family, task, 'memrl', target))
            if row['status'] != 'complete':
                raise RuntimeError(f'{name} episode {i+1} failed: {row.get("error")}')
            rows.append(dict(policy=name, index=i,
                             game=task['path'] if benchmark=='alfworld' else None,
                             reward=row['reward'], first_attempt=row.get('first_attempt'),
                             within_three=row.get('within_three'), attempts=row.get('attempts')))
            save(output / 'progress.json', dict(completed=len(rows), expected=2*len(tasks),
                                                 response_cache_hits=client.response_cache_hits,
                                                 reward={n:sum(r['reward'] for r in rows if r['policy']==n)
                                                         for n in memories}))
            print(json.dumps(rows[-1]), flush=True)
    save(output / 'rows.json', rows)
    for metric in (('first_attempt','within_three') if benchmark=='alfworld' else ('reward',)):
        a = [r[metric] for r in rows if r['policy']=='all_selected']
        b = [r[metric] for r in rows if r['policy']=='first_selected']
        print(metric, 'all',sum(a),'first_only',sum(b),'n',len(a),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--origin', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--benchmark', choices=['alfworld','clbench'], required=True)
    p.add_argument('--family', required=True)
    p.add_argument('--repeat', type=int, required=True)
    p.add_argument('--url', required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--actor-temperature', type=float, default=0.,
                   help='Greedy by default for a deterministic mechanism test; original protocol used 0.7')
    a = p.parse_args()
    run(a.origin.resolve(),a.output.resolve(),a.benchmark,a.family,a.repeat,a.url,a.limit,
        a.actor_temperature)


if __name__ == '__main__':
    main()
