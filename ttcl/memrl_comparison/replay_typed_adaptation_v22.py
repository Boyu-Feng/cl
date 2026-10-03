"""One-case, fixed-memory continuation for a new typed-evidence rule.

This diagnostic reuses a frozen prior candidate snapshot. It is not an
independent empty-to-online benchmark chain and cannot establish policy gain.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .credit_online_ablation import ControlledActorClient
from .grounded_evidence_v22 import TypedGroundedMemory
from .worker import cl_cell


SOURCES = ('replay_typed_adaptation_v22.py', 'grounded_evidence_v22.py',
           'grounded_evidence_v21.py',
           'grounded_evidence_v15.py', 'grounded_evidence_v13_base.py',
           'typed_projection.py', 'typed_executor.py',
           'credit_online_ablation.py', 'worker.py', 'memory.py')


def run(origin: Path, prior: Path, output: Path, task: str,
        repeat: int, index: int, url: str, temperature: float) -> dict:
    if output.exists():
        raise FileExistsError(output)
    plan = read(origin / 'plan.json')
    previous_design = read(prior / 'design.json')
    if (previous_design['origin'] != str(origin) or
            previous_design['task'] != task or
            previous_design['repeat'] != repeat or
            previous_design['benchmark'] != 'clbench' or
            previous_design['temperature'] != temperature or
            index < 1 or index >= previous_design['limit']):
        raise ValueError('Prior online chain or requested continuation differs')
    prior_episode = prior / 'typed_grounded' / f'episode_{index:03d}'
    snapshot_path = prior_episode / 'memory_after.json'
    prior_row = read(prior_episode / 'row.json')
    if (prior_row['status'] != 'complete' or
            sha(snapshot_path) != prior_row['memory_after_sha256']):
        raise ValueError('Prior memory snapshot changed')
    target_index = index  # zero-based next instance
    source_row = read(origin / 'runs' / 'clbench' / task / str(repeat) /
                      'none' / f'episode_{target_index+1:03d}' / 'row.json')
    if source_row['status'] != 'complete':
        raise ValueError('Target official source is incomplete')
    code_root = Path(__file__).parent
    design = dict(schema='typed_adaptation_fixed_snapshot_v22',
                  origin=str(origin), origin_plan_sha256=sha(origin / 'plan.json'),
                  prior=str(prior), prior_design_sha256=sha(prior / 'design.json'),
                  prior_snapshot=str(snapshot_path), prior_snapshot_sha256=sha(snapshot_path),
                  task=task, repeat=repeat, target_index=target_index,
                  target_input_sha256=source_row['initial_query_sha256'],
                  url=url, temperature=temperature,
                  source_sha256={name:sha(code_root / name) for name in SOURCES},
                  note='One-case frozen-snapshot mechanism replay, not a full online benchmark')
    output.mkdir(parents=True)
    save(output / 'design.json', design)
    frozen = output / 'source'
    frozen.mkdir()
    for name in SOURCES:
        shutil.copy2(code_root / name, frozen / name)
    plan['url'] = url
    plan['alf']['actor_url'] = url
    plan['alf']['actor_temperature'] = temperature
    plan['q_min_threshold'] = plan['q_min_thresholds']['clbench']
    os.environ.setdefault('TTCL_BENCH', str(origin / 'source' / 'bench'))
    os.chdir(os.environ['TTCL_BENCH'])
    client = ControlledActorClient(plan, repeat, temperature)
    memory = TypedGroundedMemory(plan, client, output / 'memory',
                                 plan['calibration'][task])
    new_signature = memory.signature
    old_signature = read(snapshot_path)['protocol_signature']
    memory.signature = old_signature
    memory.restore(snapshot_path)
    memory.signature = new_signature
    memory.snapshot(output / 'restored_snapshot.json')
    restored = read(output / 'restored_snapshot.json')
    original = read(snapshot_path)
    if {k:v for k,v in restored.items() if k != 'protocol_signature'} != {
            k:v for k,v in original.items() if k != 'protocol_signature'}:
        raise ValueError('Memory state changed during compatible restore')
    row = cl_cell(plan, client, memory, task, target_index, 'memrl',
                  output / 'episode')
    if (row['status'] != 'complete' or
            row['initial_query_sha256'] != design['target_input_sha256']):
        raise ValueError('Mechanism replay failed or target input changed')
    result = dict(reward=row['reward'], actor_calls=row['actor_calls'],
                  prior_snapshot_sha256=design['prior_snapshot_sha256'],
                  adaptation=read(output / 'episode' / 'typed_context_adaptation.json')
                    if (output / 'episode' / 'typed_context_adaptation.json').exists() else None)
    save(output / 'result.json', result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--origin', type=Path, required=True)
    p.add_argument('--prior', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--repeat', type=int, required=True)
    p.add_argument('--index', type=int, required=True,
                   help='One-based number of completed prior episodes')
    p.add_argument('--url', required=True)
    p.add_argument('--temperature', type=float, default=.7)
    a = p.parse_args()
    result = run(a.origin.resolve(), a.prior.resolve(), a.output.resolve(),
                 a.task, a.repeat, a.index, a.url, a.temperature)
    print(result)


if __name__ == '__main__':
    main()
