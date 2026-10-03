"""Diagnose v32 readout at one fixed prior online memory snapshot.

This is a development-only CLBench intervention: the same candidate bank,
target input and actor seed are replayed with and without the recurrent
feedback readout. It does not replace the full online paired benchmark.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .credit_online_ablation import ControlledActorClient
from .grounded_evidence_v27 import TypedGroundedMemory as V27Memory
from .grounded_evidence_v32 import TypedGroundedMemory
from .worker import cl_cell


class NoRecurrentReadout(TypedGroundedMemory):
    def retrieve(self, query):
        return V27Memory.retrieve(self, query)


def probe(run_output: Path, output: Path, episode: int, url: str) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    run_output = run_output.resolve()
    design = read(run_output / 'design.json')
    if design['benchmark'] != 'clbench' or not 2 <= episode <= design['limit']:
        raise ValueError('Need one previously completed CLBench target')
    previous = run_output / 'typed_grounded' / f'episode_{episode-1:03d}'
    source = previous / 'memory_after.json'
    previous_row = read(previous / 'row.json')
    if (previous_row['status'] != 'complete' or
            sha(source) != previous_row['memory_after_sha256'] or
            design['implementation_sha256'] != sha(
                Path(__file__).with_name('grounded_evidence_v32.py'))):
        raise ValueError('Changed or incomplete v32 source state')
    origin = Path(design['origin'])
    if sha(origin / 'plan.json') != design['origin_plan_sha256']:
        raise ValueError('Origin plan changed')
    plan = read(origin / 'plan.json')
    plan['url'] = url
    plan['q_min_threshold'] = plan['q_min_thresholds']['clbench']
    benchmark = os.environ['TTCL_BENCH']
    os.chdir(benchmark)
    client = ControlledActorClient(plan, design['repeat'], design['temperature'])
    calibration = plan['calibration'][design['task']]
    controls = {}
    frozen = dict(schema='recurrent_feedback_v32_fixed_snapshot_probe_v1',
                  parent=str(run_output), parent_design_sha256=sha(
                      run_output / 'design.json'), episode=episode,
                  input_sha256=design['bindings'][episode-1],
                  source_snapshot_sha256=sha(source),
                  policy_sha256=design['implementation_sha256'],
                  runner_sha256=sha(Path(__file__)), url=url,
                  note='Development-only scored-suffix diagnosis; not independent test evidence')
    output.mkdir(parents=True)
    save(output / 'design.json', frozen)
    (output / 'source').mkdir()
    for filename in ('probe_recurrent_feedback_v32_control.py',
                     'grounded_evidence_v32.py', 'grounded_evidence_v30.py',
                     'grounded_evidence_v27.py', 'recurrent_feedback.py'):
        path = Path(__file__).with_name(filename)
        shutil.copy2(path, output / 'source' / filename)
    embedder = None
    for name, cls in (('no_recurrent', NoRecurrentReadout),
                      ('recurrent', TypedGroundedMemory)):
        memory = cls(plan, client, output / name / 'memory', calibration,
                     embedder=embedder)
        memory.restore(source)
        if embedder is None:
            embedder = memory.service.embedding_provider
        row = cl_cell(plan, client, memory, design['task'], episode-1,
                      'memrl', output / name / 'episode')
        if (row['status'] != 'complete' or row['initial_query_sha256'] !=
                frozen['input_sha256']):
            raise ValueError(f'Unscored or unbound fixed-snapshot branch: {name}')
        controls[name] = row
    result = dict(schema='recurrent_feedback_v32_fixed_snapshot_probe_result_v1',
                  episode=episode, input_sha256=frozen['input_sha256'],
                  no_recurrent=controls['no_recurrent']['reward'],
                  recurrent=controls['recurrent']['reward'],
                  delta=controls['recurrent']['reward'] -
                  controls['no_recurrent']['reward'],
                  no_recurrent_calls=controls['no_recurrent']['actor_calls'],
                  recurrent_calls=controls['recurrent']['actor_calls'],
                  cache_hits=client.response_cache_hits,
                  caveat='One seed, post-hoc development diagnosis; online memory spillover excluded')
    save(output / 'analysis.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episode', type=int, required=True)
    parser.add_argument('--url', required=True)
    args = parser.parse_args()
    print(probe(args.run_output, args.output, args.episode, args.url), flush=True)


if __name__ == '__main__':
    main()
