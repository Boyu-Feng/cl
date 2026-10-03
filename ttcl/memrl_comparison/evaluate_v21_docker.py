"""Evaluate native MemRL versus v21 on the two canonical Docker CLBench runs.

The runner requires a working Docker endpoint and the official images. It
preserves one task/container lifecycle per arm, including Sales' persistent
/app workspace. A failed or interrupted arm is retained but cannot be
resumed without its live container; restart under a new output directory.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import traceback

from ttcl.icl_mem0_comparison.protocol import read, save, sha, normalize_prompt
from ttcl.icl_mem0_comparison.worker import base
from .credit_online_ablation import ControlledActorClient
from .docker_v21_protocol import OnlineDockerRecorder, OnlineDockerSystem
from .grounded_evidence_v21 import TypedGroundedMemory
from .memory import Embedder, Memory


TASKS = ('sales_prediction', 'codebase_adaptation')
SOURCE = ('evaluate_v21_docker.py', 'docker_v21_protocol.py',
          'grounded_evidence_v13_base.py', 'grounded_evidence_v15.py',
          'grounded_evidence_v21.py', 'typed_projection.py',
          'typed_executor.py', 'credit_online_ablation.py', 'worker.py',
          'memory.py')
TRANSPORT_SOURCE = {
    'structured_memory_run_benchmark.py':
        Path(__file__).resolve().parents[1] / 'structured_memory' /
        'run_benchmark.py',
    'icl_mem0_worker.py':
        Path(__file__).resolve().parents[1] / 'icl_mem0_comparison' / 'worker.py',
    'icl_mem0_protocol.py':
        Path(__file__).resolve().parents[1] / 'icl_mem0_comparison' /
        'protocol.py',
}


def make_task(name: str, seed: int, *, independent: bool):
    if name == 'sales_prediction':
        from src.tasks.sales_prediction.task import SalesPredictionTask
        return SalesPredictionTask(seed=seed, schedule='default',
            clean_workspace_between_instances=independent)
    if name == 'codebase_adaptation':
        from src.tasks.codebase_adaptation.task import CodebaseAdaptationTask
        return CodebaseAdaptationTask(seed=seed, schedule='default')
    raise ValueError(f'Unsupported Docker task: {name}')


def cleanup(task):
    if hasattr(task, '_cleanup_container'):
        task._cleanup_container()


def static_bindings(bench: Path, name: str) -> dict[str, str]:
    root = bench / 'src' / 'tasks' / name
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = [p for p in root.rglob('*') if p.is_file() and
             p.suffix in ('.py', '.json', '.yaml', '.yml', '.j2')]
    paths += [bench / 'src' / 'interface.py',
              bench / 'src' / 'runtime' / 'runner.py']
    if name == 'sales_prediction':
        paths += [p for p in (bench / 'data' / name).glob('*')
                  if p.is_file() and p.suffix in ('.json', '.jsonl')]
    else:
        paths.append(bench / 'data' / name / 'final-dataset.jsonl')
    if not paths or any(not p.is_file() for p in paths):
        raise ValueError('Missing official Docker task source or data')
    return {str(p.relative_to(bench)): sha(p) for p in sorted(set(paths))}


def images_for_task(bench: Path, name: str) -> list[str]:
    if name == 'sales_prediction':
        from src.tasks.sales_prediction.task import _DOCKER_IMAGE
        return [_DOCKER_IMAGE]
    dataset = bench / 'data' / name / 'final-dataset.jsonl'
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    if not rows or any(not row.get('image_name') for row in rows):
        raise ValueError('Codebase dataset has no image binding')
    return sorted({row['image_name'] for row in rows})


def docker_preflight(images: list[str]) -> dict:
    if shutil.which('docker') is None:
        raise RuntimeError('Docker CLI is unavailable')
    command = ['docker', 'info', '--format', '{{.ServerVersion}}']
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise RuntimeError('Docker endpoint unavailable: ' + result.stderr.strip())
    bound = {}
    for image in images:
        subprocess.run(['docker', 'pull', image], check=True, timeout=1800,
                       stdout=subprocess.DEVNULL)
        inspection = subprocess.run(['docker', 'image', 'inspect', image],
            check=True, capture_output=True, text=True, timeout=30)
        data = json.loads(inspection.stdout)[0]
        bound[image] = data['Id']
        # This is the same non-network container invocation style as the task.
        subprocess.run(['docker', 'run', '--rm', '--network=none', image,
                        '/bin/sh', '-c', 'true'], check=True, timeout=120,
                       stdout=subprocess.DEVNULL)
    return dict(endpoint=os.environ.get('DOCKER_HOST', 'unix:///var/run/docker.sock'),
                server_version=result.stdout.strip(), image_ids=bound)


def count_instances(name: str, seed: int) -> int:
    task = make_task(name, seed, independent=False)
    try:
        task.build_canonical_run_state()
        count = len(task.instances)
        if count < 2:
            raise ValueError('Official task sequence is too short')
        return count
    finally:
        cleanup(task)


def calibrate_public_prefix(name: str, seed: int, count: int,
                            embedding_path: str) -> dict:
    import numpy as np
    from ttcl.icl_mem0_comparison.worker import base as worker_base
    embedder = Embedder(embedding_path)
    prompts = []
    for index in range(int(count * .2)):
        task = make_task(name, seed, independent=True)
        try:
            query = task.reset_baseline_instance(index)
            brief = task.get_agent_brief()
            brief = worker_base.format_task_agent_brief(brief) if brief else ''
            prompts.append(normalize_prompt(query.prompt, brief, index, count))
        finally:
            cleanup(task)
    if len(prompts) < 2:
        raise ValueError('Calibration needs at least two public prefix prompts')
    vectors = np.asarray(embedder.embed(prompts))
    values = np.clip((vectors @ vectors.T)[np.triu_indices(len(prompts), k=1)],
                     -1., 1.)
    return dict(mean=float(values.mean()), std=max(float(values.std()), 1e-6),
                threshold=min(.999999, float(np.quantile(values, .8))),
                prompt_count=len(prompts), pairs=len(values),
                public_prompt_sha256=[__import__('hashlib').sha256(
                    value.encode()).hexdigest() for value in prompts],
                rule='First 20% public initial prompts; no reward or hidden tests')


def prepare(origin: Path, output: Path, name: str, repeat: int, url: str,
            limit: int | None, temperature: float) -> dict:
    if output.exists():
        raise FileExistsError(output)
    origin = origin.resolve()
    bench = Path(os.environ['TTCL_BENCH']).resolve()
    os.chdir(bench)
    source_plan = read(origin / 'plan.json')
    if name not in TASKS or repeat not in source_plan['repeats']:
        raise ValueError('Unknown Docker task or actor repeat')
    full_count = count_instances(name, source_plan['task_seed'])
    limit = full_count if limit is None else limit
    if not 1 <= limit <= full_count:
        raise ValueError('Invalid Docker task limit')
    bindings = static_bindings(bench, name)
    images = images_for_task(bench, name)
    docker = docker_preflight(images)
    calibration = calibrate_public_prefix(
        name, source_plan['task_seed'], full_count, source_plan['embedding'])
    own_hashes = {file: sha(Path(__file__).with_name(file)) for file in SOURCE}
    own_hashes.update({file: sha(path) for file, path in
                       TRANSPORT_SOURCE.items()})
    design = dict(schema='v21_docker_clbench_online_v1',
                  origin=str(origin), origin_plan_sha256=sha(origin / 'plan.json'),
                  bench=str(bench), task=name, repeat=repeat, url=url,
                  temperature=temperature, task_seed=source_plan['task_seed'],
                  full_count=full_count, limit=limit,
                  official_suffix_start=int(full_count * .2),
                  task_source_and_data=bindings, docker=docker,
                  source_sha256=own_hashes,
                  calibration=calibration,
                  policy='Official ordered task per arm; Sales /app persists; native MemRL versus unchanged v21; same model/seed/budget; missing outcome never zero')
    output.mkdir(parents=True)
    save(output / 'design.json', design)
    (output / 'source').mkdir()
    for file in SOURCE:
        shutil.copy2(Path(__file__).with_name(file), output / 'source' / file)
    for file, path in TRANSPORT_SOURCE.items():
        shutil.copy2(path, output / 'source' / file)
    return design


def verify(output: Path) -> dict:
    design = read(output / 'design.json')
    if design['schema'] != 'v21_docker_clbench_online_v1':
        raise ValueError('Unexpected Docker run schema')
    origin, bench = Path(design['origin']), Path(design['bench'])
    if sha(origin / 'plan.json') != design['origin_plan_sha256']:
        raise ValueError('Original plan changed')
    if static_bindings(bench, design['task']) != design['task_source_and_data']:
        raise ValueError('Official task source or data changed')
    for file, expected in design['source_sha256'].items():
        active = TRANSPORT_SOURCE.get(file, Path(__file__).with_name(file))
        if (sha(active) != expected or
                sha(output / 'source' / file) != expected):
            raise ValueError(f'Frozen implementation changed: {file}')
    for image, expected in design['docker']['image_ids'].items():
        inspected = subprocess.run(['docker', 'image', 'inspect', image],
            capture_output=True, text=True, timeout=30)
        if inspected.returncode or json.loads(inspected.stdout)[0]['Id'] != expected:
            raise ValueError(f'Official Docker image changed: {image}')
    return design


def analyze(output: Path, *, require_complete: bool = True) -> dict:
    design = verify(output)
    rows, missing = [], []
    for index in range(design['limit']):
        pair = []
        for arm in ('vanilla', 'typed_grounded'):
            path = output / arm / f'episode_{index + 1:03d}' / 'row.json'
            if not path.exists():
                missing.append(dict(index=index, arm=arm))
                continue
            row = read(path)
            snapshot = path.parent / 'memory_after.json'
            if (row['status'] != 'complete' or row['canonical_index'] != index or
                    row['task'] != design['task'] or
                    sha(snapshot) != row['memory_after_sha256'] or
                    not math.isfinite(row['reward'])):
                raise ValueError(f'Invalid official Docker row: {path}')
            pair.append(row)
        if len(pair) == 2:
            native, candidate = pair
            if native['instance_id'] != candidate['instance_id']:
                raise ValueError('Paired Docker instance IDs differ')
            rows.append(dict(index=index, instance_id=native['instance_id'],
                             native=native['reward'], candidate=candidate['reward'],
                             delta=candidate['reward'] - native['reward'],
                             native_calls=native['actor_calls'],
                             candidate_calls=candidate['actor_calls'],
                             official_suffix=index >= design['official_suffix_start']))
    if require_complete and missing:
        raise ValueError(f'{len(missing)} Docker arm cells missing')
    suffix = [row for row in rows if row['official_suffix']]
    result = dict(schema='v21_docker_clbench_analysis_v1',
                  design_sha256=sha(output / 'design.json'),
                  full_count=design['full_count'], limit=design['limit'],
                  complete_pairs=len(rows), missing=missing, rows=rows,
                  scored_suffix_pairs=len(suffix),
                  note='Pilot if limit < full_count; no cross-task reward pooling')
    if not missing and design['limit'] == design['full_count']:
        deltas = [row['delta'] for row in suffix]
        result.update(official_suffix_mean_delta=statistics.fmean(deltas),
                      wins=sum(x > 0 for x in deltas),
                      losses=sum(x < 0 for x in deltas),
                      ties=sum(x == 0 for x in deltas))
    return result


def run(output: Path) -> None:
    design = verify(output)
    if any((output / arm).exists() for arm in ('vanilla', 'typed_grounded')):
        raise FileExistsError('Docker chain is not resumable; use a new output')
    origin = Path(design['origin'])
    bench = Path(design['bench'])
    os.chdir(bench)
    plan = read(origin / 'plan.json')
    plan['tasks'] = dict(plan['tasks'], **{design['task']: design['full_count']})
    plan['calibration'] = dict(plan['calibration'],
                               **{design['task']: design['calibration']})
    plan['url'] = design['url']
    plan['q_min_threshold'] = plan['q_min_thresholds']['clbench']
    client = ControlledActorClient(plan, design['repeat'], design['temperature'])
    embedder = None
    for arm, cls in (('vanilla', Memory), ('typed_grounded', TypedGroundedMemory)):
        target = output / arm
        task = make_task(design['task'], design['task_seed'], independent=False)
        try:
            task.build_canonical_run_state()
            task.select_run_instances(list(range(design['limit'])))
            query = task.build_current_query()
            brief = task.get_agent_brief()
            brief = base.format_task_agent_brief(brief) if brief else ''
            memory = cls(plan, client, target / 'memory',
                         design['calibration'], embedder=embedder)
            if embedder is None:
                embedder = memory.service.embedding_provider
            system = OnlineDockerSystem(plan, client, memory, arm,
                design['task'], brief, target, design['full_count'])
            recorder = OnlineDockerRecorder(target, system, design['limit'])
            result = base.run_task(task, system, trace_recorder=recorder,
                show_progress=False, reset_system=False, initial_query=query)
            if len(result.instance_outcomes) != design['limit']:
                raise ValueError('Official Docker outcome count differs from plan')
            save(target / 'complete.json', dict(status='complete',
                outcomes=len(result.instance_outcomes),
                actor_calls=system.calls, writer_calls=memory.calls,
                response_cache_hits=client.response_cache_hits))
        except Exception as exc:
            save(target / 'failed.json', dict(status='failed',
                error=repr(exc), traceback=traceback.format_exc()))
            raise
        finally:
            cleanup(task)
        save(output / 'analysis_partial.json', analyze(output,
             require_complete=False))
    save(output / 'analysis.json', analyze(output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--repeat', type=int)
    parser.add_argument('--url')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--temperature', type=float, default=.7)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.audit_only:
        save(output / 'analysis_partial.json', analyze(output,
             require_complete=False))
        return
    if not args.origin or not args.task or args.repeat is None or not args.url:
        parser.error('Run requires origin, task, repeat, and url')
    prepare(args.origin, output, args.task, args.repeat,
            args.url, args.limit, args.temperature)
    run(output)


if __name__ == '__main__':
    main()
