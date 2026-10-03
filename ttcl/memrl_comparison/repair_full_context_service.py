"""Repair cross-service ALFWorld train full/empty comparisons.

The second-wave archived full branch used a different actor service from the
empty branch. Replay only the unchanged full context on the empty branch's
service, with the same official train game and actor seed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import statistics

from ttcl.experience_evolution.core import seed
from ttcl.icl_mem0_comparison.protocol import Client, read, save, sha
from .credit_probe import memory_arms, run_alf
from .probe_full_context import audit as audit_old


def _episode_path(root: Path, case: str, repeat: int, arm: str) -> Path:
    parts = Path(case).parts
    return (root / 'alfworld' / parts[1] / parts[2] / parts[4] /
            f'actor_repeat_{repeat}' / arm / 'episode.json')


def prepare(old_probe: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    old_probe = old_probe.resolve()
    previous = audit_old(old_probe)
    if previous['completed'] != previous['expected'] or previous['missing']:
        raise ValueError('Incomplete archived full/empty source')
    old = read(old_probe / 'design.json')
    source = Path(old['source'])
    remaining = Path(old['remaining_probe'])
    refs = []
    for ref in old['refs']:
        full_path = Path(ref['full_path'])
        if not full_path.is_relative_to(remaining):
            continue
        none_path = _episode_path(old_probe, ref['case'], ref['actor_repeat'], 'none')
        spec, arms = memory_arms(source, ref['case'])
        archived_full, none = read(full_path), read(none_path)
        if (not archived_full['game'].startswith('json_2.1.1/train/') or
                archived_full['memory'] != arms['full'] or none['memory'] != '' or
                archived_full['game'] != none['game'] or
                archived_full['seed'] != none['seed'] or
                none['seed'] != seed(ref['actor_repeat'], none['game'], 0, 'actor') or
                spec['source_input_sha256'] != ref['source_input_sha256']):
            raise ValueError('Unmatched archived target')
        refs.append(dict(case=ref['case'], actor_repeat=ref['actor_repeat'],
                         source_input_sha256=ref['source_input_sha256'],
                         snapshot_sha256=ref['snapshot_sha256'],
                         retrieval_sha256=ref['retrieval_sha256'],
                         none_path=str(none_path), none_sha256=sha(none_path),
                         archived_full_path=str(full_path),
                         archived_full_sha256=sha(full_path)))
    if not refs:
        raise ValueError('No cross-service second-wave targets')
    module = Path(__file__)
    runner = Path(run_alf.__code__.co_filename)
    reference = Path(audit_old.__code__.co_filename)
    design = dict(schema='alf_train_full_empty_service_repair_v1',
                  old_probe=str(old_probe), old_design_sha256=sha(old_probe / 'design.json'),
                  source=str(source), source_plan_sha256=sha(source / 'plan.json'),
                  runner_sha256=sha(module), credit_probe_sha256=sha(runner),
                  reference_sha256=sha(reference), url=old['url'], refs=refs,
                  note='Archived empty branch and repaired full branch on same actor service; official train only')
    output.mkdir(parents=True)
    save(output / 'design.json', design)
    (output / 'source').mkdir()
    for path in (module, runner, reference):
        shutil.copy2(path, output / 'source' / path.name)
    return design


def audit(output: Path, require_complete: bool = True) -> dict:
    design = read(output / 'design.json')
    if design['schema'] != 'alf_train_full_empty_service_repair_v1':
        raise ValueError('Unexpected repair probe')
    old_probe = Path(design['old_probe'])
    source = Path(design['source'])
    old = read(old_probe / 'design.json')
    prior = audit_old(old_probe)
    if (prior['completed'] != prior['expected'] or prior['missing'] or
            sha(old_probe / 'design.json') != design['old_design_sha256'] or
            sha(source / 'plan.json') != design['source_plan_sha256'] or
            old['url'] != design['url']):
        raise ValueError('Archived source or service changed')
    for path, key in ((Path(__file__), 'runner_sha256'),
                      (Path(run_alf.__code__.co_filename), 'credit_probe_sha256'),
                      (Path(audit_old.__code__.co_filename), 'reference_sha256')):
        if sha(path) != design[key] or sha(output / 'source' / path.name) != design[key]:
            raise ValueError(f'Frozen repair source changed: {path.name}')
    rows, missing = [], []
    for ref in design['refs']:
        spec, arms = memory_arms(source, ref['case'])
        none_path = Path(ref['none_path'])
        archived_path = Path(ref['archived_full_path'])
        if (sha(none_path) != ref['none_sha256'] or
                sha(archived_path) != ref['archived_full_sha256'] or
                spec['source_input_sha256'] != ref['source_input_sha256'] or
                spec['snapshot_sha256'] != ref['snapshot_sha256'] or
                spec['retrieval_sha256'] != ref['retrieval_sha256']):
            raise ValueError('Bound train input or old branch changed')
        path = _episode_path(output, ref['case'], ref['actor_repeat'], 'full')
        if not path.exists():
            missing.append(dict(case=ref['case'], actor_repeat=ref['actor_repeat']))
            continue
        full, none, archived = read(path), read(none_path), read(archived_path)
        if (full['status'] != 'complete' or full['game'] != none['game'] or
                full['seed'] != none['seed'] or full['memory'] != arms['full'] or
                none['memory'] != '' or full['reward'] not in (0, 1) or
                none['reward'] not in (0, 1)):
            raise ValueError(f'Unmatched repaired pair: {path}')
        rows.append(dict(case=ref['case'], task=spec['task'],
                         actor_repeat=ref['actor_repeat'],
                         source_input_sha256=ref['source_input_sha256'],
                         full=float(full['reward']), none=float(none['reward']),
                         archived_full=float(archived['reward']),
                         delta=float(full['reward'] - none['reward']),
                         service_drift=float(full['reward'] - archived['reward'])))
    if require_complete and missing:
        raise ValueError(f'Missing {len(missing)} repaired full branches')
    deltas = [r['delta'] for r in rows]
    return dict(expected=len(design['refs']), completed=len(rows), missing=missing,
                train_split='train', rows=rows,
                mean_delta=statistics.fmean(deltas) if deltas else None,
                wins=sum(x > 0 for x in deltas), losses=sum(x < 0 for x in deltas),
                ties=sum(x == 0 for x in deltas),
                service_drift_count=sum(x['service_drift'] != 0 for x in rows),
                note='Same-service repaired full versus archived empty context on selected official train games')


def run(output: Path) -> None:
    design = read(output / 'design.json')
    audit(output, require_complete=False)
    source = Path(design['source'])
    plan = read(source / 'plan.json')
    plan['url'] = design['url']
    plan['alf']['actor_url'] = design['url']
    for ref in design['refs']:
        spec, arms = memory_arms(source, ref['case'])
        path = _episode_path(output, ref['case'], ref['actor_repeat'], 'full')
        client = Client(plan, ref['actor_repeat'])
        run_alf(plan, client, dict(spec, repeat=ref['actor_repeat']),
                {'full': arms['full']}, path.parent.parent)
        print(f'{ref["case"]} seed={ref["actor_repeat"]}', flush=True)
    save(output / 'analysis.json', audit(output))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-probe', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--audit-only', action='store_true')
    a = p.parse_args()
    output = a.output.resolve()
    if a.audit_only:
        result = audit(output)
        save(output / 'analysis.json', result)
        print(f'Audited {result["completed"]}/{result["expected"]} pairs')
        return
    if not output.exists():
        if not a.old_probe:
            p.error('Preparation requires --old-probe')
        prepare(a.old_probe, output)
    run(output)


if __name__ == '__main__':
    main()
