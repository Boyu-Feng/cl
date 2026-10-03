"""Same-service full-context control for the frozen evidence-frame probe.

The archived full branch may have used another actor server. This control
replays the unchanged full context on the frame branch's server and seed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import statistics

from ttcl.experience_evolution.core import seed
from ttcl.icl_mem0_comparison.protocol import Client, read, save, sha
from .credit_probe import memory_arms, run_alf
from .probe_evidence_frame import audit as audit_frame, framed


def _frame_path(frame_output: Path, case: str, repeat: int) -> Path:
    parts = Path(case).parts
    return (frame_output / 'alfworld' / parts[1] / parts[2] / parts[4] /
            f'actor_repeat_{repeat}' / 'framed' / 'episode.json')


def _control_path(output: Path, case: str, repeat: int) -> Path:
    parts = Path(case).parts
    return (output / 'alfworld' / parts[1] / parts[2] / parts[4] /
            f'actor_repeat_{repeat}' / 'full' / 'episode.json')


def prepare(frame_output: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    frame_output = frame_output.resolve()
    prior = audit_frame(frame_output)
    if prior['completed'] != prior['expected'] or prior['missing']:
        raise ValueError('Incomplete frame branch')
    frame_design = read(frame_output / 'design.json')
    source = Path(frame_design['source'])
    refs = []
    for ref in frame_design['refs']:
        target = _frame_path(frame_output, ref['case'], ref['actor_repeat'])
        episode = read(target)
        spec, arms = memory_arms(source, ref['case'])
        if (episode['game'] != spec['original_memrl']['game'] or
                episode['memory'] != framed(arms['full']) or
                episode['seed'] != seed(ref['actor_repeat'], episode['game'], 0, 'actor') or
                spec['source_input_sha256'] != ref['source_input_sha256']):
            raise ValueError('Unmatched frame branch')
        refs.append(dict(case=ref['case'], actor_repeat=ref['actor_repeat'],
                         input_sha256=ref['source_input_sha256'],
                         retrieval_sha256=ref['retrieval_sha256'],
                         snapshot_sha256=ref['snapshot_sha256'],
                         framed_path=str(target), framed_sha256=sha(target),
                         archived_full_sha256=ref['full_sha256']))
    module = Path(__file__)
    runner = Path(run_alf.__code__.co_filename)
    frame_module = Path(audit_frame.__code__.co_filename)
    design = dict(schema='alf_train_evidence_frame_same_service_v1',
                  frame_output=str(frame_output),
                  frame_design_sha256=sha(frame_output / 'design.json'),
                  source=str(source), source_plan_sha256=sha(source / 'plan.json'),
                  runner_sha256=sha(module), credit_probe_sha256=sha(runner),
                  frame_module_sha256=sha(frame_module),
                  url=frame_design['url'], refs=refs,
                  note='Full-context control on same actor service as frozen framed branch; official train inputs only')
    output.mkdir(parents=True)
    save(output / 'design.json', design)
    (output / 'source').mkdir()
    for path in (module, runner, frame_module):
        shutil.copy2(path, output / 'source' / path.name)
    return design


def audit(output: Path, require_complete: bool = True) -> dict:
    design = read(output / 'design.json')
    if design['schema'] != 'alf_train_evidence_frame_same_service_v1':
        raise ValueError('Unexpected frozen control probe')
    frame_output = Path(design['frame_output'])
    source = Path(design['source'])
    frame_design = read(frame_output / 'design.json')
    prior = audit_frame(frame_output)
    if (prior['completed'] != prior['expected'] or prior['missing'] or
            sha(frame_output / 'design.json') != design['frame_design_sha256'] or
            sha(source / 'plan.json') != design['source_plan_sha256'] or
            frame_design['url'] != design['url']):
        raise ValueError('Frozen frame source or service changed')
    for path, key in ((Path(__file__), 'runner_sha256'),
                      (Path(run_alf.__code__.co_filename), 'credit_probe_sha256'),
                      (Path(audit_frame.__code__.co_filename), 'frame_module_sha256')):
        if sha(path) != design[key] or sha(output / 'source' / path.name) != design[key]:
            raise ValueError(f'Frozen control source changed: {path.name}')
    if {(r['case'], r['actor_repeat']) for r in design['refs']} != {
            (r['case'], r['actor_repeat']) for r in frame_design['refs']}:
        raise ValueError('Frozen pair set changed')
    rows, missing = [], []
    for ref in design['refs']:
        spec, arms = memory_arms(source, ref['case'])
        frame_path = Path(ref['framed_path'])
        framed_episode = read(frame_path)
        if (sha(frame_path) != ref['framed_sha256'] or
                framed_episode['memory'] != framed(arms['full']) or
                spec['source_input_sha256'] != ref['input_sha256'] or
                spec['retrieval_sha256'] != ref['retrieval_sha256'] or
                spec['snapshot_sha256'] != ref['snapshot_sha256']):
            raise ValueError('Framed branch or source changed')
        path = _control_path(output, ref['case'], ref['actor_repeat'])
        if not path.exists():
            missing.append(dict(case=ref['case'], actor_repeat=ref['actor_repeat']))
            continue
        full = read(path)
        if (full['status'] != 'complete' or full['game'] != framed_episode['game'] or
                full['seed'] != framed_episode['seed'] or
                full['memory'] != arms['full'] or full['reward'] not in (0, 1) or
                framed_episode['reward'] not in (0, 1)):
            raise ValueError(f'Unmatched same-service control: {path}')
        archived_path = next(Path(r['full_path']) for r in frame_design['refs']
                             if r['case'] == ref['case'] and
                             r['actor_repeat'] == ref['actor_repeat'])
        if sha(archived_path) != ref['archived_full_sha256']:
            raise ValueError('Archived full branch changed')
        archived = read(archived_path)
        rows.append(dict(case=ref['case'], task=spec['task'],
                         actor_repeat=ref['actor_repeat'],
                         input_sha256=ref['input_sha256'],
                         snapshot_sha256=ref['snapshot_sha256'],
                         full=float(full['reward']), framed=float(framed_episode['reward']),
                         archived_full=float(archived['reward']),
                         delta=float(framed_episode['reward'] - full['reward']),
                         service_drift=float(full['reward'] - archived['reward'])))
    if require_complete and missing:
        raise ValueError(f'Missing {len(missing)} same-service controls')
    deltas = [r['delta'] for r in rows]
    return dict(expected=len(design['refs']), completed=len(rows), missing=missing,
                train_split='train', rows=rows,
                mean_delta=statistics.fmean(deltas) if deltas else None,
                wins=sum(x > 0 for x in deltas), losses=sum(x < 0 for x in deltas),
                ties=sum(x == 0 for x in deltas),
                service_drift_count=sum(x['service_drift'] != 0 for x in rows),
                note='Framed versus full context on the same actor service, task, seed and frozen memory')


def run(output: Path) -> None:
    design = read(output / 'design.json')
    audit(output, require_complete=False)
    source = Path(design['source'])
    plan = read(source / 'plan.json')
    plan['url'] = design['url']
    plan['alf']['actor_url'] = design['url']
    for ref in design['refs']:
        spec, arms = memory_arms(source, ref['case'])
        path = _control_path(output, ref['case'], ref['actor_repeat'])
        client = Client(plan, ref['actor_repeat'])
        run_alf(plan, client, dict(spec, repeat=ref['actor_repeat']),
                {'full': arms['full']}, path.parent.parent)
        print(f'{ref["case"]} seed={ref["actor_repeat"]}', flush=True)
    save(output / 'analysis.json', audit(output))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frame-output', type=Path)
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
        if not a.frame_output:
            p.error('Preparation requires --frame-output')
        prepare(a.frame_output, output)
    run(output)


if __name__ == '__main__':
    main()
