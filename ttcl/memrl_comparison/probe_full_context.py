"""Freeze and replay ALFWorld train retrieval against empty context.

Only the empty-context branch is newly sampled. The matched full-context
branch is the already frozen index-0 deletion probe with the same game and
actor seed. This is a fixed-snapshot diagnostic, not an online memory chain.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from ttcl.experience_evolution.core import seed
from ttcl.icl_mem0_comparison.protocol import Client, read, save, sha
from .credit_probe import memory_arms, run_alf


def _cases(first: Path, remaining: Path):
    first_selection = read(Path(read(first / 'design.json')['origin']) /
                           'selected_probe_cases.json')
    remaining_selection = read(remaining / 'training_selection.json')
    second = read(Path(remaining_selection['selection_path']))
    if first_selection['actor_repeats'] != second['actor_repeats']:
        raise ValueError('Actor seeds differ between training waves')
    result = [(x['case'], first) for x in first_selection['cases']]
    result.extend((x['case'], remaining) for x in second['cases'])
    if len({case for case, _ in result}) != len(result):
        raise ValueError('Duplicated ALFWorld training case')
    return result, first_selection['actor_repeats']


def prepare(first: Path, remaining: Path, output: Path, url: str) -> dict:
    if output.exists():
        raise FileExistsError(output)
    source = Path(read(first / 'design.json')['origin'])
    if Path(read(remaining / 'design.json')['origin']) != source:
        raise ValueError('Training waves have different sources')
    complete = read(source / 'complete.json')
    if complete['completed'] != complete['expected']:
        raise ValueError('ALFWorld train source is incomplete')
    cases, repeats = _cases(first, remaining)
    refs = []
    for case, probe in cases:
        spec, arms = memory_arms(source, case)
        for actor_repeat in repeats:
            folder = probe / 'alfworld' / spec['task'] / str(spec['repeat']) / \
                f"episode_{spec['index']+1:03d}" / f'actor_repeat_{actor_repeat}'
            full_path = folder / 'full' / 'episode.json'
            full = read(full_path)
            if (full['status'] != 'complete' or
                    full['game'] != spec['original_memrl']['game'] or
                    full['memory'] != arms['full'] or
                    full['seed'] != seed(actor_repeat, full['game'], 0, 'actor')):
                raise ValueError(f'Unmatched full branch: {full_path}')
            refs.append(dict(case=case, actor_repeat=actor_repeat,
                             full_path=str(full_path.resolve()),
                             full_sha256=sha(full_path),
                             source_input_sha256=spec['source_input_sha256'],
                             snapshot_sha256=spec['snapshot_sha256'],
                             retrieval_sha256=spec['retrieval_sha256']))
    module = Path(__file__)
    runner = Path(run_alf.__code__.co_filename)
    design = dict(schema='alf_train_full_vs_none_v1', source=str(source.resolve()),
                  source_complete_sha256=sha(source / 'complete.json'),
                  source_plan_sha256=sha(source / 'plan.json'),
                  first_selection_sha256=sha(source / 'selected_probe_cases.json'),
                  remaining_selection_path=str(Path(read(remaining / 'training_selection.json')
                                                    ['selection_path']).resolve()),
                  remaining_selection_sha256=sha(Path(read(remaining /
                                                        'training_selection.json')['selection_path'])),
                  first_probe=str(first.resolve()), remaining_probe=str(remaining.resolve()),
                  first_probe_design_sha256=sha(first / 'design.json'),
                  remaining_probe_design_sha256=sha(remaining / 'design.json'),
                  runner_sha256=sha(module), credit_probe_sha256=sha(runner),
                  url=url, actor_repeats=repeats, refs=refs,
                  note='Outcome-blind all eligible >=2-memory train cases; fixed snapshot; paired actor seeds')
    output.mkdir(parents=True)
    save(output / 'design.json', design)
    frozen = output / 'source'
    frozen.mkdir()
    shutil.copy2(module, frozen / module.name)
    shutil.copy2(runner, frozen / runner.name)
    return design


def audit(output: Path, require_complete: bool = True) -> dict:
    design = read(output / 'design.json')
    if design['schema'] != 'alf_train_full_vs_none_v1':
        raise ValueError('Unexpected probe schema')
    source = Path(design['source'])
    if (sha(source / 'complete.json') != design['source_complete_sha256'] or
            sha(source / 'plan.json') != design['source_plan_sha256'] or
            sha(source / 'selected_probe_cases.json') != design['first_selection_sha256'] or
            sha(Path(design['remaining_selection_path'])) != design['remaining_selection_sha256'] or
            sha(Path(design['first_probe']) / 'design.json') != design['first_probe_design_sha256'] or
            sha(Path(design['remaining_probe']) / 'design.json') != design['remaining_probe_design_sha256'] or
            sha(Path(__file__)) != design['runner_sha256'] or
            sha(Path(run_alf.__code__.co_filename)) != design['credit_probe_sha256'] or
            sha(output / 'source' / Path(__file__).name) != design['runner_sha256'] or
            sha(output / 'source' / Path(run_alf.__code__.co_filename).name) !=
            design['credit_probe_sha256']):
        raise ValueError('Frozen full-context probe source changed')
    selected, repeats = _cases(Path(design['first_probe']),
                               Path(design['remaining_probe']))
    if repeats != design['actor_repeats'] or len(design['refs']) != len(selected) * len(repeats):
        raise ValueError('Frozen case set changed')
    expected = {(case, rep) for case, _ in selected for rep in repeats}
    if {(r['case'], r['actor_repeat']) for r in design['refs']} != expected:
        raise ValueError('Frozen case/seed set changed')
    rows, missing = [], []
    for ref in design['refs']:
        spec, arms = memory_arms(source, ref['case'])
        if (not spec['original_memrl']['game'].startswith('json_2.1.1/train/') or
                spec['source_input_sha256'] != ref['source_input_sha256'] or
                spec['snapshot_sha256'] != ref['snapshot_sha256'] or
                spec['retrieval_sha256'] != ref['retrieval_sha256']):
            raise ValueError('Train case binding changed')
        full_path = Path(ref['full_path'])
        if sha(full_path) != ref['full_sha256']:
            raise ValueError('Frozen full branch changed')
        full = read(full_path)
        if (full['status'] != 'complete' or full['game'] != spec['original_memrl']['game'] or
                full['memory'] != arms['full'] or
                full['seed'] != seed(ref['actor_repeat'], full['game'], 0, 'actor')):
            raise ValueError('Frozen full branch is not matched')
        target = output / 'alfworld' / spec['task'] / str(spec['repeat']) / \
            f"episode_{spec['index']+1:03d}" / f"actor_repeat_{ref['actor_repeat']}" / 'none'
        if not (target / 'episode.json').exists():
            missing.append(dict(case=ref['case'], actor_repeat=ref['actor_repeat']))
            continue
        none = read(target / 'episode.json')
        if (none['status'] != 'complete' or none['game'] != full['game'] or
                none['seed'] != full['seed'] or none['memory'] != '' or
                none['reward'] not in (0, 1) or full['reward'] not in (0, 1)):
            raise ValueError(f'Unmatched empty branch: {target}')
        rows.append(dict(case=ref['case'], task=spec['task'],
                         actor_repeat=ref['actor_repeat'],
                         source_input_sha256=spec['source_input_sha256'],
                         snapshot_sha256=spec['snapshot_sha256'],
                         retrieval_sha256=spec['retrieval_sha256'],
                         full=float(full['reward']), none=float(none['reward']),
                         delta=float(full['reward'] - none['reward']),
                         memory_ids=spec['ids']))
    if require_complete and missing:
        raise ValueError(f'Missing {len(missing)} paired branches')
    return dict(expected=len(design['refs']), completed=len(rows), missing=missing,
                rows=rows, train_split='train',
                note='Fixed-snapshot full-versus-empty first-attempt rewards; no online memory evolution')


def run(output: Path) -> None:
    design = read(output / 'design.json')
    audit(output, require_complete=False)
    source = Path(design['source'])
    plan = read(source / 'plan.json')
    plan['url'] = design['url']
    plan['alf']['actor_url'] = design['url']
    for ref in design['refs']:
        spec, _ = memory_arms(source, ref['case'])
        target = output / 'alfworld' / spec['task'] / str(spec['repeat']) / \
            f"episode_{spec['index']+1:03d}" / f"actor_repeat_{ref['actor_repeat']}"
        client = Client(plan, ref['actor_repeat'])
        replay_spec = dict(spec, repeat=ref['actor_repeat'])
        run_alf(plan, client, replay_spec, {'none': ''}, target)
        print(f"{ref['case']} seed={ref['actor_repeat']}", flush=True)
    save(output / 'analysis.json', audit(output))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first', type=Path)
    p.add_argument('--remaining', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--url')
    p.add_argument('--audit-only', action='store_true')
    a = p.parse_args()
    output = a.output.resolve()
    if a.audit_only:
        result = audit(output)
        save(output / 'analysis.json', result)
        print(f"Audited {result['completed']}/{result['expected']} pairs")
        return
    if not output.exists():
        if not a.first or not a.remaining or not a.url:
            p.error('Preparing a new probe requires --first, --remaining and --url')
        prepare(a.first.resolve(), a.remaining.resolve(), output, a.url)
    run(output)


if __name__ == '__main__':
    main()
