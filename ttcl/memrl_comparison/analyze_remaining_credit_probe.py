"""Audit outcome-blind second-wave ALFWorld train-memory probes."""
from __future__ import annotations

import argparse
from pathlib import Path

from ttcl.experience_evolution.core import seed
from ttcl.icl_mem0_comparison.protocol import read, save, sha
from ttcl.memrl_comparison.credit_probe import memory_arms


def analyze(output: Path) -> dict:
    design = read(output / 'design.json')
    if 'alf_actor_repeats' not in design or 'paired_drop_index' not in design:
        raise ValueError('Expected an ALFWorld paired-drop design')
    origin = Path(design['origin'])
    source_design = read(origin / 'design.json') if (origin / 'design.json').exists() else {}
    train_source = 'training_plan' in source_design and 'selected_games' in source_design
    if not train_source:
        raise ValueError('Second-wave probes require an official train source')
    if sha(origin / 'plan.json') != design['origin_plan_sha256']:
        raise ValueError('Origin plan content changed')
    plan = read(origin / 'plan.json')
    index = design['paired_drop_index']
    if train_source:
        completed = read(origin / 'complete.json')
        selection_manifest = read(output / 'training_selection.json')
        selection_path = Path(selection_manifest['selection_path'])
        selection = read(selection_path)
        if (selection_manifest['selection_sha256'] != sha(selection_path) or
                selection_manifest['design_sha256'] != sha(output / 'design.json') or
                selection['selection_code_sha256'] != sha(
                    Path(__file__).with_name('select_remaining_credit_train_probes.py')) or
                selection['context_reconstruction_sha256'] != sha(
                    Path(memory_arms.__code__.co_filename))):
            raise ValueError('Second-wave selection or execution changed')
        if (completed['completed'] != completed['expected'] or
                completed['design_sha256'] != sha(origin / 'design.json') or
                selection['source_complete_sha256'] != sha(origin / 'complete.json') or
                selection['source_plan_sha256'] != sha(origin / 'plan.json') or
                selection['first_selection_sha256'] != sha(
                    origin / 'selected_probe_cases.json') or
                selection['source'] != str(origin)):
            raise ValueError('Train source or outcome-blind selection changed')
        if (design['cases'] != selection['by_index'].get(str(index), []) or
                design['alf_actor_repeats'] != selection['actor_repeats']):
            raise ValueError('Probe cases or actor seeds differ from frozen train selection')
    drop = f'drop_{index}'
    rows, missing = [], []
    for case in design['cases']:
        spec, arms = memory_arms(origin, case)
        if train_source:
            planned = {item['path'] for item in source_design['selected_games'][spec['task']]}
            game = spec['original_memrl']['game']
            if game not in planned or not game.startswith('json_2.1.1/train/'):
                raise ValueError(f'Probe source is outside the frozen train split: {case}')
        if spec['benchmark'] != 'alfworld' or index >= len(spec['ids']):
            raise ValueError(f'Invalid ALFWorld memory index: {case}')
        source = {key:value for key,value in spec.items()
                  if key not in {'original_memrl', 'original_none', 'original_update'}}
        folder = output / 'alfworld' / spec['task'] / str(spec['repeat']) / f"episode_{spec['index']+1:03d}"
        if read(folder / 'source.json') != source:
            raise ValueError(f'Source binding changed: {case}')
        if sha(Path(plan['alf']['data_root']) / spec['original_memrl']['game']) != source['source_input_sha256']:
            raise ValueError(f'Game content changed: {case}')
        for actor_repeat in design['alf_actor_repeats']:
            target = folder / f'actor_repeat_{actor_repeat}'
            if not (target / 'summary.json').exists():
                missing.append(dict(case=case, actor_repeat=actor_repeat))
                continue
            summary = read(target / 'summary.json')
            if summary['ids'] != spec['ids'] or summary['actor_repeat'] != actor_repeat:
                raise ValueError(f'Summary identity changed: {target}')
            if set(summary['replay']) != {'full', drop}:
                raise ValueError(f'Unexpected arms: {target}')
            outcomes = {}
            for arm in ('full', drop):
                episode = read(target / arm / 'episode.json')
                if episode['status'] != 'complete':
                    raise ValueError(f'Incomplete replay: {target / arm}')
                if episode['game'] != spec['original_memrl']['game'] or episode['memory'] != arms[arm]:
                    raise ValueError(f'Replay input changed: {target / arm}')
                if episode['memory_sha256'] != source['arm_context_sha256'][arm]:
                    raise ValueError(f'Context hash changed: {target / arm}')
                if episode['seed'] != seed(actor_repeat, episode['game'], 0, 'actor'):
                    raise ValueError(f'Actor seed changed: {target / arm}')
                if episode['reward'] != summary['replay'][arm]['reward'] or episode['steps'] != summary['replay'][arm]['steps']:
                    raise ValueError(f'Summary reward differs from replay: {target / arm}')
                outcomes[arm] = float(episode['reward'])
            mid = spec['ids'][index]
            rows.append(dict(case=case, task=spec['task'], actor_repeat=actor_repeat,
                             memory_id=mid, memory_success=source['memory_features'][mid]['metadata']['success'],
                             memory_features=source['memory_features'][mid],
                             memory_text_sha256=source['memory_text_sha256'][mid],
                             source_input_sha256=source['source_input_sha256'],
                             retrieval_sha256=source['retrieval_sha256'],
                             snapshot_sha256=source['snapshot_sha256'],
                             native_post_update_q=spec['original_update']['q_updates'].get(mid),
                             original_full_reward=spec['original_memrl']['first_attempt'],
                             full=outcomes['full'], drop=outcomes[drop],
                             delta=outcomes['full']-outcomes[drop]))
    by_case = []
    for case in design['cases']:
        values = [row['delta'] for row in rows if row['case'] == case]
        by_case.append(dict(case=case, paired_repeats=len(values),
                            mean_delta=sum(values)/len(values) if values else None,
                            positive=sum(v > 0 for v in values),
                            negative=sum(v < 0 for v in values),
                            ties=sum(v == 0 for v in values)))
    return dict(design_sha256=sha(output / 'design.json'), source_split='train' if train_source else 'development',
                train_selection_sha256=sha(selection_path),
                expected=len(design['cases'])*len(design['alf_actor_repeats']),
                completed=len(rows), missing=missing, rows=rows, by_case=by_case,
                mean_delta=sum(row['delta'] for row in rows)/len(rows) if rows else None,
                caveat=('Official train-split fixed-snapshot labels; review input bindings before utility fitting.'
                        if train_source else
                        'Fixed evaluation snapshots selected for development; not an untouched utility training split.'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.output.resolve())
    save(args.output / 'analysis.json', report)
    print(f"Audited {report['completed']}/{report['expected']} paired replays; mean delta {report['mean_delta']}")


if __name__ == '__main__':
    main()
