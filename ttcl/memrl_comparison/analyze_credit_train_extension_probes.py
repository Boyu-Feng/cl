"""Audit source-bound leave-one-out probes on new ALFWorld train inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.experience_evolution.core import seed
from ttcl.icl_mem0_comparison.protocol import read, save, sha

from .audit_credit_train_extension import audit as audit_source
from .credit_probe import memory_arms
from .select_credit_train_extension_probes import select


def analyze(output: Path) -> dict:
    design = read(output / 'design.json')
    selection_manifest = read(output / 'training_selection.json')
    selection_path = Path(selection_manifest['selection_path'])
    frozen = read(selection_path)
    source = Path(design['origin'])
    if (selection_manifest['selection_sha256'] != sha(selection_path) or
            selection_manifest['design_sha256'] != sha(output / 'design.json') or
            frozen != select(source) or
            frozen['selection_code_sha256'] !=
            sha(Path(__file__).with_name('select_credit_train_extension_probes.py')) or
            frozen['context_reconstruction_sha256'] !=
            sha(Path(memory_arms.__code__.co_filename)) or
            design['cases'] != frozen['by_index'].get(
                str(design['paired_drop_index']), []) or
            design['alf_actor_repeats'] != frozen['actor_repeats']):
        raise ValueError('Probe design differs from frozen train selection')
    source_audit = audit_source(source)
    if (not source_audit['complete'] or source_audit['missing'] or
            sha(source / 'plan.json') != design['origin_plan_sha256']):
        raise ValueError('New source lineage changed')
    freeze = read(output / 'execution_freeze.json')
    if (freeze['selection_sha256'] != sha(selection_path) or
            freeze['design_sha256'] != sha(output / 'design.json')):
        raise ValueError('Execution freeze changed')
    for name, digest in freeze['files'].items():
        path = Path(__file__).with_name(name)
        if sha(path) != digest or sha(output / 'source' / name) != digest:
            raise ValueError(f'Frozen probe source changed: {name}')
    plan = read(source / 'plan.json')
    drop_index = design['paired_drop_index']
    drop = f'drop_{drop_index}'
    source_design = read(source / 'design.json')
    rows, missing = [], []
    for case in design['cases']:
        spec, arms = memory_arms(source, case)
        game = spec['original_memrl']['game']
        planned = {item['path'] for item in
                   source_design['selected_games'][spec['task']]}
        if (spec['benchmark'] != 'alfworld' or
                drop_index >= len(spec['ids']) or
                game not in planned or
                not game.startswith('json_2.1.1/train/') or
                sha(Path(plan['alf']['data_root']) / game) !=
                spec['source_input_sha256']):
            raise ValueError(f'Invalid new train probe source: {case}')
        source_record = {key: value for key, value in spec.items()
                         if key not in {'original_memrl', 'original_none',
                                        'original_update'}}
        folder = (output / 'alfworld' / spec['task'] / str(spec['repeat']) /
                  f"episode_{spec['index'] + 1:03d}")
        if read(folder / 'source.json') != source_record:
            raise ValueError(f'Probe source record changed: {case}')
        for actor_repeat in design['alf_actor_repeats']:
            target = folder / f'actor_repeat_{actor_repeat}'
            if not (target / 'summary.json').exists():
                missing.append(dict(case=case, actor_repeat=actor_repeat))
                continue
            summary = read(target / 'summary.json')
            if (summary['ids'] != spec['ids'] or
                    summary['actor_repeat'] != actor_repeat or
                    set(summary['replay']) != {'full', drop}):
                raise ValueError(f'Probe branch set changed: {target}')
            outcomes = {}
            for arm in ('full', drop):
                episode = read(target / arm / 'episode.json')
                if (episode['status'] != 'complete' or
                        episode['game'] != game or
                        episode['memory'] != arms[arm] or
                        episode['memory_sha256'] !=
                        source_record['arm_context_sha256'][arm] or
                        episode['seed'] != seed(actor_repeat, game, 0, 'actor') or
                        episode['reward'] != summary['replay'][arm]['reward'] or
                        episode['steps'] != summary['replay'][arm]['steps']):
                    raise ValueError(f'Probe input, seed or reward changed: {target}')
                outcomes[arm] = float(episode['reward'])
            memory_id = spec['ids'][drop_index]
            rows.append(dict(case=case, task=spec['task'],
                             actor_repeat=actor_repeat,
                             memory_id=memory_id,
                             memory_text_sha256=
                             spec['memory_text_sha256'][memory_id],
                             source_input_sha256=spec['source_input_sha256'],
                             retrieval_sha256=spec['retrieval_sha256'],
                             snapshot_sha256=spec['snapshot_sha256'],
                             memory_features=spec['memory_features'][memory_id],
                             full=outcomes['full'], drop=outcomes[drop],
                             delta=outcomes['full'] - outcomes[drop]))
    return dict(schema='alf_credit_train_extension_probe_v1',
                design_sha256=sha(output / 'design.json'),
                selection_sha256=sha(selection_path),
                expected=len(design['cases']) * len(design['alf_actor_repeats']),
                completed=len(rows), missing=missing, rows=rows,
                caveat='Official ALFWorld train inputs selected by retrieval count and content hash; fixed-snapshot first-attempt marginal only')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.output.resolve())
    save(args.output / 'analysis.json', report)
    print(json.dumps({key: (len(value) if key in {'rows', 'missing'} else value)
                      for key, value in report.items() if key != 'caveat'},
                     sort_keys=True))


if __name__ == '__main__':
    main()
