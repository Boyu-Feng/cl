"""Audit held-out ALFWorld fixed-snapshot tests of causal-credit transfer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.experience_evolution.core import seed
from ttcl.icl_mem0_comparison.protocol import read, save, sha

from .audit_credit_train_extension import audit as audit_source
from .causal_credit_cache import decision
from .credit_probe import memory_arms
from .select_credit_cache_transfer import select


def analyze(outputs: list[Path], selection_path: Path) -> dict:
    frozen = read(selection_path)
    source = Path(frozen['source'])
    cache_path = Path(frozen['cache_path'])
    donors = [Path(path) for path in frozen['donor_paths']]
    if (frozen != select(source, cache_path, donors) or
            frozen['selection_code_sha256'] !=
            sha(Path(__file__).with_name('select_credit_cache_transfer.py'))):
        raise ValueError('Held-out selection changed')
    source_audit = audit_source(source)
    if not source_audit['complete'] or source_audit['missing']:
        raise ValueError('Held-out source changed')
    cache = read(cache_path)
    plan = read(source / 'plan.json')
    chosen = {item['case']: item for item in frozen['selected']}
    if len(chosen) != len(frozen['selected']):
        raise ValueError('Duplicate held-out case')
    expected_positions = {int(key) for key in frozen['by_index']}
    if {read(output / 'design.json')['paired_drop_index'] for output in outputs} \
            != expected_positions:
        raise ValueError('Probe positions differ from frozen selection')
    rows, missing = [], []
    full_rewards = {}
    for output in outputs:
        design = read(output / 'design.json')
        position = design['paired_drop_index']
        if (Path(design['origin']) != source or
                design['cases'] != frozen['by_index'][str(position)] or
                design['alf_actor_repeats'] != frozen['actor_repeats'] or
                design['origin_plan_sha256'] != sha(source / 'plan.json')):
            raise ValueError('Probe design differs from selected new inputs')
        manifest = read(output / 'training_selection.json')
        if (manifest['selection_path'] != str(selection_path) or
                manifest['selection_sha256'] != sha(selection_path) or
                manifest['design_sha256'] != sha(output / 'design.json')):
            raise ValueError('Probe selection binding changed')
        freeze = read(output / 'execution_freeze.json')
        if (freeze['selection_sha256'] != sha(selection_path) or
                freeze['design_sha256'] != sha(output / 'design.json')):
            raise ValueError('Probe execution freeze changed')
        for name, digest in freeze['files'].items():
            path = Path(__file__).with_name(name)
            if sha(path) != digest or sha(output / 'source' / name) != digest:
                raise ValueError(f'Frozen probe source changed: {name}')
        for case in design['cases']:
            selected = chosen[case]
            spec, arms = memory_arms(source, case)
            if (decision(spec, cache)['drop_position'] != position or
                    spec['ids'][position] != selected['memory_id'] or
                    spec['memory_text_sha256'][selected['memory_id']] !=
                    selected['memory_text_sha256'] or
                    spec['source_input_sha256'] != selected['input_sha256'] or
                    spec['retrieval_sha256'] != selected['retrieval_sha256'] or
                    spec['snapshot_sha256'] != selected['snapshot_sha256']):
                raise ValueError('Selected memory or source content changed')
            game = spec['original_memrl']['game']
            if (not game.startswith('json_2.1.1/train/') or
                    sha(Path(plan['alf']['data_root']) / game) !=
                    spec['source_input_sha256']):
                raise ValueError('Probe game is not bound official train content')
            folder = (output / 'alfworld' / spec['task'] / str(spec['repeat']) /
                      f"episode_{spec['index'] + 1:03d}")
            source_record = {key: value for key, value in spec.items()
                             if key not in {'original_memrl', 'original_none',
                                            'original_update'}}
            if read(folder / 'source.json') != source_record:
                raise ValueError('Probe source record changed')
            drop = f'drop_{position}'
            for actor_repeat in design['alf_actor_repeats']:
                target = folder / f'actor_repeat_{actor_repeat}'
                if not (target / 'summary.json').exists():
                    missing.append(dict(case=case, actor_repeat=actor_repeat))
                    continue
                summary = read(target / 'summary.json')
                if (summary['ids'] != spec['ids'] or
                        summary['actor_repeat'] != actor_repeat or
                        set(summary['replay']) != {'full', drop}):
                    raise ValueError('Probe arms or actor repeat changed')
                rewards = {}
                for arm in ('full', drop):
                    episode = read(target / arm / 'episode.json')
                    if (episode['status'] != 'complete' or
                            episode['game'] != game or
                            episode['memory'] != arms[arm] or
                            episode['memory_sha256'] !=
                            source_record['arm_context_sha256'][arm] or
                            episode['seed'] !=
                            seed(actor_repeat, game, 0, 'actor') or
                            episode['reward'] !=
                            summary['replay'][arm]['reward'] or
                            episode['steps'] !=
                            summary['replay'][arm]['steps']):
                        raise ValueError('Probe input, seed or outcome changed')
                    rewards[arm] = float(episode['reward'])
                full_key = (case, actor_repeat)
                if (full_key in full_rewards and
                        full_rewards[full_key] != rewards['full']):
                    raise ValueError('Full arm differed between probe positions')
                full_rewards[full_key] = rewards['full']
                rows.append(dict(case=case, task=spec['task'],
                                 actor_repeat=actor_repeat,
                                 memory_id=selected['memory_id'],
                                 memory_text_sha256=
                                 selected['memory_text_sha256'],
                                 source_input_sha256=
                                 selected['input_sha256'],
                                 position=position,
                                 full=rewards['full'], drop=rewards[drop],
                                 delta=rewards['full'] - rewards[drop]))
    expected = len(chosen) * len(frozen['actor_repeats'])
    return dict(schema='causal_credit_cache_transfer_audit_v1',
                selection_sha256=sha(selection_path),
                cache_sha256=sha(cache_path),
                source_design_sha256=sha(source / 'design.json'),
                selected_games=len(chosen), expected=expected,
                completed=len(rows), missing=missing, rows=rows,
                caveat='Fixed-snapshot first-attempt transfer on new official train games; no online-chain result')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', nargs='*', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = analyze([path.resolve() for path in args.outputs],
                     args.selection.resolve())
    save(args.report, report)
    print(json.dumps({key: (len(value) if key in {'rows', 'missing'} else value)
                      for key, value in report.items()
                      if key not in {'rows', 'caveat'}}, sort_keys=True))


if __name__ == '__main__':
    main()
