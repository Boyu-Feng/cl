"""Audit extra paired seeds for outcome-selected negative training memories.

These probes reuse the same source inputs. They test within-case replication,
not generalization to new tasks or independent benchmark performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from ttcl.experience_evolution.core import seed as alf_seed
from ttcl.icl_mem0_comparison.protocol import read, sha

from .analyze_cl_credit_probe import analyze as analyze_cl
from .credit_probe import memory_arms


def audit(dataset: dict, output: Path) -> dict:
    expected_dataset_hash = dataset['dataset_sha256']
    body = {key: value for key, value in dataset.items()
            if key != 'dataset_sha256'}
    if hashlib.sha256(json.dumps(body, sort_keys=True,
                                 allow_nan=False).encode()).hexdigest() != expected_dataset_hash:
        raise ValueError('Prior training dataset content changed')
    design = read(output / 'design.json')
    selection = read(output / 'selection_reason.json')
    if (selection['prior_dataset_sha256'] != expected_dataset_hash or
            selection['design_sha256'] != sha(output / 'design.json') or
            selection['probe_code_sha256'] != sha(Path(__file__).with_name('credit_probe.py')) or
            design['cases'] != [selection['case']]):
        raise ValueError('Replication selection or probe code changed')
    matches = [row for row in dataset['examples']
               if row['binding'][1] == selection['case'] and
               row['binding'][2] == selection['memory_id']]
    if len(matches) != 1:
        raise ValueError('Selected memory lacks a unique prior example')
    prior = matches[0]
    if (prior['binding'][3] != selection['memory_text_sha256'] or
            prior['actor_repeats'] != selection['prior_actor_repeats'] or
            prior['raw_deltas'] != selection['prior_raw_deltas'] or
            not sum(delta < 0 for delta in prior['raw_deltas']) >= 2 or
            any(delta > 0 for delta in prior['raw_deltas'])):
        raise ValueError('Prior negative selection changed')
    repeats = selection['new_actor_repeats']
    if (repeats != design.get('alf_actor_repeats',
                             design.get('cl_sampling_repeats')) or
            len(repeats) != len(set(repeats)) or
            set(repeats) & set(prior['actor_repeats'])):
        raise ValueError('Extra seeds are missing, duplicate, or reused')
    origin = Path(design['origin'])
    if sha(origin / 'plan.json') != design['origin_plan_sha256']:
        raise ValueError('Origin plan changed')
    spec, arms = memory_arms(origin, selection['case'])
    source_binding = (spec['source_input_sha256'], spec['retrieval_sha256'],
                      spec['snapshot_sha256'])
    if (spec['memory_text_sha256'][selection['memory_id']] != prior['binding'][3] or
            source_binding != tuple(prior['binding'][4:7])):
        raise ValueError('Source input, retrieval, or memory snapshot changed')
    position = spec['ids'].index(selection['memory_id'])
    new_deltas = []
    if spec['benchmark'] == 'clbench':
        report = analyze_cl(output)
        if report['completed'] != report['expected'] or report['missing']:
            raise ValueError('Incomplete CLBench paired replays')
        for row in report['rows']:
            if row['case'] != selection['case']:
                raise ValueError('Unexpected CLBench case')
            new_deltas.append(row['delta_by_memory'][position])
    elif spec['benchmark'] == 'alfworld':
        plan = read(origin / 'plan.json')
        game = spec['original_memrl']['game']
        if (not game.startswith('json_2.1.1/train/') or
                sha(Path(plan['alf']['data_root']) / game) !=
                spec['original_memrl']['input_sha256'] or
                design['paired_drop_index'] != position):
            raise ValueError('ALFWorld split, game, or memory position changed')
        folder = output / 'alfworld' / spec['task'] / str(spec['repeat']) / \
            f"episode_{spec['index'] + 1:03d}"
        expected_source = {key: value for key, value in spec.items()
                           if key not in {'original_memrl', 'original_none',
                                          'original_update'}}
        if read(folder / 'source.json') != expected_source:
            raise ValueError('ALFWorld source binding changed')
        drop = f'drop_{position}'
        for repeat in repeats:
            target = folder / f'actor_repeat_{repeat}'
            summary = read(target / 'summary.json')
            if (summary['ids'] != spec['ids'] or
                    summary['actor_repeat'] != repeat or
                    set(summary['replay']) != {'full', drop}):
                raise ValueError('ALFWorld branch summary changed')
            rewards = {}
            for arm in ('full', drop):
                episode = read(target / arm / 'episode.json')
                if (episode['status'] != 'complete' or episode['game'] != game or
                        episode['memory'] != arms[arm] or
                        episode['memory_sha256'] != spec['arm_context_sha256'][arm] or
                        episode['seed'] != alf_seed(repeat, game, 0, 'actor') or
                        episode['reward'] != summary['replay'][arm]['reward'] or
                        episode['steps'] != summary['replay'][arm]['steps']):
                    raise ValueError('ALFWorld replay content or reward changed')
                rewards[arm] = float(episode['reward'])
            new_deltas.append(rewards['full'] - rewards[drop])
    else:
        raise ValueError('Unsupported benchmark')
    if len(new_deltas) != len(repeats):
        raise ValueError('Missing paired seed result')
    combined = prior['raw_deltas'] + new_deltas
    return dict(case=selection['case'], benchmark=spec['benchmark'],
                memory_id=selection['memory_id'],
                prior_dataset_sha256=expected_dataset_hash,
                design_sha256=selection['design_sha256'],
                prior_seeds=prior['actor_repeats'],
                new_seeds=repeats, prior_deltas=prior['raw_deltas'],
                new_deltas=new_deltas, combined_deltas=combined,
                combined_mean=statistics.fmean(combined),
                combined_negative=sum(value < 0 for value in combined),
                combined_positive=sum(value > 0 for value in combined),
                combined_ties=sum(value == 0 for value in combined),
                caveat='Outcome-selected fixed-snapshot training cases; no new-task or online-policy claim')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(read(args.dataset), args.output.resolve())
    target = args.output / 'replication_audit.json'
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
