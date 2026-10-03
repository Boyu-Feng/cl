"""Audit current-chain-seed counterfactual Q labels for cache-hit memories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.experience_evolution.core import seed
from ttcl.icl_mem0_comparison.protocol import read, save, sha

from .audit_credit_train_extension import audit as audit_source
from .credit_probe import memory_arms
from .select_credit_cache_transfer import select


ACTOR_REPEAT = 92721


def audit(output: Path, selection_path: Path) -> dict:
    selection = read(selection_path)
    source = Path(selection['source'])
    if (selection != select(source, Path(selection['cache_path']),
                            [Path(path) for path in selection['donor_paths']]) or
            not audit_source(source)['complete']):
        raise ValueError('New train source or cache selection changed')
    design = read(output / 'design.json')
    expected_cases = selection['by_index'].get('0', [])
    if (design['origin'] != str(source) or
            design['origin_plan_sha256'] != sha(source / 'plan.json') or
            design['cases'] != expected_cases or
            design['paired_drop_index'] != 0 or
            design['alf_actor_repeats'] != [ACTOR_REPEAT]):
        raise ValueError('Online-seed probe differs from frozen hit set')
    manifest = read(output / 'training_selection.json')
    if (manifest['selection_path'] != str(selection_path) or
            manifest['selection_sha256'] != sha(selection_path) or
            manifest['design_sha256'] != sha(output / 'design.json') or
            manifest['actor_repeat'] != ACTOR_REPEAT):
        raise ValueError('Online-seed selection binding changed')
    freeze = read(output / 'execution_freeze.json')
    if (freeze['selection_sha256'] != sha(selection_path) or
            freeze['design_sha256'] != sha(output / 'design.json')):
        raise ValueError('Online-seed source freeze changed')
    for name, digest in freeze['files'].items():
        local = Path(__file__).with_name(name)
        if sha(local) != digest or sha(output / 'source' / name) != digest:
            raise ValueError(f'Frozen probe code changed: {name}')
    chosen = {row['case']:row for row in selection['selected']}
    rows, missing = [], []
    for case in expected_cases:
        item = chosen[case]
        spec, arms = memory_arms(source, case)
        if (spec['ids'][0] != item['memory_id'] or
                spec['memory_text_sha256'][item['memory_id']] !=
                item['memory_text_sha256'] or
                spec['source_input_sha256'] != item['input_sha256']):
            raise ValueError('Selected causal memory changed')
        folder = (output / 'alfworld' / spec['task'] / str(spec['repeat']) /
                  f"episode_{spec['index'] + 1:03d}")
        target = folder / f'actor_repeat_{ACTOR_REPEAT}'
        if not (target / 'summary.json').exists():
            missing.append(case)
            continue
        summary = read(target / 'summary.json')
        if (summary['ids'] != spec['ids'] or
                summary['actor_repeat'] != ACTOR_REPEAT or
                set(summary['replay']) != {'full', 'drop_0'}):
            raise ValueError('Online-seed arms changed')
        game = spec['original_memrl']['game']
        rewards, episodes = {}, {}
        for arm in ('full', 'drop_0'):
            episode = read(target / arm / 'episode.json')
            if (episode['status'] != 'complete' or
                    episode['game'] != game or
                    episode['memory'] != arms[arm] or
                    episode['memory_sha256'] !=
                    spec['arm_context_sha256'][arm] or
                    episode['seed'] !=
                    seed(ACTOR_REPEAT, game, 0, 'actor') or
                    episode['reward'] != summary['replay'][arm]['reward'] or
                    episode['steps'] != summary['replay'][arm]['steps']):
                raise ValueError('Online-seed actor context or reward changed')
            rewards[arm] = float(episode['reward'])
            episodes[arm] = episode
        native = read(source / 'runs' / case / 'attempt_1' / 'episode.json')
        native_retrieval = read(source / 'runs' / case / 'retrieval_1.json')
        native_update = read(source / 'runs' / case / 'update_1.json')
        if (native['status'] != 'complete' or
                native['memory'] != arms['full'] or
                native_retrieval['ids'] != spec['ids'] or
                item['memory_id'] not in native_update['q_updates']):
            raise ValueError('Native first attempt or Q update changed')
        native_full_match = (
            native['reward'] == episodes['full']['reward'] and
            [step['action'] for step in native['trajectory']] ==
            [step['action'] for step in episodes['full']['trajectory']])
        rows.append(dict(case=case, family=spec['task'],
                         memory_id=item['memory_id'],
                         memory_text_sha256=item['memory_text_sha256'],
                         source_input_sha256=item['input_sha256'],
                         actor_repeat=ACTOR_REPEAT,
                         native_full_match=native_full_match,
                         native_reward=native['reward'],
                         full=rewards['full'], drop=rewards['drop_0'],
                         marginal=rewards['full'] - rewards['drop_0']))
    return dict(schema='online_seed_counterfactual_q_audit_v1',
                selection_sha256=sha(selection_path),
                design_sha256=sha(output / 'design.json'),
                expected=len(expected_cases), completed=len(rows),
                missing=missing, rows=rows,
                caveat='First-attempt task reward marginal on frozen snapshots; only exact native-full matches can be attributed to the online seed')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.output.resolve(), args.selection.resolve())
    save(args.report, report)
    print(json.dumps({key:(len(value) if key in {'rows', 'missing'} else value)
                      for key, value in report.items()
                      if key not in {'rows', 'caveat'}}, sort_keys=True))


if __name__ == '__main__':
    main()
