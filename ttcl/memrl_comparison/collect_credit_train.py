"""Collect source-bound native MemRL chains on official ALFWorld train games.

These trajectories are reward-only sources for later reviewed utility probes.
No trajectory text is converted into a supervised annotation target here.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import traceback

from ttcl.icl_mem0_comparison.protocol import Client, read, save, sha
from ttcl.memrl_comparison.memory import Embedder, Memory
from ttcl.memrl_comparison.worker import alf_cell


SOURCE_FILES = (
    'memrl_comparison/collect_credit_train.py',
    'memrl_comparison/credit_probe.py',
    'memrl_comparison/worker.py',
    'memrl_comparison/memory.py',
    'alfworld_comparison/environment.py',
    'experience_evolution/environment.py',
    'experience_evolution/core.py',
    'icl_mem0_comparison/protocol.py',
)
TTCL_ROOT = Path(__file__).resolve().parents[1]


def selected_games(training_plan: dict, data_root: Path, groups_per_family: int) -> list[dict]:
    if groups_per_family < 1:
        raise ValueError('groups_per_family must be positive')
    groups = {}
    seen_paths = set()
    for group in training_plan['training']:
        family = group['family']
        bucket = groups.setdefault(family, [])
        if len(bucket) >= groups_per_family:
            continue
        games = []
        for game in group['games']:
            relative = Path(game)
            if relative.is_absolute() or relative.parts[:2] != ('json_2.1.1', 'train'):
                raise ValueError(f'Not an official train game: {game}')
            path = (data_root / relative).resolve()
            if path in seen_paths or not path.is_file() or not path.is_relative_to(data_root.resolve()):
                raise ValueError(f'Missing, repeated or escaping train game: {game}')
            seen_paths.add(path)
            games.append(dict(path=game, sha256=sha(path), family=family,
                              training_group=group['id']))
        bucket.append(games)
    if len(groups) != 6 or any(len(value) != groups_per_family for value in groups.values()):
        raise ValueError('Need the same number of historical train groups in all six families')
    return [dict(family=family, tasks=[game for group in collection for game in group])
            for family, collection in sorted(groups.items())]


def prepare(origin: Path, training: Path, output: Path, url: str,
            repeat: int, groups_per_family: int) -> tuple[dict, dict]:
    if output.exists():
        plan, design = read(output / 'plan.json'), read(output / 'design.json')
        if (design['origin'] != str(origin) or design['training_plan'] != str(training)
                or design['url'] != url or design['repeat'] != repeat
                or plan['credit_training_source']['groups_per_family'] != groups_per_family):
            raise ValueError('Existing train-source design differs')
        return plan, design
    base = read(origin / 'plan.json')
    old = read(training)
    if Path(base['alf']['data_root']).resolve() != Path(old['data_root']).resolve():
        raise ValueError('ALFWorld data roots differ')
    sequences = selected_games(old, Path(base['alf']['data_root']), groups_per_family)
    eval_hashes = {task['sha256'] for seq in base['alf']['sequences'] for task in seq['tasks']}
    if any(task['sha256'] in eval_hashes for seq in sequences for task in seq['tasks']):
        raise ValueError('Selected train game overlaps frozen valid_unseen evaluation content')
    plan = copy.deepcopy(base)
    plan['url'] = url
    plan['alf']['actor_url'] = url
    plan['alf']['sequences'] = [dict(family=seq['family'], repeat=repeat,
                                    tasks=seq['tasks']) for seq in sequences]
    plan['q_min_threshold'] = plan['q_min_thresholds']['alfworld']
    plan['credit_training_source'] = dict(split='train', repeat=repeat,
                                           groups_per_family=groups_per_family,
                                           original_plan_sha256=sha(origin / 'plan.json'),
                                           training_plan_sha256=sha(training),
                                           policy='native MemRL, independent family banks; no utility labels or annotation targets')
    output.mkdir(parents=True)
    save(output / 'plan.json', plan)
    design = dict(plan_sha256=sha(output / 'plan.json'), origin=str(origin),
                  training_plan=str(training), url=url, repeat=repeat,
                  families=[seq['family'] for seq in sequences],
                  selected_games={seq['family']:seq['tasks'] for seq in sequences},
                  source_sha256={name:sha(TTCL_ROOT / name) for name in SOURCE_FILES},
                  note='Official ALFWorld train split only; rewards are environment feedback, not reviewed text targets')
    save(output / 'design.json', design)
    source = output / 'source' / 'ttcl'
    source.mkdir(parents=True)
    for name in design['source_sha256']:
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TTCL_ROOT / name, destination)
    return plan, design


def collect(output: Path, plan: dict, design: dict) -> None:
    if sha(output / 'plan.json') != design['plan_sha256']:
        raise ValueError('Frozen train-source plan changed')
    if sha(Path(design['origin']) / 'plan.json') != plan['credit_training_source']['original_plan_sha256']:
        raise ValueError('Original MemRL plan changed')
    if sha(Path(design['training_plan'])) != plan['credit_training_source']['training_plan_sha256']:
        raise ValueError('Historical train selection changed')
    for name, expected in design['source_sha256'].items():
        if sha(output / 'source' / 'ttcl' / name) != expected or sha(TTCL_ROOT / name) != expected:
            raise ValueError(f'Collector source changed: {name}')
    expected_cells = sum(len(items) for items in design['selected_games'].values()) * 2
    completed = 0
    client = Client(plan, design['repeat'])
    embedder = Embedder(plan['embedding'])
    for family in design['families']:
        directory = output / 'runs' / 'alfworld' / family / str(design['repeat'])
        memory = Memory(plan, client, directory / 'memrl' / 'memory',
                        plan['calibration']['alfworld'], embedder=embedder)
        for index, item in enumerate(design['selected_games'][family], 1):
            path = Path(plan['alf']['data_root']) / item['path']
            if sha(path) != item['sha256']:
                raise ValueError(f'Train input content changed: {path}')
            for arm, bank in (('none', None), ('memrl', memory)):
                target = directory / arm / f'episode_{index:03d}'
                if (target / 'row.json').exists():
                    row = read(target / 'row.json')
                    if bank is not None:
                        if sha(target / 'memory_after.json') != row['memory_after_sha256']:
                            raise ValueError(f'Memory snapshot changed: {target}')
                        bank.restore(target / 'memory_after.json')
                elif target.exists():
                    raise RuntimeError(f'Interrupted partial cell requires review: {target}')
                else:
                    try:
                        row = alf_cell(plan, client, bank, item, design['repeat'],
                                       'memrl' if bank else 'none', target)
                    except Exception as exc:
                        save(output / 'failure.json', dict(family=family, index=index,
                            arm=arm, target=str(target), error=repr(exc),
                            traceback=traceback.format_exc()))
                        raise
                if row['input_sha256'] != item['sha256'] or row['status'] != 'complete':
                    save(output / 'failure.json', dict(family=family, index=index, arm=arm,
                                                       target=str(target), row=row))
                    raise RuntimeError(f'Train source cell failed: {target}')
                completed += 1
                save(output / 'progress.json', dict(completed=completed, expected=expected_cells))
                print(json.dumps(dict(family=family, index=index, arm=arm,
                                      reward=row['reward'], attempts=row['attempts'])), flush=True)
    save(output / 'complete.json', dict(completed=completed, expected=expected_cells,
                                        design_sha256=sha(output / 'design.json')))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin', type=Path, required=True)
    parser.add_argument('--training-plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', required=True)
    parser.add_argument('--repeat', type=int, default=92721)
    parser.add_argument('--groups-per-family', type=int, default=2)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    plan, design = prepare(args.origin.resolve(), args.training_plan.resolve(),
                           output, args.url, args.repeat, args.groups_per_family)
    if not args.prepare_only:
        collect(output, plan, design)


if __name__ == '__main__':
    main()
