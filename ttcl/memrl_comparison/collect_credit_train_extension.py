"""Continue audited native ALFWorld train memory banks onto a new group.

This collects reward-only source trajectories for later counterfactual probes.
It does not create or reuse supervised annotation targets.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import traceback

from ttcl.icl_mem0_comparison.protocol import Client, read, save, sha

from .audit_credit_train_source import audit as audit_prior
from .collect_credit_train import SOURCE_FILES, TTCL_ROOT
from .memory import Embedder, Memory
from .worker import alf_cell


FILES = (*SOURCE_FILES,
         'memrl_comparison/collect_credit_train_extension.py',
         'memrl_comparison/audit_credit_train_extension.py')


def prepare(prior: Path, training: Path, output: Path,
            url: str, group_index: int = 2) -> tuple[dict, dict]:
    if output.exists():
        plan, design = read(output / 'plan.json'), read(output / 'design.json')
        if (design['prior'] != str(prior) or
                design['training_plan'] != str(training) or
                design['url'] != url or design['group_index'] != group_index):
            raise ValueError('Existing extension design differs')
        return plan, design
    prior_audit = audit_prior(prior)
    if (not prior_audit['complete'] or prior_audit['missing'] or
            prior_audit['audited_pairs'] != prior_audit['expected_pairs']):
        raise ValueError('Previous train source is not complete')
    prior_plan, prior_design = read(prior / 'plan.json'), read(prior / 'design.json')
    historical = read(training)
    if (sha(training) != prior_plan['credit_training_source']['training_plan_sha256'] or
            group_index < prior_plan['credit_training_source']['groups_per_family'] or
            prior_plan['credit_training_source']['repeat'] != prior_design['repeat']):
        raise ValueError('Historical train plan or extension group changed')
    grouped = {}
    for group in historical['training']:
        grouped.setdefault(group['family'], []).append(group)
    if set(grouped) != set(prior_design['families']):
        raise ValueError('Train family set changed')
    known = {item['sha256'] for games in prior_design['selected_games'].values()
             for item in games}
    original = Path(prior_design['origin'])
    original_plan = read(original / 'plan.json')
    eval_hashes = {task['sha256'] for sequence in original_plan['alf']['sequences']
                   for task in sequence['tasks']}
    selected = {}
    bootstrap = {}
    repeat = prior_design['repeat']
    data_root = Path(prior_plan['alf']['data_root'])
    for family in prior_design['families']:
        groups = grouped[family]
        if group_index >= len(groups):
            raise ValueError(f'No group {group_index} for {family}')
        entries = []
        for game in groups[group_index]['games']:
            if not game.startswith('json_2.1.1/train/'):
                raise ValueError(f'Non-train game: {game}')
            digest = sha(data_root / game)
            if digest in known or digest in eval_hashes:
                raise ValueError(f'New source overlaps old source or evaluation: {game}')
            known.add(digest)
            entries.append(dict(path=game, sha256=digest, family=family,
                                training_group=groups[group_index]['id']))
        selected[family] = entries
        last_index = len(prior_design['selected_games'][family])
        snapshot = (prior / 'runs' / 'alfworld' / family / str(repeat) /
                    'memrl' / f'episode_{last_index:03d}' /
                    'memory_after.json')
        prior_row = snapshot.parent / 'row.json'
        if sha(snapshot) != read(prior_row)['memory_after_sha256']:
            raise ValueError(f'Prior family bank changed: {family}')
        bootstrap[family] = dict(index=last_index,
                                 snapshot=str(snapshot),
                                 sha256=sha(snapshot),
                                 prior_row_sha256=sha(prior_row))
    plan = copy.deepcopy(prior_plan)
    plan['url'] = url
    plan['alf']['actor_url'] = url
    plan['alf']['sequences'] = [dict(family=family, repeat=repeat,
                                    tasks=selected[family])
                                for family in prior_design['families']]
    plan['credit_training_extension'] = dict(prior=str(prior),
                                             prior_design_sha256=sha(prior / 'design.json'),
                                             prior_complete_sha256=sha(prior / 'complete.json'),
                                             group_index=group_index,
                                             split='train',
                                             policy='native MemRL continuation; no reviewed labels')
    output.mkdir(parents=True)
    save(output / 'plan.json', plan)
    design = dict(plan_sha256=sha(output / 'plan.json'), prior=str(prior),
                  training_plan=str(training), training_plan_sha256=sha(training),
                  url=url, repeat=repeat, group_index=group_index,
                  families=prior_design['families'], selected_games=selected,
                  bootstrap=bootstrap,
                  source_sha256={name:sha(TTCL_ROOT / name) for name in FILES},
                  note='Official train group after prior source; bootstrap bank is copied and hash-bound, unscored')
    save(output / 'design.json', design)
    for name in FILES:
        destination = output / 'source' / 'ttcl' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TTCL_ROOT / name, destination)
    for family in design['families']:
        boot = design['bootstrap'][family]
        target = (output / 'runs' / 'alfworld' / family / str(repeat) /
                  'memrl' / f"episode_{boot['index']:03d}")
        target.mkdir(parents=True)
        shutil.copy2(boot['snapshot'], target / 'memory_after.json')
        save(target / 'bootstrap_lineage.json', boot)
    return plan, design


def collect(output: Path, plan: dict, design: dict) -> None:
    if sha(output / 'plan.json') != design['plan_sha256']:
        raise ValueError('Extension plan changed')
    if (sha(Path(design['prior']) / 'design.json') !=
            plan['credit_training_extension']['prior_design_sha256'] or
            sha(Path(design['prior']) / 'complete.json') !=
            plan['credit_training_extension']['prior_complete_sha256'] or
            sha(Path(design['training_plan'])) != design['training_plan_sha256']):
        raise ValueError('Train lineage changed')
    for name, expected in design['source_sha256'].items():
        if (sha(TTCL_ROOT / name) != expected or
                sha(output / 'source' / 'ttcl' / name) != expected):
            raise ValueError(f'Frozen collector code changed: {name}')
    expected_cells = sum(len(items) for items in design['selected_games'].values()) * 2
    completed = 0
    client = Client(plan, design['repeat'])
    embedder = Embedder(plan['embedding'])
    for family in design['families']:
        directory = output / 'runs' / 'alfworld' / family / str(design['repeat'])
        boot = design['bootstrap'][family]
        bank_path = (directory / 'memrl' / f"episode_{boot['index']:03d}" /
                     'memory_after.json')
        if (sha(Path(boot['snapshot'])) != boot['sha256'] or
                sha(bank_path) != boot['sha256']):
            raise ValueError(f'Bootstrap bank changed: {family}')
        memory = Memory(plan, client, directory / 'memrl' / 'memory',
                        plan['calibration']['alfworld'], embedder=embedder)
        memory.restore(bank_path)
        for offset, item in enumerate(design['selected_games'][family], 1):
            index = boot['index'] + offset
            if sha(Path(plan['alf']['data_root']) / item['path']) != item['sha256']:
                raise ValueError(f'New train input changed: {item["path"]}')
            for arm, bank in (('none', None), ('memrl', memory)):
                target = directory / arm / f'episode_{index:03d}'
                if (target / 'row.json').exists():
                    row = read(target / 'row.json')
                    if bank is not None:
                        if sha(target / 'memory_after.json') != row['memory_after_sha256']:
                            raise ValueError(f'Completed memory snapshot changed: {target}')
                        bank.restore(target / 'memory_after.json')
                elif target.exists():
                    raise RuntimeError(f'Interrupted partial cell requires review: {target}')
                else:
                    try:
                        row = alf_cell(plan, client, bank, item, design['repeat'],
                                       arm, target)
                    except Exception as exc:
                        save(output / 'failure.json', dict(family=family,
                            index=index, arm=arm, target=str(target),
                            error=repr(exc), traceback=traceback.format_exc()))
                        raise
                if (row['status'] != 'complete' or row['input_sha256'] != item['sha256'] or
                        row['game'] != item['path']):
                    save(output / 'failure.json', dict(family=family,
                        index=index, arm=arm, target=str(target), row=row))
                    raise RuntimeError(f'Train extension cell failed: {target}')
                completed += 1
                save(output / 'progress.json', dict(completed=completed,
                                                     expected=expected_cells))
                print(json.dumps(dict(family=family, index=index, arm=arm,
                                      reward=row['reward'], attempts=row['attempts'])),
                      flush=True)
    save(output / 'complete.json', dict(completed=completed,
                                        expected=expected_cells,
                                        design_sha256=sha(output / 'design.json')))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--training-plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', required=True)
    parser.add_argument('--group-index', type=int, default=2)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    plan, design = prepare(args.prior.resolve(), args.training_plan.resolve(),
                           output, args.url, args.group_index)
    if not args.prepare_only:
        collect(output, plan, design)


if __name__ == '__main__':
    main()
