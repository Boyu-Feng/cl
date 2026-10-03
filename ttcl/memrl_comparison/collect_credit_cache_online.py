"""Replay a frozen causal-credit cache through a full ALFWorld train chain.

The native source arm is already recorded.  This collector starts each family
from the same episode-6 memory bank, then lets the cache arm evolve its own
writer and Q state over three new official-train games.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import traceback

from ttcl.icl_mem0_comparison.protocol import Client, read, save, sha

from .audit_credit_train_extension import audit as audit_source
from .collect_credit_train import SOURCE_FILES, TTCL_ROOT
from .credit_cache_policy import CacheMemory
from .memory import Embedder
from .worker import alf_cell


FILES = (*SOURCE_FILES,
         'memrl_comparison/audit_credit_train_extension.py',
         'memrl_comparison/causal_credit_cache.py',
         'memrl_comparison/credit_cache_policy.py',
         'memrl_comparison/collect_credit_cache_online.py',
         'memrl_comparison/audit_credit_cache_online.py')


def prepare(source: Path, cache_path: Path, output: Path,
            url: str) -> tuple[dict, dict]:
    if output.exists():
        plan, design = read(output / 'plan.json'), read(output / 'design.json')
        if (design['source'] != str(source) or
                design['cache_path'] != str(cache_path) or
                design['url'] != url):
            raise ValueError('Existing online-cache design differs')
        return plan, design
    source_report = audit_source(source)
    if (not source_report['complete'] or source_report['missing'] or
            source_report['audited_pairs'] != source_report['expected_pairs']):
        raise ValueError('Native train source is incomplete')
    source_plan, source_design = read(source / 'plan.json'), read(source / 'design.json')
    cache = read(cache_path)
    if (cache['schema'] != 'causal_credit_cache_v1' or
            cache['source_code_sha256'] !=
            sha(TTCL_ROOT / 'memrl_comparison/causal_credit_cache.py')):
        raise ValueError('Frozen causal-credit cache changed')
    plan = copy.deepcopy(source_plan)
    plan['url'] = url
    plan['alf']['actor_url'] = url
    plan['credit_cache_online'] = dict(source=str(source),
                                       source_design_sha256=sha(source / 'design.json'),
                                       source_complete_sha256=sha(source / 'complete.json'),
                                       cache_sha256=sha(cache_path),
                                       policy='First-attempt exact-text negative-credit suppression; at most one memory')
    output.mkdir(parents=True)
    save(output / 'plan.json', plan)
    bootstrap = {}
    for family in source_design['families']:
        boot = source_design['bootstrap'][family]
        snapshot = (source / 'runs' / 'alfworld' / family /
                    str(source_design['repeat']) / 'memrl' /
                    f"episode_{boot['index']:03d}" / 'memory_after.json')
        if sha(snapshot) != boot['sha256']:
            raise ValueError(f'Native start bank changed: {family}')
        bootstrap[family] = dict(index=boot['index'], snapshot=str(snapshot),
                                 sha256=sha(snapshot))
    design = dict(plan_sha256=sha(output / 'plan.json'), source=str(source),
                  source_design_sha256=sha(source / 'design.json'),
                  source_complete_sha256=sha(source / 'complete.json'),
                  cache_path=str(cache_path), cache_sha256=sha(cache_path),
                  url=url, repeat=source_design['repeat'],
                  families=source_design['families'],
                  selected_games=source_design['selected_games'],
                  bootstrap=bootstrap,
                  source_sha256={name: sha(TTCL_ROOT / name) for name in FILES},
                  note='Official train online cache continuation, same root bank and game order as native source')
    save(output / 'design.json', design)
    for name in FILES:
        target = output / 'source' / 'ttcl' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TTCL_ROOT / name, target)
    for family, boot in bootstrap.items():
        target = (output / 'runs' / 'alfworld' / family /
                  str(design['repeat']) / 'cache' / f"episode_{boot['index']:03d}")
        target.mkdir(parents=True)
        shutil.copy2(boot['snapshot'], target / 'memory_after.json')
        save(target / 'bootstrap_lineage.json', boot)
    return plan, design


def collect(output: Path, plan: dict, design: dict) -> None:
    source = Path(design['source'])
    cache_path = Path(design['cache_path'])
    if (sha(output / 'plan.json') != design['plan_sha256'] or
            sha(source / 'design.json') != design['source_design_sha256'] or
            sha(source / 'complete.json') != design['source_complete_sha256'] or
            sha(cache_path) != design['cache_sha256']):
        raise ValueError('Online cache lineage changed')
    for name, expected in design['source_sha256'].items():
        if (sha(TTCL_ROOT / name) != expected or
                sha(output / 'source' / 'ttcl' / name) != expected):
            raise ValueError(f'Frozen online collector changed: {name}')
    cache = read(cache_path)
    client = Client(plan, design['repeat'])
    embedder = Embedder(plan['embedding'])
    expected = sum(len(items) for items in design['selected_games'].values())
    completed = 0
    for family in design['families']:
        directory = (output / 'runs' / 'alfworld' / family /
                     str(design['repeat']) / 'cache')
        boot = design['bootstrap'][family]
        bootstrap = directory / f"episode_{boot['index']:03d}" / 'memory_after.json'
        if (sha(Path(boot['snapshot'])) != boot['sha256'] or
                sha(bootstrap) != boot['sha256']):
            raise ValueError(f'Online root bank changed: {family}')
        memory = CacheMemory(plan, client, directory / 'memory',
                             plan['calibration']['alfworld'], embedder=embedder,
                             family=family, credit_cache=cache)
        memory.restore(bootstrap)
        for offset, item in enumerate(design['selected_games'][family], 1):
            index = boot['index'] + offset
            if sha(Path(plan['alf']['data_root']) / item['path']) != item['sha256']:
                raise ValueError(f'Official train game changed: {item["path"]}')
            target = directory / f'episode_{index:03d}'
            before = directory / f'episode_{index:03d}_memory_before.json'
            if (target / 'row.json').exists():
                row = read(target / 'row.json')
                if (sha(target / 'memory_after.json') !=
                        row['memory_after_sha256']):
                    raise ValueError('Completed cache bank changed')
                memory.restore(target / 'memory_after.json')
            elif target.exists() or before.exists():
                raise RuntimeError(f'Interrupted partial cache cell needs review: {target}')
            else:
                memory.snapshot(before)
                memory.begin_episode()
                try:
                    row = alf_cell(plan, client, memory, item, design['repeat'],
                                   'cache', target)
                except Exception as exc:
                    save(output / 'failure.json', dict(family=family, index=index,
                        target=str(target), error=repr(exc),
                        traceback=traceback.format_exc()))
                    raise
            if (row['status'] != 'complete' or row['game'] != item['path'] or
                    row['input_sha256'] != item['sha256']):
                save(output / 'failure.json', dict(family=family, index=index,
                                                  target=str(target), row=row))
                raise RuntimeError(f'Online cache cell failed: {target}')
            completed += 1
            save(output / 'progress.json', dict(completed=completed,
                                                expected=expected))
            first = read(target / 'retrieval_1.json')
            print(json.dumps(dict(family=family, index=index,
                                  reward=row['reward'], attempts=row['attempts'],
                                  suppressed=first['suppressed_ids'])), flush=True)
    save(output / 'complete.json', dict(completed=completed, expected=expected,
                                        design_sha256=sha(output / 'design.json')))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    plan, design = prepare(args.source.resolve(), args.cache.resolve(),
                           output, args.url)
    if not args.prepare_only:
        collect(output, plan, design)


if __name__ == '__main__':
    main()
