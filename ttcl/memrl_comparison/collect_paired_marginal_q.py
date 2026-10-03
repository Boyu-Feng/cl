"""Run sampled paired-marginal Q over a new official ALFWorld train chain."""
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
from .memory import Embedder
from .paired_marginal_q import (CALL_BUDGET, COST_WEIGHT, PairedQMemory,
                                alf_cell_paired_q)


FILES = (*SOURCE_FILES,
         'memrl_comparison/audit_credit_train_extension.py',
         'memrl_comparison/paired_marginal_q.py',
         'memrl_comparison/collect_paired_marginal_q.py',
         'memrl_comparison/audit_paired_marginal_q.py')


def prepare(source: Path, output: Path, url: str) -> tuple[dict, dict]:
    if output.exists():
        plan, design = read(output / 'plan.json'), read(output / 'design.json')
        if design['source'] != str(source) or design['url'] != url:
            raise ValueError('Existing paired-Q design differs')
        return plan, design
    source_report = audit_source(source)
    if (not source_report['complete'] or source_report['missing'] or
            source_report['audited_pairs'] != source_report['expected_pairs']):
        raise ValueError('Native source for paired-Q is incomplete')
    source_plan, source_design = read(source / 'plan.json'), read(source / 'design.json')
    if source_plan['alf']['max_steps'] != CALL_BUDGET:
        raise ValueError('Paired-Q call budget differs from official task budget')
    plan = copy.deepcopy(source_plan)
    plan['url'] = url
    plan['alf']['actor_url'] = url
    plan['paired_marginal_q'] = dict(
        source=str(source), source_design_sha256=sha(source / 'design.json'),
        source_complete_sha256=sha(source / 'complete.json'),
        selection='first retrieved experience on first attempt of every task',
        extra_actor_runs_per_task=1,
        cost_weight=COST_WEIGHT, call_budget=CALL_BUDGET,
        q_rule='selected memory gets clipped paired reward-minus-cost advantage; other retrieved memories retain native official reward')
    output.mkdir(parents=True)
    save(output / 'plan.json', plan)
    bootstrap = {}
    for family in source_design['families']:
        boot = source_design['bootstrap'][family]
        snapshot = (source / 'runs' / 'alfworld' / family /
                    str(source_design['repeat']) / 'memrl' /
                    f"episode_{boot['index']:03d}" / 'memory_after.json')
        if sha(snapshot) != boot['sha256']:
            raise ValueError('Source root bank changed')
        bootstrap[family] = dict(index=boot['index'], snapshot=str(snapshot),
                                 sha256=sha(snapshot))
    design = dict(plan_sha256=sha(output / 'plan.json'), source=str(source),
                  source_design_sha256=sha(source / 'design.json'),
                  source_complete_sha256=sha(source / 'complete.json'),
                  url=url, repeat=source_design['repeat'],
                  families=source_design['families'],
                  selected_games=source_design['selected_games'],
                  bootstrap=bootstrap,
                  source_sha256={name:sha(TTCL_ROOT / name) for name in FILES},
                  note='Reward-only paired-Q train chain; counterfactual arms never update writer or Q')
    save(output / 'design.json', design)
    for name in FILES:
        target = output / 'source' / 'ttcl' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TTCL_ROOT / name, target)
    for family, boot in bootstrap.items():
        target = (output / 'runs' / 'alfworld' / family /
                  str(design['repeat']) / 'paired_q' /
                  f"episode_{boot['index']:03d}")
        target.mkdir(parents=True)
        shutil.copy2(boot['snapshot'], target / 'memory_after.json')
        save(target / 'bootstrap_lineage.json', boot)
    return plan, design


def collect(output: Path, plan: dict, design: dict) -> None:
    source = Path(design['source'])
    if (sha(output / 'plan.json') != design['plan_sha256'] or
            sha(source / 'design.json') != design['source_design_sha256'] or
            sha(source / 'complete.json') != design['source_complete_sha256']):
        raise ValueError('Paired-Q source lineage changed')
    for name, expected in design['source_sha256'].items():
        if (sha(TTCL_ROOT / name) != expected or
                sha(output / 'source' / 'ttcl' / name) != expected):
            raise ValueError(f'Frozen paired-Q source changed: {name}')
    expected = sum(len(items) for items in design['selected_games'].values())
    client = Client(plan, design['repeat'])
    embedder = Embedder(plan['embedding'])
    completed = 0
    for family in design['families']:
        directory = (output / 'runs' / 'alfworld' / family /
                     str(design['repeat']) / 'paired_q')
        boot = design['bootstrap'][family]
        bootstrap = directory / f"episode_{boot['index']:03d}" / 'memory_after.json'
        if (sha(Path(boot['snapshot'])) != boot['sha256'] or
                sha(bootstrap) != boot['sha256']):
            raise ValueError('Paired-Q root bank changed')
        memory = PairedQMemory(plan, client, directory / 'memory',
                               plan['calibration']['alfworld'], embedder=embedder)
        memory.restore(bootstrap)
        for offset, item in enumerate(design['selected_games'][family], 1):
            index = boot['index'] + offset
            if sha(Path(plan['alf']['data_root']) / item['path']) != item['sha256']:
                raise ValueError('Official ALFWorld train game changed')
            target = directory / f'episode_{index:03d}'
            before = directory / f'episode_{index:03d}_memory_before.json'
            if (target / 'row.json').exists():
                row = read(target / 'row.json')
                if sha(target / 'memory_after.json') != row['memory_after_sha256']:
                    raise ValueError('Completed paired-Q bank changed')
                memory.restore(target / 'memory_after.json')
            elif target.exists() or before.exists():
                raise RuntimeError(f'Interrupted partial paired-Q cell needs review: {target}')
            else:
                memory.snapshot(before)
                try:
                    row = alf_cell_paired_q(plan, client, memory, item,
                                            design['repeat'], target)
                except Exception as exc:
                    save(output / 'failure.json', dict(family=family, index=index,
                        target=str(target), error=repr(exc),
                        traceback=traceback.format_exc()))
                    raise
            if (row['status'] != 'complete' or row['game'] != item['path'] or
                    row['input_sha256'] != item['sha256']):
                save(output / 'failure.json', dict(family=family, index=index,
                                                  target=str(target), row=row))
                raise RuntimeError(f'Paired-Q cell failed: {target}')
            completed += 1
            save(output / 'progress.json', dict(completed=completed,
                                                expected=expected))
            credit = (read(target / 'counterfactual_1.json')
                      if (target / 'counterfactual_1.json').exists() else None)
            print(json.dumps(dict(family=family, index=index,
                                  reward=row['reward'], attempts=row['attempts'],
                                  q_advantage=credit['q_advantage']
                                  if credit else None,
                                  extra_actor_calls=row['counterfactual_actor_calls'])),
                  flush=True)
    save(output / 'complete.json', dict(completed=completed, expected=expected,
                                        design_sha256=sha(output / 'design.json')))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    plan, design = prepare(args.source.resolve(), output, args.url)
    if not args.prepare_only:
        collect(output, plan, design)


if __name__ == '__main__':
    main()
