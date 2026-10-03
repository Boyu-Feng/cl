"""Run the frozen source-success retrieval filter through an evolving train chain."""
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
from .source_success_policy import SourceSuccessMemory
from .worker import alf_cell


FILES = (*SOURCE_FILES,
         'memrl_comparison/audit_credit_train_extension.py',
         'memrl_comparison/source_success_policy.py',
         'memrl_comparison/collect_source_success_online.py',
         'memrl_comparison/audit_source_success_online.py')


def prepare(source: Path, validation: Path, output: Path,
            url: str) -> tuple[dict, dict]:
    if output.exists():
        plan, design = read(output / 'plan.json'), read(output / 'design.json')
        if (design['source'] != str(source) or
                design['validation'] != str(validation) or design['url'] != url):
            raise ValueError('Existing online success-filter design differs')
        return plan, design
    report = audit_source(source)
    if (not report['complete'] or report['missing'] or
            report['audited_pairs'] != report['expected_pairs']):
        raise ValueError('Native source is incomplete')
    validation_design = read(validation / 'design.json')
    validation_report = read(validation / 'analysis.json')
    if (validation_design['schema'] != 'alf_source_success_filter_validation_v1' or
            validation_design['origin'] != str(source) or
            validation_report['schema'] != 'alf_source_success_filter_audit_v1' or
            validation_report['audited'] != validation_report['expected'] or
            validation_report['missing']):
        raise ValueError('Frozen fixed-snapshot rule validation is incomplete')
    source_plan, source_design = read(source / 'plan.json'), read(source / 'design.json')
    plan = copy.deepcopy(source_plan)
    plan['url'] = url
    plan['alf']['actor_url'] = url
    plan['source_success_online'] = dict(
        source=str(source), source_design_sha256=sha(source / 'design.json'),
        source_complete_sha256=sha(source / 'complete.json'),
        validation_design_sha256=sha(validation / 'design.json'),
        validation_analysis_sha256=sha(validation / 'analysis.json'),
        policy='On each game first retrieval only, if exactly 3 memories and 1-2 source successes, keep source-success entries in original order')
    output.mkdir(parents=True)
    save(output / 'plan.json', plan)
    bootstrap = {}
    for family in source_design['families']:
        boot = source_design['bootstrap'][family]
        snapshot = (source / 'runs' / 'alfworld' / family /
                    str(source_design['repeat']) / 'memrl' /
                    f"episode_{boot['index']:03d}" / 'memory_after.json')
        if sha(snapshot) != boot['sha256']:
            raise ValueError('Native root bank changed')
        bootstrap[family] = dict(index=boot['index'], snapshot=str(snapshot),
                                 sha256=sha(snapshot))
    design = dict(schema='alf_source_success_online_v1',
                  plan_sha256=sha(output / 'plan.json'), source=str(source),
                  source_design_sha256=sha(source / 'design.json'),
                  source_complete_sha256=sha(source / 'complete.json'),
                  validation=str(validation),
                  validation_design_sha256=sha(validation / 'design.json'),
                  validation_analysis_sha256=sha(validation / 'analysis.json'),
                  url=url, repeat=source_design['repeat'],
                  families=source_design['families'],
                  selected_games=source_design['selected_games'],
                  bootstrap=bootstrap,
                  source_sha256={name:sha(TTCL_ROOT / name) for name in FILES},
                  note='Development online train continuation from identical episode-6 root; each arm evolves writer and Q independently')
    save(output / 'design.json', design)
    for name in FILES:
        target = output / 'source' / 'ttcl' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TTCL_ROOT / name, target)
    for family, boot in bootstrap.items():
        target = (output / 'runs' / 'alfworld' / family /
                  str(design['repeat']) / 'success_filter' /
                  f"episode_{boot['index']:03d}")
        target.mkdir(parents=True)
        shutil.copy2(boot['snapshot'], target / 'memory_after.json')
        save(target / 'bootstrap_lineage.json', boot)
    return plan, design


def collect(output: Path, plan: dict, design: dict) -> None:
    source, validation = Path(design['source']), Path(design['validation'])
    if (sha(output / 'plan.json') != design['plan_sha256'] or
            sha(source / 'design.json') != design['source_design_sha256'] or
            sha(source / 'complete.json') != design['source_complete_sha256'] or
            sha(validation / 'design.json') != design['validation_design_sha256'] or
            sha(validation / 'analysis.json') != design['validation_analysis_sha256']):
        raise ValueError('Frozen online rule lineage changed')
    for name, expected in design['source_sha256'].items():
        if (sha(TTCL_ROOT / name) != expected or
                sha(output / 'source' / 'ttcl' / name) != expected):
            raise ValueError(f'Frozen online source changed: {name}')
    client = Client(plan, design['repeat'])
    embedder = Embedder(plan['embedding'])
    expected = sum(len(items) for items in design['selected_games'].values())
    completed = 0
    for family in design['families']:
        directory = (output / 'runs' / 'alfworld' / family /
                     str(design['repeat']) / 'success_filter')
        boot = design['bootstrap'][family]
        bootstrap = directory / f"episode_{boot['index']:03d}" / 'memory_after.json'
        if (sha(Path(boot['snapshot'])) != boot['sha256'] or
                sha(bootstrap) != boot['sha256']):
            raise ValueError('Online root bank changed')
        memory = SourceSuccessMemory(plan, client, directory / 'memory',
                                     plan['calibration']['alfworld'], embedder=embedder)
        memory.restore(bootstrap)
        for offset, item in enumerate(design['selected_games'][family], 1):
            index = boot['index'] + offset
            if sha(Path(plan['alf']['data_root']) / item['path']) != item['sha256']:
                raise ValueError('Official train game changed')
            target = directory / f'episode_{index:03d}'
            before = directory / f'episode_{index:03d}_memory_before.json'
            if (target / 'row.json').exists():
                row = read(target / 'row.json')
                if sha(target / 'memory_after.json') != row['memory_after_sha256']:
                    raise ValueError('Completed online memory bank changed')
                memory.restore(target / 'memory_after.json')
            elif target.exists() or before.exists():
                raise RuntimeError(f'Interrupted online cell needs review: {target}')
            else:
                memory.snapshot(before)
                memory.begin_episode()
                try:
                    row = alf_cell(plan, client, memory, item, design['repeat'],
                                   'success_filter', target)
                except Exception as exc:
                    save(output / 'failure.json', dict(family=family, index=index,
                         target=str(target), error=repr(exc),
                         traceback=traceback.format_exc()))
                    raise
            if (row['status'] != 'complete' or row['game'] != item['path'] or
                    row['input_sha256'] != item['sha256']):
                save(output / 'failure.json', dict(family=family, index=index,
                                                  target=str(target), row=row))
                raise RuntimeError('Online source-success cell failed')
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
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18559')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    plan, design = prepare(args.source.resolve(), args.validation.resolve(),
                           args.output.resolve(), args.url)
    if not args.prepare_only:
        collect(args.output.resolve(), plan, design)


if __name__ == '__main__':
    main()
