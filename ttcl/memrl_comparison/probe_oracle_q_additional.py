"""Two frozen ALFWorld train-task checks after oracle Q changes retrieval membership.

The target cases are selected from a read-only retrieval audit by a fixed
predicate before any actor replay. Earlier-task Shapley values are posthoc
oracles and may not be useful on the later task. No online memory update runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import Client, read, save, sha
from .credit_probe import memory_arms, run_alf
from .memory import Embedder, Memory


REPEATS = (93911, 93912, 93913)
ARMS = ('native', 'oracle_q', 'native_repeat')


def design_for(audit_path: Path, output: Path, url: str) -> dict:
    audit = read(audit_path)
    if audit['schema'] != 'alf_oracle_q_retrieval_audit_v1':
        raise ValueError('Wrong upstream audit')
    selected = [row for row in audit['rows']
                if row['membership_changed'] and
                'pick_heat_then_place_in_recep' not in row['prior_case']]
    if len(selected) != 2:
        raise ValueError('Expected exactly two as-yet untested membership changes')
    embedder = None
    cases = []
    for case_id, row in enumerate(selected):
        origin = Path(read(Path(row['source_output']) / 'design.json')['origin'])
        plan = read(origin / 'plan.json')
        plan['url'] = url
        plan['alf']['actor_url'] = url
        client = Client(plan, 92721)
        if embedder is None:
            embedder = Embedder(plan['embedding'])
        family = row['prior_case'].split('/')[1]
        later_case = (f'alfworld/{family}/92721/memrl/'
                      f"episode_{row['later_index']:03d}")
        spec, _ = memory_arms(origin, later_case)
        if (spec['source_input_sha256'] != row['later_input_sha256'] or
                sha(origin / 'runs' / later_case / 'retrieval_1.json') !=
                row['source_retrieval_sha256']):
            raise ValueError('Later source input changed')
        snapshot = (origin / 'runs' / 'alfworld' / family / '92721' / 'memrl' /
                    f"episode_{row['later_index']-1:03d}" / 'memory_after.json')
        if sha(snapshot) != row['later_snapshot_sha256']:
            raise ValueError('Later snapshot changed')
        memory = Memory(plan, client, output / f'retrieval_{case_id}',
                        plan['calibration']['alfworld'], embedder=embedder)
        memory.restore(snapshot)
        query = read(origin / 'runs' / later_case / 'retrieval_1.json')['query']
        native = memory.retrieve(query)
        memory.service._q_cache[row['memory_id']] = row['oracle_q']
        oracle = memory.retrieve(query)
        if (native['ids'] != row['native_ids'] or
                oracle['ids'] != row['oracle_ids'] or
                native['context_sha256'] != row['native_context_sha256'] or
                oracle['context_sha256'] != row['oracle_context_sha256']):
            raise ValueError('Frozen retrieval audit did not reproduce')
        cases.append(dict(case_id=case_id, origin=str(origin),
                          source_plan_sha256=sha(origin / 'plan.json'),
                          prior_case=row['prior_case'], later_case=later_case,
                          later_input_sha256=row['later_input_sha256'],
                          row_sha256=sha(origin / 'runs' / later_case / 'row.json'),
                          memory_id=row['memory_id'],
                          memory_text_sha256=row['memory_text_sha256'],
                          stable_shapley=row['stable_shapley'],
                          native_q=row['native_q'], oracle_q=row['oracle_q'],
                          native_ids=native['ids'], oracle_ids=oracle['ids'],
                          native_context=native['context'],
                          oracle_context=oracle['context'],
                          native_context_sha256=native['context_sha256'],
                          oracle_context_sha256=oracle['context_sha256']))
    return dict(schema='alf_oracle_q_additional_v1',
                audit_path=str(audit_path), audit_sha256=sha(audit_path),
                runner_sha256=sha(Path(__file__)),
                credit_probe_sha256=sha(Path(__file__).with_name('credit_probe.py')),
                memory_adapter_sha256=sha(Path(__file__).with_name('memory.py')),
                cases=cases, repeats=list(REPEATS), url=url,
                budget='Two official train games, three new seeds, native/oracle_q/native_repeat; at most 50 environment actions per arm',
                caveat='Posthoc earlier-task local credit directly overwrites one Q in each later fixed snapshot; not a prospective learned or online policy')


def run(audit_path: Path, output: Path, url: str, prepare_only: bool) -> None:
    design = design_for(audit_path, output, url)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'design.json'
    if path.exists():
        if read(path) != design:
            raise ValueError('Frozen design changed')
    else:
        save(path, design)
    print(json.dumps(dict(design_sha256=sha(path),
                          cases=[dict(case_id=c['case_id'],
                                      native=c['native_ids'],
                                      oracle=c['oracle_ids'])
                                 for c in design['cases']])), flush=True)
    if prepare_only:
        return
    for case in design['cases']:
        origin = Path(case['origin'])
        plan = read(origin / 'plan.json')
        plan['url'] = url
        plan['alf']['actor_url'] = url
        spec, _ = memory_arms(origin, case['later_case'])
        contexts = dict(native=case['native_context'],
                        oracle_q=case['oracle_context'],
                        native_repeat=case['native_context'])
        for repeat in REPEATS:
            target = output / f"case_{case['case_id']}" / f'actor_repeat_{repeat}'
            target.mkdir(parents=True, exist_ok=True)
            summary = target / 'summary.json'
            if summary.exists():
                continue
            result = run_alf(plan, Client(plan, repeat),
                             dict(spec, repeat=repeat), contexts, target)
            save(summary, dict(case_id=case['case_id'], repeat=repeat,
                               design_sha256=sha(path), replay=result))
            print(json.dumps(dict(case_id=case['case_id'], repeat=repeat,
                                  rewards={name:arm['reward'] for name,arm in result.items()})),
                  flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18559')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run(args.audit.resolve(), args.output.resolve(), args.url,
        args.prepare_only)


if __name__ == '__main__':
    main()
