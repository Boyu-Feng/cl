"""Audit and summarize train-only ALFWorld context interventions.

Replicas are grouped by bound public game input before bootstrapping. All
interventions must have completed and passed their own frozen-source audits.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import random
import statistics

from ttcl.icl_mem0_comparison.protocol import save, sha
from .probe_full_context import audit as audit_original
from .repair_full_context_service import audit as audit_repair
from .probe_evidence_frame_control import audit as audit_frame_control
from .probe_retry_context import audit as audit_retry


def _summary(rows: list[dict], *, draws: int = 10000, seed: int = 20261003) -> dict:
    if not rows:
        raise ValueError('No paired rows')
    groups = defaultdict(list)
    for row in rows:
        key = row.get('source_input_sha256', row.get('input_sha256'))
        if not key:
            raise ValueError('Missing source input binding')
        groups[key].append(float(row['delta']))
    sizes = {len(x) for x in groups.values()}
    if sizes != {3}:
        raise ValueError(f'Expected three actor seeds per input, saw {sizes}')
    rng = random.Random(seed)
    keys = sorted(groups)
    means = [statistics.fmean(groups[k]) for k in keys]
    samples = sorted(statistics.fmean(rng.choices(means, k=len(means)))
                     for _ in range(draws))
    vals = [float(r['delta']) for r in rows]
    by_family = {}
    for family in sorted({r['task'] for r in rows}):
        family_rows = [r for r in rows if r['task'] == family]
        family_vals = [float(r['delta']) for r in family_rows]
        by_family[family] = dict(inputs=len(family_rows) // 3,
                                 pairs=len(family_rows),
                                 mean_delta=statistics.fmean(family_vals),
                                 wins=sum(v > 0 for v in family_vals),
                                 losses=sum(v < 0 for v in family_vals),
                                 ties=sum(v == 0 for v in family_vals))
    return dict(inputs=len(groups), pairs=len(rows),
                mean_delta=statistics.fmean(vals),
                cluster_bootstrap_95=[samples[int(.025 * draws)],
                                      samples[int(.975 * draws)]],
                bootstrap_draws=draws, bootstrap_seed=seed,
                wins=sum(v > 0 for v in vals),
                losses=sum(v < 0 for v in vals),
                ties=sum(v == 0 for v in vals),
                families=by_family)


def analyze(original: Path, repair: Path, frame_control: Path,
            retry: Path) -> dict:
    original_result = audit_original(original)
    repair_result = audit_repair(repair)
    frame_result = audit_frame_control(frame_control)
    retry_result = audit_retry(retry)
    if any(result['missing'] or result['completed'] != result['expected']
           for result in (original_result, repair_result,
                          frame_result, retry_result)):
        raise ValueError('A source probe is incomplete')
    repair_keys = {(r['case'], r['actor_repeat']) for r in repair_result['rows']}
    first_rows = [r for r in original_result['rows']
                  if (r['case'], r['actor_repeat']) not in repair_keys]
    if len(first_rows) + len(repair_result['rows']) != len(original_result['rows']):
        raise ValueError('Repaired pair alignment mismatch')
    fixed_set = first_rows + repair_result['rows']
    if {(r['case'], r['actor_repeat']) for r in fixed_set} != {
            (r['case'], r['actor_repeat']) for r in frame_result['rows']}:
        raise ValueError('Frame and set-level input bindings differ')
    # The retry probe binds a different, third-attempt source selection. It
    # reports input-content clusters independently.
    return dict(schema='alf_train_context_probe_summary_v1',
                source_design_sha256={
                    'original': sha(original / 'design.json'),
                    'repair': sha(repair / 'design.json'),
                    'frame_control': sha(frame_control / 'design.json'),
                    'retry': sha(retry / 'design.json')},
                full_vs_empty=_summary(fixed_set),
                framed_vs_full=_summary(frame_result['rows']),
                retry_full_vs_empty=_summary(retry_result['rows']),
                repaired_pairs=len(repair_result['rows']),
                repaired_service_drift_count=repair_result['service_drift_count'],
                frame_service_drift_count=frame_result['service_drift_count'],
                note='Official train only; fixed-snapshot same-service paired actor replays. No online memory-chain effect inferred.')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--repair', type=Path, required=True)
    p.add_argument('--frame-control', type=Path, required=True)
    p.add_argument('--retry', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = analyze(a.original, a.repair, a.frame_control, a.retry)
    save(a.output, result)
    for name in ('full_vs_empty', 'framed_vs_full', 'retry_full_vs_empty'):
        row = result[name]
        print(name, row['pairs'], row['mean_delta'],
              row['cluster_bootstrap_95'], row['wins'], row['losses'],
              row['ties'])


if __name__ == '__main__':
    main()
