"""Combine audited ALFWorld per-memory probes without treating them as test data."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import read, save, sha
from ttcl.memrl_comparison.analyze_credit_probe import analyze


def summarize(inputs: list[Path]) -> dict:
    if len(inputs) < 2:
        raise ValueError('Need separate frozen memory-index probes')
    first = read(inputs[0] / 'design.json')
    origin_hash = first['origin_plan_sha256']
    actor_repeats = first['alf_actor_repeats']
    all_rows = []
    indices = set()
    case_sources = {}
    for root in inputs:
        design = read(root / 'design.json')
        index = design['paired_drop_index']
        if index in indices:
            raise ValueError(f'Repeated memory index {index}')
        indices.add(index)
        if design['origin_plan_sha256'] != origin_hash or design['alf_actor_repeats'] != actor_repeats:
            raise ValueError('Probes do not share a frozen source plan and actor seeds')
        report = analyze(root)
        if report['completed'] != report['expected'] or report['missing']:
            raise ValueError(f'Incomplete probe: {root}')
        for row in report['rows']:
            parts = Path(row['case']).parts
            source = read(root / 'alfworld' / parts[1] / parts[2] / parts[4] / 'source.json')
            if row['case'] in case_sources and case_sources[row['case']] != source:
                raise ValueError(f'Case source changed across probes: {row["case"]}')
            case_sources[row['case']] = source
            if source['ids'][index] != row['memory_id']:
                raise ValueError(f'Memory position changed: {row["case"]}')
            all_rows.append(dict(row, memory_index=index))
    by_memory = defaultdict(list)
    for row in all_rows:
        by_memory[(row['case'], row['memory_index'])].append(row)
    memories = []
    for (case, index), rows in sorted(by_memory.items()):
        if {row['actor_repeat'] for row in rows} != set(actor_repeats):
            raise ValueError(f'Missing paired seed: {case}, {index}')
        source = case_sources[case]
        mid = source['ids'][index]
        features = source['memory_features'][mid]
        deltas = [row['delta'] for row in rows]
        memories.append(dict(case=case, index=index, memory_id=mid,
                             prior_success=features['metadata']['success'],
                             prior_q=features['metadata']['q_value'],
                             retrieval_similarity=features['retrieval']['similarity'],
                             text_characters=features['text_characters'],
                             mean_delta=sum(deltas)/len(deltas),
                             positive=sum(x > 0 for x in deltas),
                             negative=sum(x < 0 for x in deltas),
                             ties=sum(x == 0 for x in deltas)))
    deltas = [row['delta'] for row in all_rows]
    return dict(origin_plan_sha256=origin_hash, actor_repeats=actor_repeats,
                input_design_sha256=[sha(root / 'design.json') for root in inputs],
                cases=len(case_sources), memories=len(memories), pairs=len(all_rows),
                positive=sum(x > 0 for x in deltas), negative=sum(x < 0 for x in deltas),
                ties=sum(x == 0 for x in deltas),
                mean_delta=sum(deltas)/len(deltas), by_memory=memories,
                caveat='Outcome-stratified valid_unseen development snapshots; these rewards cannot train a policy and then serve as untouched valid_unseen test evidence.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('inputs', type=Path, nargs='+')
    args = parser.parse_args()
    report = summarize([p.resolve() for p in args.inputs])
    save(args.output, report)
    print(f"Audited {report['pairs']} pairs across {report['memories']} memories; "
          f"positive={report['positive']} negative={report['negative']} ties={report['ties']}")


if __name__ == '__main__':
    main()
