"""Conservative, content-bound transfer of observed ALFWorld memory harm.

This is a diagnostic policy: it reuses a negative leave-one-out estimate only
for the identical experience text within the same task family, and removes at
most one retrieved entry.  Its new-game value must be tested separately.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from ttcl.icl_mem0_comparison.protocol import read, save, sha

from .analyze_credit_train_extension_probes import analyze


ACTOR_SEEDS = (92831, 92832, 92833)


def fit(outputs: list[Path]) -> dict:
    if len(outputs) != 3:
        raise ValueError('All three deletion positions are required')
    reports = [analyze(path) for path in outputs]
    if (any(report['completed'] != report['expected'] or report['missing']
            for report in reports) or
            len({report['selection_sha256'] for report in reports}) != 1 or
            {read(path / 'design.json')['paired_drop_index'] for path in outputs}
            != {0, 1, 2}):
        raise ValueError('Incomplete or inconsistent donor probes')
    grouped = defaultdict(list)
    for report in reports:
        for row in report['rows']:
            key = (row['task'], row['memory_text_sha256'])
            grouped[key].append(row)
    examples, entries = [], []
    for (family, text_hash), rows in sorted(grouped.items()):
        if (len(rows) != 3 or
                {row['actor_repeat'] for row in rows} != set(ACTOR_SEEDS) or
                len({row['case'] for row in rows}) != 1 or
                len({row['source_input_sha256'] for row in rows}) != 1):
            raise ValueError('Donor text has ambiguous or incomplete source')
        rows.sort(key=lambda row: row['actor_repeat'])
        deltas = [row['delta'] for row in rows]
        negative = sum(delta < 0 for delta in deltas)
        positive = sum(delta > 0 for delta in deltas)
        example = dict(family=family, memory_text_sha256=text_hash,
                       source_case=rows[0]['case'],
                       source_input_sha256=rows[0]['source_input_sha256'],
                       actor_seeds=list(ACTOR_SEEDS), seed_deltas=deltas,
                       mean_delta=statistics.fmean(deltas),
                       negative=negative, positive=positive)
        examples.append(example)
        if negative >= 2 and positive == 0:
            entries.append(example)
    return dict(schema='causal_credit_cache_v1',
                donor_selection_sha256=reports[0]['selection_sha256'],
                donor_design_sha256=[report['design_sha256'] for report in reports],
                donor_analysis_sha256=[sha(path / 'analysis.json') for path in outputs],
                source_code_sha256=sha(Path(__file__)),
                rule='Same task family and exact experience-text SHA-256; three paired seeds, at least two negative and no positive; suppress at most one entry with smallest mean delta.',
                examples=examples, entries=entries)


def decision(spec: dict, policy: dict) -> dict:
    if policy['schema'] != 'causal_credit_cache_v1':
        raise ValueError('Unknown causal-credit cache')
    matching = [dict(position=position, memory_id=memory_id, donor=entry)
                for position, memory_id in enumerate(spec['ids'])
                for entry in policy['entries']
                if (entry['family'] == spec['task'] and
                    entry['memory_text_sha256'] ==
                    spec['memory_text_sha256'][memory_id])]
    if not matching:
        return dict(drop_position=None, drop_memory_id=None, donor=None)
    selected = min(matching, key=lambda item: (item['donor']['mean_delta'],
                                               item['position']))
    return dict(drop_position=selected['position'],
                drop_memory_id=selected['memory_id'], donor=selected['donor'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', nargs=3, type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    args = parser.parse_args()
    policy = fit([path.resolve() for path in args.outputs])
    if args.policy.exists() and read(args.policy) != policy:
        raise ValueError('Frozen causal-credit cache changed')
    save(args.policy, policy)
    print(json.dumps(dict(examples=len(policy['examples']),
                          entries=len(policy['entries']),
                          families=[entry['family'] for entry in policy['entries']]),
                     sort_keys=True))


if __name__ == '__main__':
    main()
