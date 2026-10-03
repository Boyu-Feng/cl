"""Audit hash-ranked new-input Poker credit probes across two source chains."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from ttcl.icl_mem0_comparison.protocol import read, sha

from .analyze_cl_credit_probe import analyze as analyze_cl
from .select_credit_prefix_extension import select


def analyze(outputs: list[Path]) -> dict:
    if len(outputs) != 2:
        raise ValueError('Expected the two independently evolving source chains')
    chains = []
    all_input_hashes = []
    for output in outputs:
        design = read(output / 'design.json')
        selection_path = Path(design['selection_path'])
        frozen = read(selection_path)
        if sha(selection_path) != design['selection_sha256']:
            raise ValueError('Selection file changed')
        prior_path = Path(frozen['prior_selection'])
        if sha(prior_path) != frozen['prior_selection_sha256']:
            raise ValueError('Previous probe selection changed')
        regenerated = select(Path(frozen['origin']), prior_path,
                             frozen['repeat'], len(frozen['cases']))
        if regenerated != frozen:
            raise ValueError('Hash-ranked selection does not regenerate')
        report = analyze_cl(output)
        if report['missing'] or report['completed'] != report['expected']:
            raise ValueError('New-input replay chain is incomplete')
        two_memory_estimates = 0
        shapley_sign_reversals = 0
        shapley_interaction_nonzero = 0
        for row in report['rows']:
            if len(row['memory_ids']) != 2:
                continue
            rewards = row['rewards']
            for position in range(2):
                loo = row['delta_by_memory'][position]
                share = .5 * ((rewards[f'drop_{1 - position}'] -
                               rewards['none']) +
                              (rewards['full'] -
                               rewards[f'drop_{position}']))
                two_memory_estimates += 1
                shapley_sign_reversals += loo * share < 0
                shapley_interaction_nonzero += abs(loo - share) > 1e-9
        by_memory = defaultdict(list)
        for row in report['rows']:
            for memory_id, delta in zip(row['memory_ids'],
                                        row['delta_by_memory']):
                by_memory[row['case'], row['source_input_sha256'],
                          memory_id, row['memory_text_sha256'][memory_id]].append(delta)
        summaries = []
        for (case, input_hash, memory_id, text_hash), values in sorted(
                by_memory.items()):
            if len(values) != len(design['cl_sampling_repeats']):
                raise ValueError('Incomplete paired memory seeds')
            summaries.append(dict(case=case, input_sha256=input_hash,
                                  memory_id=memory_id,
                                  memory_text_sha256=text_hash,
                                  deltas=values,
                                  mean=statistics.fmean(values),
                                  positive=sum(value > 0 for value in values),
                                  negative=sum(value < 0 for value in values),
                                  ties=sum(value == 0 for value in values)))
        selected_inputs = [item['source_input_sha256']
                           for item in frozen['cases']]
        if len(selected_inputs) != len(set(selected_inputs)):
            raise ValueError('Repeated public input within a chain')
        all_input_hashes.append(set(selected_inputs))
        chains.append(dict(repeat=frozen['repeat'],
                           selection_sha256=design['selection_sha256'],
                           cases=len(frozen['cases']),
                           memory_examples=len(summaries),
                           seed_differences=sum(len(row['deltas'])
                                                for row in summaries),
                           repeated_negative=sum(row['negative'] >= 2 and
                                                 row['positive'] == 0
                                                 for row in summaries),
                           repeated_positive=sum(row['positive'] >= 2 and
                                                 row['negative'] == 0
                                                 for row in summaries),
                           conflicting=sum(row['positive'] > 0 and
                                           row['negative'] > 0
                                           for row in summaries),
                           all_zero=sum(row['ties'] == len(row['deltas'])
                                        for row in summaries),
                           two_memory_seed_estimates=two_memory_estimates,
                           shapley_interaction_nonzero=shapley_interaction_nonzero,
                           shapley_sign_reversals=shapley_sign_reversals,
                           examples=summaries))
    if len({chain['repeat'] for chain in chains}) != 2 or \
            all_input_hashes[0] != all_input_hashes[1]:
        raise ValueError('Two source chains do not cover the same new inputs')
    return dict(schema='credit_prefix_extension_audit_v1',
                distinct_public_inputs=len(all_input_hashes[0]),
                chains=chains,
                caveat='Official CLBench calibration prefix, fixed-snapshot replay; repeated inputs across 303/404 must be grouped together and are not independent test cases')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', nargs=2, type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = analyze([path.resolve() for path in args.outputs])
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key != 'chains'}, sort_keys=True))
    for chain in result['chains']:
        print({key: value for key, value in chain.items() if key != 'examples'})


if __name__ == '__main__':
    main()
