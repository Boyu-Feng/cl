"""Check whether source-discordant sampling yields more ALFWorld credit labels.

The enriched set is development-only: it was selected using native-vs-none
source rewards. These diagnostics cannot estimate unbiased benchmark benefit.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import read, sha
from .audit_alf_coalition_credit import audit as audit_coalitions
from .simulate_staged_credit_probe import _robust_shapley_signs


ARMS = ('none', 'only_0', 'only_1', 'only_2',
        'drop_0', 'drop_1', 'drop_2')


def summarize(output: Path) -> dict:
    report = audit_coalitions(output)
    if report['missing'] or report['audited'] != report['expected']:
        raise ValueError(f'Incomplete coalition audit: {output}')
    design = read(output / 'design.json')
    origin = Path(design['origin'])
    grouped = defaultdict(list)
    for unit in report['units']:
        grouped[unit['case']].append(unit)
    cases = []
    for case, units in sorted(grouped.items()):
        units.sort(key=lambda u: u['repeat'])
        source_native = read(origin / 'runs' / case / 'row.json')
        source_none = read(origin / 'runs' / case.replace('/memrl/', '/none/') / 'row.json')
        stable = [u for u in units if u['full_repeat_reward_equal']]
        if len(units) != 3 or len(stable) < 2:
            raise ValueError(f'Insufficient stable actor repeats: {case}')
        nonzero = sum(u['rewards'][arm] != u['rewards']['full']
                      for u in stable for arm in ARMS)
        signs = _robust_shapley_signs(units)
        cases.append(dict(case=case, input_sha256=next(
            item['input_sha256'] for item in design['cases'] if item['case'] == case),
            source_first_discordant=(source_native['first_attempt'] !=
                                     source_none['first_attempt']),
            source_within_three_discordant=(source_native['within_three'] !=
                                            source_none['within_three']),
            stable_repeats=[u['repeat'] for u in stable],
            nonzero_subset_comparisons=nonzero,
            informative=nonzero > 0,
            robust_shapley_signs=signs,
            robust_signs=sum(sign != 0 for sign in signs),
            actor_calls=sum(u['actor_calls'] for u in units)))
    return dict(output=str(output),
                design_sha256=sha(output / 'design.json'),
                analysis_sha256=sha(output / 'analysis.json'),
                cases=cases,
                totals=dict(cases=len(cases),
                            informative=sum(c['informative'] for c in cases),
                            nonzero_subset_comparisons=sum(c['nonzero_subset_comparisons']
                                                           for c in cases),
                            robust_signs=sum(c['robust_signs'] for c in cases),
                            actor_calls=sum(c['actor_calls'] for c in cases)))


def compare(blind: Path, enriched: Path) -> dict:
    original, selected = summarize(blind), summarize(enriched)
    if ({c['input_sha256'] for c in original['cases']} &
            {c['input_sha256'] for c in selected['cases']}):
        raise ValueError('Blind and enriched public inputs overlap')
    return dict(schema='alf_source_discordant_credit_enrichment_v1',
                analysis_script_sha256=sha(Path(__file__)),
                blind=original, enriched=selected,
                caveat='Outcome-informed source selection for development; fixed-snapshot subset labels and actor calls are not online policy gains or unbiased benchmark estimates')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blind', type=Path, required=True)
    parser.add_argument('--enriched', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.blind.resolve(), args.enriched.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({name: result[name]['totals'] for name in ('blind', 'enriched')}))


if __name__ == '__main__':
    main()
