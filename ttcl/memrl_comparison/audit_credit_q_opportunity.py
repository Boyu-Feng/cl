"""Audit whether robust coalition credit could affect later native retrieval.

MemRL first similarity-filters and caps query keys, then uses Q to rank the
remaining memory candidates. A Q correction has no downstream retrieval
opportunity in a source chain if that memory never becomes a later candidate.
This is a necessary-opportunity check, not a counterfactual Q intervention.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import read, sha
from .audit_alf_coalition_credit import audit as audit_coalitions
from .credit_probe import memory_arms
from .simulate_staged_credit_probe import _robust_shapley_signs


def _text_sha(item: dict) -> str:
    md = item['metadata']
    text = 'Task: ' + md['task_description'] + '\nExperience: ' + md['public_abstract']
    return hashlib.sha256(text.encode()).hexdigest()


def inspect(output: Path) -> dict:
    report = audit_coalitions(output)
    if report['missing'] or report['audited'] != report['expected']:
        raise ValueError(f'Incomplete coalition probe: {output}')
    design = read(output / 'design.json')
    origin = Path(design['origin'])
    source = read(origin / 'design.json')
    groups = defaultdict(list)
    for unit in report['units']:
        groups[unit['case']].append(unit)
    cases = []
    for case, units in sorted(groups.items()):
        spec, _ = memory_arms(origin, case)
        family = case.split('/')[1]
        last = (source['bootstrap'][family]['index'] +
                len(source['selected_games'][family]))
        if spec['index'] + 1 > last:
            raise ValueError('Probe index outside source extension')
        later = []
        for index in range(spec['index'] + 2, last + 1):
            stem = (origin / 'runs' / 'alfworld' / family /
                    str(source['repeat']) / 'memrl' / f'episode_{index:03d}')
            retrieval = read(stem / 'retrieval_1.json')
            bank = read(stem.parent / f'episode_{index-1:03d}' / 'memory_after.json')
            candidates = {item['memory_id'] for item in retrieval['candidates']}
            selected = set(retrieval['ids'])
            for mid in spec['ids']:
                if mid in candidates and (mid not in bank['items'] or
                        _text_sha(bank['items'][mid]) != spec['memory_text_sha256'][mid]):
                    raise ValueError('Later candidate reused memory ID with changed text')
            later.append(dict(index=index, candidates=candidates, selected=selected))
        signs = _robust_shapley_signs(units)
        memories = []
        for i, sign in enumerate(signs):
            mid = spec['ids'][i]
            candidate_at = [row['index'] for row in later if mid in row['candidates']]
            selected_at = [row['index'] for row in later if mid in row['selected']]
            candidate_not_selected_at = [row['index'] for row in later
                                         if mid in row['candidates'] and
                                         mid not in row['selected']]
            actionable = (sign > 0 and bool(candidate_not_selected_at) or
                          sign < 0 and bool(selected_at))
            memories.append(dict(memory_index=i, memory_id=mid,
                                 memory_text_sha256=spec['memory_text_sha256'][mid],
                                 robust_shapley_sign=sign,
                                 candidate_at=candidate_at,
                                 selected_at=selected_at,
                                 candidate_not_selected_at=candidate_not_selected_at,
                                 actionable_q_opportunity=actionable))
        cases.append(dict(case=case, input_sha256=spec['source_input_sha256'],
                          later_source_tasks=len(later), memories=memories))
    robust = [memory for case in cases for memory in case['memories']
              if memory['robust_shapley_sign']]
    return dict(output=str(output), design_sha256=sha(output / 'design.json'),
                source_design_sha256=sha(origin / 'design.json'),
                source_complete_sha256=sha(origin / 'complete.json'),
                source_audit_sha256=(sha(origin / 'analysis.json')
                                     if (origin / 'analysis.json').exists() else None),
                cases=cases,
                totals=dict(cases=len(cases), robust_signs=len(robust),
                            robust_never_candidate=sum(not m['candidate_at'] for m in robust),
                            robust_candidate_and_selected=sum(bool(m['selected_at']) for m in robust),
                            robust_candidate_not_selected=sum(bool(m['candidate_not_selected_at'])
                                                              for m in robust),
                            robust_actionable_q_opportunity=sum(m['actionable_q_opportunity']
                                                                for m in robust)))


def audit(outputs: list[Path]) -> dict:
    results = [inspect(output) for output in outputs]
    return dict(schema='alf_credit_q_downstream_opportunity_v1',
                script_sha256=sha(Path(__file__)), outputs=results,
                totals={key:sum(result['totals'][key] for result in results)
                        for key in ('cases', 'robust_signs', 'robust_never_candidate',
                                    'robust_candidate_and_selected',
                                    'robust_candidate_not_selected',
                                    'robust_actionable_q_opportunity')},
                caveat='Read-only necessary-opportunity audit of actual later source retrieval; correcting Q could also alter memory evolution, prompt order and actor behavior, so these are not intervention outcomes')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, nargs='+', required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = audit([path.resolve() for path in args.outputs])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result['totals']))


if __name__ == '__main__':
    main()
