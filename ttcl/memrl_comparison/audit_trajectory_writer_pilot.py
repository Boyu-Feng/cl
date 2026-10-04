"""Audit frozen writer outputs without relaxing evidence-grounding rules."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.experience_evolution.core import digest
from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .probe_trajectory_writer import prompt_for
from .trajectory_experience_teacher import claims_from_trajectory


def parse_raw(raw: str, expected: list[dict]) -> dict:
    try:
        start = min((i for i, char in enumerate(raw) if char in '{['), default=-1)
        if start < 0:
            raise ValueError('No JSON')
        value, _ = json.JSONDecoder().raw_decode(raw[start:])
        format_ok = isinstance(value, dict) and isinstance(value.get('claims'), list)
        claims = value['claims'] if format_ok else value if isinstance(value, list) else []
        if not isinstance(claims, list):
            raise ValueError('Claims are not an array')
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return dict(format_ok=False, accepted=[], rejected=[dict(reason=str(exc))],
                    expected_count=len(expected), recall=0.0)
    expected_keys = {(claim['kind'], tuple(claim['evidence_indices']),
                      claim.get('operation'), claim.get('tool_type')) for claim in expected}
    accepted, rejected = set(), []
    for claim in claims:
        if not isinstance(claim, dict):
            rejected.append(dict(claim=claim, reason='Claim must be an object'))
            continue
        indices = claim.get('evidence_indices')
        if not isinstance(indices, list) or any(not isinstance(i, int) for i in indices):
            rejected.append(dict(claim=claim, reason='Invalid evidence indices'))
            continue
        key = (claim.get('kind'), tuple(indices), claim.get('operation'), claim.get('tool_type'))
        if key not in expected_keys:
            rejected.append(dict(claim=claim, reason='No exact source-supported claim'))
            continue
        accepted.add(key)
    return dict(format_ok=format_ok,
                accepted=[list(key) for key in sorted(accepted, key=str)], rejected=rejected,
                expected_count=len(expected),
                recall=len(accepted) / len(expected_keys) if expected_keys else 1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    design_path = root / 'design.json'
    design = read(design_path)
    if (design['schema'] != 'alf_trajectory_writer_pilot_v1'
            or sha(Path(design['corpus_path'])) != design['corpus_sha256']):
        raise ValueError('Frozen writer pilot changed')
    rows = []
    for index, case in enumerate(design['cases']):
        episode_path = Path(case['source_episode'])
        if sha(episode_path) != case['source_episode_sha256']:
            raise ValueError('Source episode changed')
        episode = read(episode_path)
        if episode['game'] != case['game'] or claims_from_trajectory(episode['trajectory']) != case['expected_claims']:
            raise ValueError('Source evidence changed')
        response_path = root / f'{index:02d}.json'
        response = read(response_path)
        if (response['index'] != index or response['source_episode_sha256'] != case['source_episode_sha256']
                or response['input_sha256'] != case['input_sha256']
                or response['prompt_sha256'] != digest(json.dumps(prompt_for(episode), ensure_ascii=False))
                or response['raw_response_sha256'] != digest(response['raw_response'])):
            raise ValueError('Writer response binding changed')
        rows.append(dict(index=index, game=case['game'], input_sha256=case['input_sha256'],
            response_sha256=sha(response_path),
            judgment=parse_raw(response['raw_response'], case['expected_claims'])))
    report = dict(schema='alf_trajectory_writer_pilot_audit_v1',
        design_sha256=sha(design_path), auditor_sha256=sha(Path(__file__)), rows=rows,
        totals=dict(cases=len(rows), format_ok=sum(row['judgment']['format_ok'] for row in rows),
            accepted_claims=sum(len(row['judgment']['accepted']) for row in rows),
            rejected_claims=sum(len(row['judgment']['rejected']) for row in rows),
            expected_claims=sum(row['judgment']['expected_count'] for row in rows)),
        caveat='Stratified diagnostic on six known train trajectories, not a generalization estimate or trained writer result')
    save(args.output, report)
    print(json.dumps(report['totals']), flush=True)


if __name__ == '__main__':
    main()
