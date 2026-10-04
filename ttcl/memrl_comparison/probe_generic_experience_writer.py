"""Cross-domain frozen-model experience-writer diagnostic without action rules.

The model proposes conditional advice from bounded public trajectory windows.
An independent model prompt checks semantic support; exact source indices and
input-content hashes are checked mechanically. Neither step hard-codes action
names, domain feedback text, or downstream labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import requests

from ttcl.experience_evolution.core import digest
from ttcl.icl_mem0_comparison.protocol import read, save, sha


DOMAINS = ('blind_spectrum_monitoring', 'cohort_studies',
           'database_exploration', 'exploitable_poker')
CLAIM_SCHEMA = {'type': 'object', 'properties': {
    'condition': {'type': 'string'}, 'advice': {'type': 'string'},
    'evidence_steps': {'type': 'array', 'items': {'type': 'integer'}},
    'uncertainty': {'type': 'string'}},
    'required': ['condition', 'advice', 'evidence_steps', 'uncertainty'],
    'additionalProperties': False}
AUDIT_SCHEMA = {'type': 'object', 'properties': {
    'supported': {'type': 'boolean'}, 'reason': {'type': 'string'},
    'narrow_condition': {'type': 'string'}, 'narrow_advice': {'type': 'string'}},
    'required': ['supported', 'reason', 'narrow_condition', 'narrow_advice'],
    'additionalProperties': False}


def bound(text: str, limit: int = 1000) -> str:
    value = str(text)
    if len(value) <= limit:
        return value
    half = limit // 2
    return value[:half] + '\n[...content omitted...]\n' + value[-half:]


def windows(rows: list[dict], size: int = 8) -> list[list[dict]]:
    if not rows:
        return []
    first = rows[:size]
    last = rows[-size:]
    return [first] if first == last else [first, last]


def alf_cases(root: Path) -> list[dict]:
    design = read(root / 'design.json')
    cases = []
    for family in sorted(design['selected_games']):
        options = []
        for offset, game in enumerate(design['selected_games'][family], 1):
            index = design['bootstrap'][family]['index'] + offset
            path = root / 'runs' / 'alfworld' / family / str(design['repeat']) / 'memrl' / f'episode_{index:03d}' / 'attempt_1' / 'episode.json'
            if not path.is_file():
                continue
            episode = read(path)
            if not episode['game'].startswith('json_2.1.1/train/'):
                raise ValueError('ALF source is not official train')
            rows = [dict(step=i, action=row['action'], feedback=bound(row['observation'], 700))
                    for i, row in enumerate(episode['trajectory'])]
            options.append(dict(source_path=str(path.resolve()), source_sha256=sha(path),
                input_sha256=game['sha256'], task=bound(episode['initial_observation'], 700),
                outcome=episode['reward'], rows=rows, family=family))
        if options:
            cases.append(min(options, key=lambda row: row['input_sha256']))
    return cases


def cl_cases(root: Path) -> list[dict]:
    plan = read(root / 'plan.json')
    cases = []
    for domain in DOMAINS:
        prefix = int(plan['tasks'][domain] * .2)
        candidates = []
        for index in range(prefix):
            path = (root / 'runs' / 'clbench' / domain / '303' / 'memrl' /
                    f'episode_{index+1:03d}')
            row_path, trace_path = path / 'row.json', path / 'public_trajectory.json'
            if not row_path.is_file() or not trace_path.is_file():
                continue
            record, trace = read(row_path), read(trace_path)
            if record['status'] != 'complete' or not trace:
                continue
            rows = [dict(step=i, query=bound(step['query'], 700),
                action=bound(json.dumps(step['action'], ensure_ascii=False), 700),
                feedback=bound(step.get('public_feedback', ''), 700))
                for i, step in enumerate(trace)]
            candidates.append(dict(source_path=str(trace_path.resolve()),
                source_sha256=sha(trace_path), row_path=str(row_path.resolve()),
                row_sha256=sha(row_path), input_sha256=record['initial_query_sha256'],
                task=bound(trace[0]['query'], 700), outcome=record['reward'],
                rows=rows, domain=domain, canonical_index=index))
        candidates.sort(key=lambda row: row['input_sha256'])
        cases.extend(candidates[:2])
    return cases


def messages(case: dict, window: list[dict]) -> list[dict]:
    instruction = ('Learn at most one reusable conditional experience from this '
        'completed interaction window. Identify a situation and an action or '
        'decision that the observed feedback actually supports. Infer the '
        'experience yourself; no action types or domain rules are supplied. '
        'Cite exact step IDs. Do not turn one observation into a universal '
        'requirement. If no useful evidence exists, use empty condition and '
        'advice and an empty evidence_steps list. Keep each text field under '
        '30 words. Return JSON only.')
    content = dict(task=case['task'], outcome=case['outcome'], window=window)
    return [dict(role='system', content=instruction),
            dict(role='user', content=json.dumps(content, ensure_ascii=False))]


def audit_messages(proposal: dict, cited: list[dict]) -> list[dict]:
    instruction = ('Audit the proposed general experience against only the '
        'cited public trajectory rows. Reject claims that need an unobserved '
        'cause or contradict feedback. If the wording is too broad, rewrite '
        'a narrow conditional suggestion. A source task reward does not '
        'prove that every action was good. Keep each text field under '
        '30 words. Return JSON only.')
    return [dict(role='system', content=instruction),
            dict(role='user', content=json.dumps(dict(proposal=proposal,
                cited_evidence=cited), ensure_ascii=False))]


def complete(url: str, messages: list[dict], schema: dict, name: str, seed: int) -> dict:
    payload = dict(model='frozen-actor', messages=messages, temperature=0,
        max_tokens=512, seed=seed,
        response_format={'type':'json_schema', 'json_schema':
            {'name':name, 'schema':schema, 'strict':True}})
    response = requests.post(url.rstrip('/') + '/v1/chat/completions',
        json=payload, timeout=240)
    response.raise_for_status()
    choice = response.json()['choices'][0]
    raw = choice['message']['content']
    try:
        parsed, parse_error = json.loads(raw), None
    except json.JSONDecodeError as exc:
        parsed, parse_error = None, str(exc)
    return dict(raw=raw, parsed=parsed, parse_error=parse_error,
                finish_reason=choice['finish_reason'],
                prompt_sha256=digest(json.dumps(messages, ensure_ascii=False)),
                usage=response.json().get('usage', {}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alf-origin', type=Path, required=True)
    parser.add_argument('--cl-origin', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18559')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    alf_root, cl_root, output = (args.alf_origin.resolve(),
        args.cl_origin.resolve(), args.output.resolve())
    cases = alf_cases(alf_root) + cl_cases(cl_root)
    if len(cases) != 14 or any(not c['rows'] for c in cases):
        raise ValueError('Need six ALF train families and two prefix cases per CL domain')
    rows = []
    for case_index, case in enumerate(cases):
        for window_index, window in enumerate(windows(case['rows'])):
            rows.append(dict(case_index=case_index, window_index=window_index,
                step_ids=[x['step'] for x in window]))
    design = dict(schema='cross_domain_frozen_writer_diagnostic_v1',
        alf_origin=str(alf_root), alf_design_sha256=sha(alf_root / 'design.json'),
        cl_origin=str(cl_root), cl_plan_sha256=sha(cl_root / 'plan.json'),
        cases=cases, windows=rows, writer_schema=CLAIM_SCHEMA,
        auditor_schema=AUDIT_SCHEMA, runner_sha256=sha(Path(__file__)),
        url=args.url,
        selection='One minimum-input-SHA existing official train source per ALF family; two minimum-input-SHA completed CL calibration-prefix sources per domain; first/last eight-step windows, no outcome-based selection',
        caveat='Model-proposal/critique diagnostic, not trained writer or benchmark policy')
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / 'design.json'
    if frozen.exists():
        if read(frozen) != design:
            raise ValueError('Frozen writer pilot changed')
    else:
        save(frozen, design)
    print(json.dumps(dict(design_sha256=sha(frozen), cases=len(cases),
                          windows=len(rows))), flush=True)
    if args.prepare_only:
        return
    for index, item in enumerate(rows):
        path = output / f'{index:02d}.json'
        if path.exists():
            continue
        case = cases[item['case_index']]
        window = [case['rows'][i] for i in item['step_ids']]
        proposal_messages = messages(case, window)
        proposal = complete(args.url, proposal_messages, CLAIM_SCHEMA,
            'experience_proposal', 95300 + index)
        candidate = proposal['parsed'] if isinstance(proposal['parsed'], dict) else {}
        ids = candidate.get('evidence_steps', [])
        valid_ids = (isinstance(ids, list) and
            all(type(i) is int and i in item['step_ids'] for i in ids) and
            len(set(ids)) == len(ids))
        critic = None
        if (valid_ids and ids and candidate.get('condition') and candidate.get('advice')
                and proposal['finish_reason'] == 'stop'):
            critic = complete(args.url, audit_messages(candidate,
                [case['rows'][i] for i in ids]), AUDIT_SCHEMA,
                'experience_audit', 95400 + index)
        result = dict(index=index, case_index=item['case_index'],
            window_index=item['window_index'], source_sha256=case['source_sha256'],
            input_sha256=case['input_sha256'], window_step_ids=item['step_ids'],
            proposal=proposal, cited_indices_valid=valid_ids, critic=critic)
        save(path, result)
        print(json.dumps(dict(index=index, family=case.get('family',case.get('domain')),
            valid_ids=valid_ids, proposed=bool(ids),
            critic_supported=critic['parsed'].get('supported')
                if critic and isinstance(critic['parsed'], dict) else None,
            proposal_parse_error=proposal['parse_error'])), flush=True)


if __name__ == '__main__':
    main()
