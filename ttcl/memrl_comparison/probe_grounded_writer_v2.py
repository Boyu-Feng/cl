"""Domain-neutral trajectory writer pilot with literal feedback grounding.

This is a source-quality diagnostic, not a benchmark evaluation. It deliberately
does not define action names, feedback classes, or a success heuristic. Sources
are frozen existing public trajectories from ALF train and CL calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import requests

from ttcl.icl_mem0_comparison.protocol import read, save, sha


SCHEMA = {'type': 'object', 'properties': {
    'condition': {'type': 'string'},
    'advice': {'type': 'string'},
    'evidence_step': {'type': 'integer'},
    'feedback_quote': {'type': 'string'},
    'limitation': {'type': 'string'}},
    'required': ['condition', 'advice', 'evidence_step', 'feedback_quote', 'limitation'],
    'additionalProperties': False}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def short(value, n):
    value = str(value)
    return value if len(value) <= n else value[:n // 2] + ' […] ' + value[-n // 2:]


def sources(alf_root, cl_root):
    design = read(alf_root / 'design.json')
    out = []
    for family, games in sorted(design['selected_games'].items()):
        options = []
        for offset, game in enumerate(games, 1):
            index = design['bootstrap'][family]['index'] + offset
            path = (alf_root / 'runs' / 'alfworld' / family / str(design['repeat']) /
                    'memrl' / f'episode_{index:03d}' / 'attempt_1' / 'episode.json')
            if not path.is_file():
                continue
            episode = read(path)
            if not episode['game'].startswith('json_2.1.1/train/'):
                raise ValueError('Non-train ALF source')
            rows = [dict(step=i, action=row['action'], feedback=str(row['observation']))
                    for i, row in enumerate(episode['trajectory'])]
            options.append(dict(benchmark='alfworld', domain=family,
                source_path=str(path.resolve()), source_sha256=sha(path),
                input_sha256=game['sha256'], task=short(episode['initial_observation'], 700),
                reward=episode['reward'], rows=rows))
        for outcome in (0, 1):
            eligible = [x for x in options if bool(x['reward']) == bool(outcome)]
            if eligible:
                out.append(min(eligible, key=lambda x: x['input_sha256']))
    plan = read(cl_root / 'plan.json')
    for domain in ('blind_spectrum_monitoring', 'cohort_studies',
                   'database_exploration', 'exploitable_poker'):
        candidates = []
        for index in range(int(plan['tasks'][domain] * .2)):
            directory = cl_root / 'runs' / 'clbench' / domain / '303' / 'memrl' / f'episode_{index+1:03d}'
            row_path, trace_path = directory / 'row.json', directory / 'public_trajectory.json'
            if not row_path.is_file() or not trace_path.is_file():
                continue
            record, trace = read(row_path), read(trace_path)
            if record['status'] != 'complete' or not trace:
                continue
            rows = [dict(step=i, query=short(step['query'], 450),
                         action=short(json.dumps(step['action'], ensure_ascii=False), 450),
                         feedback=str(step.get('public_feedback', '')))
                    for i, step in enumerate(trace)]
            candidates.append(dict(benchmark='clbench', domain=domain,
                source_path=str(trace_path.resolve()), source_sha256=sha(trace_path),
                row_path=str(row_path.resolve()), row_sha256=sha(row_path),
                input_sha256=record['initial_query_sha256'],
                task=short(trace[0]['query'], 700), reward=record['reward'], rows=rows,
                canonical_index=index))
        out += sorted(candidates, key=lambda x: x['input_sha256'])[:2]
    return out


def prompt(case):
    system = ('Read the whole completed public interaction. Propose AT MOST ONE '
        'reusable experience about an action and its observed feedback. Learn '
        'the situation and action yourself; no domain action ontology is given. '
        'The condition must be observable before the action. The advice must '
        'be no stronger than the cited feedback: no assumptions about hidden '
        'state, unseen alternatives, opponent cards, or optimality. Cite ONE '
        'step number and copy a short exact substring of that step feedback. '
        'A task-level reward cannot prove an individual action was good. If '
        'no supported reusable advice exists, return empty condition, advice '
        'and quote with evidence_step -1. Keep each text field brief. JSON only.')
    rows = []
    for row in case['rows']:
        rows.append({key: short(value, 380) if key == 'feedback' else value
                     for key, value in row.items()})
    return [dict(role='system', content=system),
            dict(role='user', content=json.dumps(dict(task=case['task'],
                task_reward=case['reward'], trajectory=rows), ensure_ascii=False))]


def complete(url, messages, seed):
    body = dict(model='frozen-actor', messages=messages, temperature=0, seed=seed,
        max_tokens=1024, response_format={'type': 'json_schema',
        'json_schema': {'name': 'grounded_experience', 'schema': SCHEMA, 'strict': True}})
    response = requests.post(url.rstrip('/') + '/v1/chat/completions',
                             json=body, timeout=300)
    response.raise_for_status()
    value = response.json()
    choice = value['choices'][0]
    raw = choice['message']['content']
    try:
        parsed, error = json.loads(raw), None
    except (ValueError, TypeError) as exc:
        parsed, error = None, str(exc)
    return dict(raw=raw, parsed=parsed, error=error,
                finish_reason=choice['finish_reason'], usage=value.get('usage', {}),
                prompt_sha256=digest(messages))


def verify(case, completion):
    proposal = completion['parsed']
    if not isinstance(proposal, dict) or completion['finish_reason'] != 'stop':
        return 'unparseable_or_truncated'
    condition, advice = proposal.get('condition'), proposal.get('advice')
    if not condition and not advice:
        return 'abstained'
    index, quote = proposal.get('evidence_step'), proposal.get('feedback_quote')
    if not isinstance(condition, str) or not isinstance(advice, str) or not condition or not advice:
        return 'incomplete_claim'
    if type(index) is not int or index < 0 or index >= len(case['rows']):
        return 'invalid_step'
    if not isinstance(quote, str) or len(quote.strip()) < 8 or quote not in case['rows'][index]['feedback']:
        return 'unmatched_feedback_quote'
    return 'literal_quote_verified_semantics_unreviewed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alf-origin', type=Path, required=True)
    parser.add_argument('--cl-origin', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18559')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    alf_root, cl_root, output = args.alf_origin.resolve(), args.cl_origin.resolve(), args.output.resolve()
    cases = sources(alf_root, cl_root)
    design = dict(schema='grounded_writer_v2_diagnostic',
        source_alf_design_sha256=sha(alf_root / 'design.json'),
        source_cl_plan_sha256=sha(cl_root / 'plan.json'),
        runner_sha256=sha(Path(__file__)), model_url=args.url,
        selection='One input-SHA-minimum failure and one success per ALF train family where available; two input-SHA-minimum CL calibration-prefix trajectories per domain',
        cases=cases, claim_schema=SCHEMA,
        caution='Exact feedback quotation proves provenance only, not usefulness or semantic correctness')
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / 'design.json'
    if frozen.is_file() and read(frozen) != design:
        raise ValueError('Frozen design changed')
    save(frozen, design)
    print(json.dumps(dict(cases=len(cases), design_sha256=sha(frozen))), flush=True)
    if args.prepare_only:
        return
    for index, case in enumerate(cases):
        path = output / f'{index:02d}.json'
        if path.is_file():
            continue
        messages = prompt(case)
        completion = complete(args.url, messages, 97000 + index)
        verdict = verify(case, completion)
        save(path, dict(index=index, source_sha256=case['source_sha256'],
                        input_sha256=case['input_sha256'], completion=completion,
                        mechanical_verdict=verdict))
        print(json.dumps(dict(index=index, domain=case['domain'], reward=case['reward'],
                              verdict=verdict)), flush=True)


if __name__ == '__main__':
    main()
