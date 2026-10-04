"""Annotate action-before memory claims for audited ALFWorld coalitions.

The frozen actor sees only the current public task and source-bound memory
text. Its labels are candidate features, never official rewards or reviewed
supervision targets. One result is saved per input-content/memory-text pair.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import Client, read, save, sha
from .audit_alf_coalition_credit import audit as audit_coalitions
from .credit_probe import memory_arms
from .probe_set_conditional_credit import _task, _validate_reviews


PROMPT = """Classify the past experience for a new ALFWorld task. Use only the
two texts below. Do not predict success and do not assume a past failure proves
the current task is impossible.

Current task: {task}

Past experience:
{experience}

Return exactly one JSON object with these three Boolean keys:
"global_environment_block": true only if the experience asserts that the
environment generally cannot support the goal action or interaction, beyond
one failed attempt or missing action precondition;
"actionable_precondition": true only if it recommends a concrete action the
agent can perform before repeating the failed action (such as locating,
opening, taking, activating, checking inventory, or checking target state);
"environment_rewrite": true only if it recommends changing the task wording,
game mechanics, observations, or reward design rather than the agent's actions.
An experience may have more than one true flag. Do not infer from success
metadata; none is supplied.
"""
KEYS = ('global_environment_block', 'actionable_precondition',
        'environment_rewrite')


def _parse(raw: str) -> dict[str, bool]:
    value = raw.strip()
    if value.startswith('```'):
        lines = value.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == '```':
            value = '\n'.join(lines[1:-1])
    parsed = json.loads(value)
    if not isinstance(parsed, dict) or set(parsed) != set(KEYS) or \
            any(type(parsed[key]) is not bool for key in KEYS):
        raise ValueError('Invalid premise annotation schema')
    return {key: parsed[key] for key in KEYS}


def annotate(outputs: list[Path], reviews: list[Path], target: Path,
             url: str) -> dict:
    if len(outputs) != len(reviews) or not outputs:
        raise ValueError('Each coalition group needs its reviewed targets')
    origins = {}
    bindings = []
    for output, review_path in zip(outputs, reviews):
        audit = audit_coalitions(output)
        if audit['missing'] or audit['audited'] != audit['expected']:
            raise ValueError(f'Incomplete coalition source: {output}')
        design = read(output / 'design.json')
        origin = Path(design['origin'])
        kind = output.name
        rows = []
        for item in design['cases']:
            spec, arms = memory_arms(origin, item['case'])
            if spec['source_input_sha256'] != item['input_sha256'] or \
                    spec['snapshot_sha256'] != item['snapshot_sha256'] or \
                    spec['retrieval_sha256'] != item['retrieval_sha256']:
                raise ValueError('Coalition source binding changed')
            rows.append(dict(origin=kind, case=item['case'],
                             input_sha256=item['input_sha256']))
            for i, mid in enumerate(spec['ids']):
                bindings.append(dict(kind=kind, origin=str(origin),
                                     case=item['case'],
                                     input_sha256=item['input_sha256'],
                                     snapshot_sha256=spec['snapshot_sha256'],
                                     retrieval_sha256=spec['retrieval_sha256'],
                                     memory_id=mid, memory_index=i,
                                     memory_text_sha256=spec['memory_text_sha256'][mid],
                                     task=_task(origin, item['case']),
                                     experience=arms[f'only_{i}']))
        review = read(review_path)
        # Review origins are group5/6/7/8, not output directory names.
        actual_kind = {'20261003_alf_three_memory_coalitions_holdout6_v1':'group5',
                       '20261003_alf_three_memory_coalitions_group6_v1':'group6',
                       '20261003_alf_three_memory_coalitions_group7_v1':'group7',
                       '20261004_alf_three_memory_coalitions_group8_v1':'group8'}.get(kind)
        if actual_kind is None:
            raise ValueError(f'Unknown reviewed group: {kind}')
        rows = [dict(row, origin=actual_kind) for row in rows]
        relevant = dict(review, targets=[item for item in review['targets']
                                         if item['origin'] == actual_kind])
        _validate_reviews(rows, relevant, {actual_kind: origin})
        origins[kind] = dict(design_sha256=sha(output / 'design.json'),
                             reviews_sha256=sha(review_path),
                             source_plan_sha256=sha(origin / 'plan.json'))
    hashes = [item['input_sha256'] for item in bindings[::3]]
    if len(bindings) != 3 * len(set(hashes)):
        raise ValueError('Expected three memories per distinct input')
    manifest = dict(schema='alf_credit_premise_annotations_v1',
                    annotator_script_sha256=sha(Path(__file__)),
                    prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),
                    url=url, origins=origins,
                    input_sha256=sorted(set(hashes)),
                    expected=len(bindings),
                    caveat='Actor-generated action-before features, not reviewed targets or causal labels')
    target.mkdir(parents=True, exist_ok=True)
    path = target / 'design.json'
    if path.exists():
        if read(path) != manifest:
            raise ValueError('Frozen premise annotation design changed')
    else:
        save(path, manifest)
    first_origin = Path(bindings[0]['origin'])
    plan = read(first_origin / 'plan.json')
    plan['url'] = url
    client = Client(plan, 0)
    for item in bindings:
        key = hashlib.sha256((item['input_sha256'] + '/' +
                              item['memory_text_sha256']).encode()).hexdigest()
        result_path = target / f'{key}.json'
        if result_path.exists():
            row = read(result_path)
            if any(row.get(field) != item[field] for field in
                   ('input_sha256', 'memory_text_sha256', 'snapshot_sha256',
                    'retrieval_sha256')) or row['design_sha256'] != sha(path):
                raise ValueError('Cached premise binding changed')
            continue
        prompt = PROMPT.format(task=item['task'], experience=item['experience'])
        messages = [{'role':'user', 'content':prompt}]
        result = client.complete(messages, int(key[:8], 16), tokens=192,
                                 temperature=0., top_p=1.)
        row = {key: value for key, value in item.items()
               if key not in ('task', 'experience')}
        row.update(design_sha256=sha(path),
                   task_sha256=hashlib.sha256(item['task'].encode()).hexdigest(),
                   prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                   raw_response=result['raw_response'],
                   input_tokens=result['input_tokens'],
                   output_tokens=result['output_tokens'],
                   rendered_prompt_sha256=result['rendered_prompt_sha256'])
        try:
            row['features'] = _parse(result['raw_response'])
        except (ValueError, json.JSONDecodeError) as exc:
            row['parse_error'] = str(exc)
        save(result_path, row)
        print(json.dumps(dict(case=item['case'], memory_index=item['memory_index'],
                              features=row.get('features'),
                              parse_error=row.get('parse_error'))), flush=True)
    return dict(annotations=sum(1 for p in target.glob('*.json') if p.name != 'design.json'),
                expected=len(bindings))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, nargs='+', required=True)
    parser.add_argument('--reviews', type=Path, nargs='+', required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18559')
    args = parser.parse_args()
    print(json.dumps(annotate([p.resolve() for p in args.outputs],
                              [p.resolve() for p in args.reviews],
                              args.target.resolve(), args.url)))


if __name__ == '__main__':
    main()
