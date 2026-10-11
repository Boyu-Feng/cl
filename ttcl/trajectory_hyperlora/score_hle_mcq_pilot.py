"""Strict option-letter scoring for the local HLE text-MCQ pilot.

Not the official full-HLE LLM judge. Unparseable responses count as failures.
No answer keys are copied into the output used to select source trajectories.
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from ttcl.trajectory_hyperlora.prepare_hle_failure_retry import digest


def parse_letter(response):
    fields = []
    for line in response.splitlines():
        # Handle bold field labels or bold letters without searching reasoning.
        plain = line.strip().replace('**', '').replace('`', '')
        match = re.match(r'^Answer\s*:\s*(.*?)\s*$', plain, flags=re.I)
        if match:
            fields.append(match.group(1))
    if len(fields) != 1:
        return None, 'missing_answer_field' if not fields else 'multiple_answer_fields'
    match = re.fullmatch(r'(?:([A-Z])|\(([A-Z])\))[.]?', fields[0])
    if not match:
        return None, 'answer_not_single_option_letter'
    return match.group(1) or match.group(2), 'parsed'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs-dir', type=Path, required=True)
    p.add_argument('--responses', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    raw = args.responses.read_bytes()
    report = json.loads(raw)
    if not report.get('complete') or report['failures']:
        raise ValueError('Complete generation without runtime failures required')
    manifest_path = args.inputs_dir/'manifest.json'
    actor_path = args.inputs_dir/'actor_inputs.json'
    manifest = json.loads(manifest_path.read_text())
    inputs = json.loads(actor_path.read_text())
    keys = json.loads((args.inputs_dir/'grading_keys.json').read_text())
    if digest(inputs) != manifest['actor_inputs_sha256'] or digest(keys) != manifest['grading_keys_sha256']:
        raise ValueError('Prepared inputs or grading keys changed')
    if report['inputs_sha256'] != hashlib.sha256(actor_path.read_bytes()).hexdigest():
        raise ValueError('Responses bind a different actor input file')
    if report['manifest_sha256'] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError('Responses bind a different preparation manifest')
    if len(report['rows']) != len(inputs) or len(keys) != len(inputs):
        raise ValueError('Incomplete response/key coverage')
    if len({x['id'] for x in inputs}) != len(inputs):
        raise ValueError('Duplicate task ids')
    records = []
    for response, question, key in zip(report['rows'], inputs, keys, strict=True):
        if not (response['id'] == question['id'] == key['id']
                and response['partition'] == question['partition']
                and response['input_content_sha256'] == question['input_content_sha256'] == key['input_content_sha256']):
            raise ValueError('Question/response/key content binding mismatch')
        if response['response_sha256'] != digest(response['response']):
            raise ValueError('Response content hash mismatch')
        if not re.fullmatch('[A-Z]', key['answer'].strip()):
            raise ValueError('Unexpected reference format; do not silently reinterpret')
        letter, status = parse_letter(response['response'])
        records.append({'id':question['id'], 'partition':question['partition'],
                        'input_content_sha256':question['input_content_sha256'],
                        'response_sha256':response['response_sha256'], 'parse_status':status,
                        'reward':int(letter == key['answer'].strip()),
                        'hit_token_limit':response['hit_token_limit']})
    summary = {}
    for partition in ('source_collection', 'evaluation'):
        rows = [r for r in records if r['partition'] == partition]
        summary[partition] = {'n':len(rows), 'wins':sum(r['reward'] for r in rows),
                              'parse_status_counts':dict(Counter(r['parse_status'] for r in rows)),
                              'token_limit_count':sum(r['hit_token_limit'] for r in rows)}
    output = {'scope':'Strict letter scoring of a text-MCQ pilot, not official full HLE judging',
              'responses_sha256':hashlib.sha256(raw).hexdigest(),
              'scorer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'grading_keys_sha256':manifest['grading_keys_sha256'],
              'denominator_policy':'All scheduled questions; unparsable answers count as failures',
              'source_success_meaning':'Option letter matches reference; reasoning correctness not judged',
              'rows':records, 'summary':summary,
              'eligible_own_source_ids':[r['id'] for r in records
                                         if r['partition']=='source_collection' and r['reward']==1]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
