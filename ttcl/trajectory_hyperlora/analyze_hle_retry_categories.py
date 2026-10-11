"""Separate strict HLE retry outcomes by the original failure type.

Does not change scoring or select retries. An originally unparseable answer
does not establish whether the underlying reasoning/option was correct.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/trajectory_hyperlora'))
    args = parser.parse_args(); root = args.results
    score_path = root/'hle_base_scores_20261010.json'
    retry_path = root/'hle_retry_controls_20261010.json'
    audit_path = retry_path.with_suffix('.audit.json')
    plan_path = Path('data/annotations/hle_retry_controls_20261010.json')
    output = root/'hle_retry_failure_categories_20261010.json'
    if output.exists(): raise FileExistsError(output)
    scores, retries, audit, plan = [json.loads(p.read_text()) for p in (score_path,retry_path,audit_path,plan_path)]
    assert retries['complete'] and not retries['failures'] and audit['passed']
    assert audit['retries_sha256'] == sha(retry_path)
    assert audit['plan_sha256'] == retries['plan_sha256'] == sha(plan_path)
    assert plan['scores_sha256'] == sha(score_path)
    assert scores['responses_sha256'] == sha(root/'hle_base_responses_20261010.json')
    first = {r['id']:r for r in scores['rows'] if r['partition']=='evaluation'}
    failed = {k:v for k,v in first.items() if not v['reward']}
    assert len(audit['rows']) == len(failed) and {r['id'] for r in audit['rows']} == set(failed)
    arms = tuple(audit['summary'])
    categories = {}
    for category in ('parsed_wrong','unparseable'):
        rows = [r for r in audit['rows'] if (failed[r['id']]['parse_status']=='parsed') == (category=='parsed_wrong')]
        categories[category] = {'first_failures':len(rows),
            'retry_successes':{a:sum(r['arms'][a]['reward'] for r in rows) for a in arms},
            'failure_residual_vs_controls':{a:{
                'residual_only':[r['id'] for r in rows if r['arms']['failure_residual']['reward']>r['arms'][a]['reward']],
                'control_only':[r['id'] for r in rows if r['arms']['failure_residual']['reward']<r['arms'][a]['reward']]}
                for a in arms if a!='failure_residual'}}
    assert sum(c['first_failures'] for c in categories.values()) == len(failed)
    for a in arms:
        assert sum(c['retry_successes'][a] for c in categories.values()) == audit['summary'][a]['retry_successes']
    result = {'passed_binding_checks':True,'audit_sha256':sha(audit_path),'scores_sha256':sha(score_path),
              'first_wins':sum(r['reward'] for r in first.values()),'evaluation_n':len(first),
              'categories':categories,'scope':'Predeclared strict option-letter scores unchanged. Descriptive stratification; no new answer parser or retry selection.',
              'limitation':'A rescue among initially unparseable answers may involve formatting, answer revision, or both; it is not automatically a reasoning improvement.'}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(categories),flush=True)


if __name__ == '__main__': main()
