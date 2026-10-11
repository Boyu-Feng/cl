"""Score HLE MCQ retries and audit saved bindings/factor arithmetic.

Does not independently rerun the hypernetwork or regenerate answers.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.prepare_hle_retry_controls import ARMS, digest, file_hash
from ttcl.trajectory_hyperlora.score_hle_mcq_pilot import parse_letter
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash


def norm(b, a):
    squared = 0.
    for value, basis in zip(b, a, strict=True):
        value, basis = value[0].double(), basis.double()
        squared += float(((value.T@value)*(basis@basis.T)).sum())/basis.shape[0]**2
    return math.sqrt(max(0., squared))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'baseline', 'scores', 'retries', 'keys', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    plan, baseline, scores, retries, keys = [json.loads(getattr(args,k).read_text())
                                           for k in ('plan','baseline','scores','retries','keys')]
    if not retries.get('complete') or retries['failures']:
        raise ValueError('Complete failure-free retries required')
    assert retries['plan_sha256'] == file_hash(args.plan)
    assert plan['responses_sha256'] == file_hash(args.baseline) == scores['responses_sha256']
    assert plan['scores_sha256'] == file_hash(args.scores)
    assert digest(keys) == scores['grading_keys_sha256']
    assert retries['runner_sha256'] == file_hash(Path(__file__).with_name('evaluate_hle_retry_controls.py'))
    for name, expected in retries['dependencies'].items():
        assert file_hash(Path(__file__).with_name(name)) == expected
    answers = {r['id']:r for r in keys}
    first = {r['id']:r for r in baseline['rows']}
    first_scores = {r['id']:r for r in scores['rows']}
    # Recheck original option scores instead of trusting an outcome flag alone.
    for key, row in first.items():
        assert row['input_content_sha256'] == answers[key]['input_content_sha256']
        letter, status = parse_letter(row['response'])
        assert first_scores[key]['reward'] == int(letter == answers[key]['answer'].strip())
        assert first_scores[key]['parse_status'] == status
    evaluation = [r for r in scores['rows'] if r['partition']=='evaluation']
    failed = {r['id'] for r in evaluation if r['reward']==0}
    assert len(plan['targets']) == len(retries['rows']) == len(failed)
    assert {r['id'] for r in plan['targets']} == failed
    results = []
    for target, row in zip(plan['targets'], retries['rows'], strict=True):
        assert row['id'] == target['id'] and row['retry_binding_sha256'] == target['retry_binding_sha256']
        assert first_scores[target['source_id']]['partition']=='source_collection'
        assert first_scores[target['source_id']]['reward']==1
        assert row['factors_file_sha256'] == file_hash(Path(row['factors_path']))
        f = torch.load(row['factors_path'], map_location='cpu', weights_only=True)
        assert all(torch.isfinite(x).all() for values in f.values() for x in values)
        reference = norm(f['good'], f['A'])
        difference = [bad-good for bad,good in zip(f['bad'],f['good'],strict=True)]
        magnitude = norm(difference, f['A'])
        scale = min(1., .25*reference/magnitude) if magnitude else 0.
        assert abs(scale-row['norms']['scale']) < 1e-5
        expected = [-x*row['norms']['scale'] for x in difference]
        assert all(torch.equal(a,b) for a,b in zip(expected,f['correction'],strict=True))
        assert factors_hash(f['correction']) == row['correction_sha256']
        correction_norm = norm(f['correction'], f['A'])
        random_norm = norm(f['random'], f['A'])
        assert abs(correction_norm-random_norm) <= max(1e-7,correction_norm*1e-5)
        assert correction_norm <= .25*reference*(1+1e-5)
        candidates = {'unchanged_retry':None, 'success_source_only':f['good'],
                      'failure_residual':[a+b for a,b in zip(f['good'],f['correction'],strict=True)],
                      'matched_norm_random':[a+b for a,b in zip(f['good'],f['random'],strict=True)]}
        assert set(row['retries']) == set(ARMS)
        item = {'id':row['id'], 'arms':{}}
        for arm in ARMS:
            response = row['retries'][arm]
            assert response['B_sha256'] == factors_hash(candidates[arm])
            assert response['response_sha256'] == digest(response['response'])
            assert response['prompt_sha256'] == first[row['id']]['prompt_sha256']
            assert response['completion_tokens'] <= plan['max_new_tokens']
            letter, status = parse_letter(response['response'])
            item['arms'][arm] = {'reward':int(letter == answers[row['id']]['answer'].strip()),
                                 'parse_status':status, 'hit_token_limit':response['hit_token_limit']}
        same = row['retries']['unchanged_retry']['response'] == first[row['id']]['response']
        assert same == row['unchanged_response_identical']
        item['unchanged_response_identical'] = same
        results.append(item)
    first_wins = sum(r['reward'] for r in evaluation)
    summary = {arm:{'first_wins':first_wins, 'retry_successes':sum(r['arms'][arm]['reward'] for r in results),
                   'success_within_two_attempts':first_wins+sum(r['arms'][arm]['reward'] for r in results),
                   'evaluation_questions':len(evaluation), 'actual_retries':len(results),
                   'parse_status_counts':dict(Counter(r['arms'][arm]['parse_status'] for r in results))}
               for arm in ARMS}
    output = {'passed':True, 'scope':'Bindings, saved B arithmetic and strict option scoring; no neural/output regeneration',
              'audit_sha256':file_hash(Path(__file__)), 'retries_sha256':file_hash(args.retries),
              'plan_sha256':file_hash(args.plan), 'summary':summary, 'rows':results,
              'unchanged_responses_identical':sum(r['unchanged_response_identical'] for r in results),
              'source_collection_attempts':sum(r['partition']=='source_collection' for r in scores['rows']),
              'limitation':'Subset pilot, old ALFWorld H, four alternative two-attempt methods; not full HLE official score'}
    with args.output.open('x') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
