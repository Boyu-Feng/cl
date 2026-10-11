"""Bind HLE failure retries to own successful source responses, without gold text.

Run after baseline collection and strict pilot scoring complete. The frozen
ALFWorld hypernetwork is a cross-domain transfer, not a freshly trained HLE H.
"""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from ttcl.trajectory_hyperlora.contextual_alf_source import source_text, task_context_text
from ttcl.trajectory_hyperlora.prepare_hle_failure_retry import digest

ARMS = ('unchanged_retry', 'failure_residual', 'matched_norm_random')


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def head_tail(tokenizer, text, budget):
    tokens = tokenizer(text, add_special_tokens=False).input_ids
    if len(tokens) <= budget:
        return text, False
    half = budget//2
    return (tokenizer.decode(tokens[:half], skip_special_tokens=True)
            + '\n[Middle omitted to fit encoder budget]\n'
            + tokenizer.decode(tokens[-half:], skip_special_tokens=True)), True


def trajectory(tokenizer, question, response, success):
    observation, question_cut = head_tail(tokenizer, question, 768)
    action, response_cut = head_tail(tokenizer, response, 1024)
    records = [{'observation':observation, 'action':action,
                'feedback':'task completed.' if success else 'task not completed.'}]
    size = len(tokenizer(source_text(records), add_special_tokens=False).input_ids)
    if size > 2048:
        raise ValueError('Encoded trajectory exceeds declared budget')
    return records, {'question_truncated':question_cut, 'response_truncated':response_cut,
                     'encoded_tokens':size, 'records_sha256':digest(records)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--responses', type=Path, required=True)
    p.add_argument('--scores', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    p.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    args = p.parse_args()
    inputs = json.loads(args.inputs.read_text())
    report = json.loads(args.responses.read_text())
    scores = json.loads(args.scores.read_text())
    if report.get('adapter') != 'frozen_hypernetwork':
        raise ValueError('Requires H-first responses')
    first_audit = args.responses.with_suffix('.audit.json')
    first_checked = json.loads(first_audit.read_text())
    assert first_checked['passed'] and first_checked['responses_sha256'] == file_hash(args.responses)
    if not report.get('complete') or report['failures']:
        raise ValueError('Baseline collection is not complete')
    if scores['responses_sha256'] != file_hash(args.responses) or report['inputs_sha256'] != file_hash(args.inputs):
        raise ValueError('Changed baseline inputs/responses')
    if scores['scorer_sha256'] != file_hash(Path(__file__).with_name('score_hle_mcq_pilot.py')):
        raise ValueError('Different scoring implementation')
    if len(inputs) != len(report['rows']) or len(inputs) != len(scores['rows']):
        raise ValueError('Incomplete bindings')
    entries = []
    for question, response, score in zip(inputs, report['rows'], scores['rows'], strict=True):
        if not (question['id'] == response['id'] == score['id']
                and question['partition'] == response['partition'] == score['partition']
                and question['input_content_sha256'] == response['input_content_sha256'] == score['input_content_sha256']
                and digest(response['response']) == response['response_sha256'] == score['response_sha256']
                and score['reward'] in (0, 1)):
            raise ValueError('Question/response/score binding mismatch')
        entries.append((question, response, score))
    sources = [e for e in entries if e[0]['partition']=='source_collection' and e[2]['reward']==1]
    if not sources:
        raise ValueError('No verified own source success; correction arm cannot be constructed')
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    fixed = {'scope':'Previously ALFWorld-trained H transfer to HLE text-MCQ H-first failure retry',
             'arms':list(ARMS), 'max_attempts_per_method':2, 'max_new_tokens':report['max_new_tokens'],
             'system_prompt':report['system_prompt'], 'do_sample':False,
             'relative_residual_cap':.25, 'residual':'clip(B_good-B_failed, 0.25*effective_norm(B_good))',
             'retry_factors':'original H B_good for unchanged; B_good+delta for failure/random',
             'source_selection':'minimum SHA256([hle_retry_source_v1,target_id,source_id]) among disjoint own source wins',
             'random_direction':'CPU float32 normal, SHA256 target_id seed; effective BA/r norm matched to true residual',
             'trajectory_encoding':'question head384+tail384, response head512+tail512, explicit omission markers, binary feedback; <=2048 tokens',
             'target_encoding':'question head768+tail768 if needed, explicit omission marker; <=2048 tokens',
             'training_updates':0, 'reference_answer_text_in_actor':False,
             'inputs_sha256':file_hash(args.inputs), 'responses_sha256':file_hash(args.responses),
             'first_audit_sha256':file_hash(first_audit), 'scores_sha256':file_hash(args.scores), 'checkpoint_sha256':file_hash(args.checkpoint),
             'model_config_sha256':file_hash(args.model/'config.json'),
             'runner_sha256':file_hash(Path(__file__)),
             'limitations':['Not full official HLE judging.', '64 extra source-collection attempts must be counted.',
                            'Strict option success does not verify reasoning.', 'Old H has ALFWorld pretraining.']}
    targets = []
    for question, response, score in entries:
        if question['partition'] != 'evaluation' or score['reward']:
            continue
        sq, sr, _ = min(sources, key=lambda e:digest(['hle_retry_source_v1', question['id'], e[0]['id']]))
        good, good_info = trajectory(tok, sq['question'], sr['response'], True)
        bad, bad_info = trajectory(tok, question['question'], response['response'], False)
        target_text, target_cut = head_tail(tok, question['question'], 1536)
        query = task_context_text(target_text, target_text)
        if len(tok(query, add_special_tokens=False).input_ids)>2048:
            raise ValueError('Target encoder input exceeds budget')
        content = {'id':question['id'], 'input_content_sha256':question['input_content_sha256'],
                   'first_response_sha256':response['response_sha256'], 'first_B_sha256':response['B_sha256'], 'first_factors_path':response['factors_path'], 'first_factors_file_sha256':response['factors_file_sha256'], 'source_id':sq['id'],
                   'source_input_content_sha256':sq['input_content_sha256'],
                   'source_response_sha256':sr['response_sha256'], 'good_records':good, 'bad_records':bad,
                   'good_encoding':good_info, 'bad_encoding':bad_info,
                   'target_context':query, 'target_truncated':target_cut, 'protocol_sha256':digest(fixed)}
        targets.append({**content, 'retry_binding_sha256':digest(content), 'reviewed_target':True})
    output = {**fixed, 'targets':targets, 'targets_sha256':digest(targets),
              'source_success_count':len(sources), 'stage':'retry_inputs_prepared_not_evaluated'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'retry_targets':len(targets), 'own_sources':len(sources), 'evaluation_started':False}))


if __name__ == '__main__':
    main()
