"""Frozen H first attempts on all64 HLE evaluation questions, then audit.

Only disjoint source-partition rewards select experience. Evaluation answers
and rewards never enter target construction or generation. No HLE training.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ttcl.trajectory_hyperlora.prepare_hle_retry_controls import (
    digest, file_hash, trajectory, head_tail)
from ttcl.trajectory_hyperlora.contextual_alf_source import (
    task_context_text, contextual_source_fields, contextual_text_fields)
from ttcl.trajectory_hyperlora.evaluate_hle_retry_controls import factors
from ttcl.trajectory_hyperlora.collect_hle_base_responses import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash


ROOT = Path('results/trajectory_hyperlora')
INPUTS = Path('data/hle/failure_retry_pilot_20261010/actor_inputs.json')
BASE = ROOT/'hle_base_responses_20261010.json'
SCORES = ROOT/'hle_base_scores_20261010.json'
PLAN = Path('data/annotations/hle_hyper_first_v91_20261010.json')
OUTPUT = ROOT/'hle_hyper_first_v91_responses_20261010.json'
CHECKPOINT = ROOT/'alf_sibling_simple40_onpolicy600_20261006.pt'
MODEL = Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507')
DEPENDENCIES = ('hle_hyper_first_v91.py', 'prepare_hle_retry_controls.py',
    'evaluate_hle_retry_controls.py', 'collect_hle_base_responses.py',
    'contextual_alf_source.py', 'alfworld_zero_shot_probe.py',
    'direct_composition_pilot.py', 'alfworld_online_accumulate_lora_v1.py',
    'score_hle_mcq_pilot.py')


def prepare_content():
    inputs = json.loads(INPUTS.read_text())
    base = json.loads(BASE.read_text())
    scores = json.loads(SCORES.read_text())
    assert base['complete'] and not base['failures'] and base['adapter'] is None
    assert base['inputs_sha256'] == file_hash(INPUTS)
    assert scores['responses_sha256'] == file_hash(BASE)
    assert scores['scorer_sha256'] == file_hash(Path(__file__).with_name('score_hle_mcq_pilot.py'))
    assert len(inputs) == 128 and len({x['id'] for x in inputs}) == 128
    source_inputs = inputs[:64]
    target_inputs = inputs[64:]
    assert all(x['partition'] == 'source_collection' for x in source_inputs)
    assert all(x['partition'] == 'evaluation' for x in target_inputs)
    responses = {r['id']: r for r in base['rows'] if r['partition'] == 'source_collection'}
    source_scores = {r['id']: r for r in scores['rows'] if r['partition'] == 'source_collection'}
    sources = []
    for q in source_inputs:
        r, s = responses[q['id']], source_scores[q['id']]
        assert q['input_content_sha256'] == r['input_content_sha256'] == s['input_content_sha256']
        assert digest(r['response']) == r['response_sha256'] == s['response_sha256']
        if s['reward'] == 1:
            sources.append((q, r))
    assert len(sources) == 8
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    targets = []
    for q in target_inputs:
        sq, sr = min(sources, key=lambda e: digest(['hle_retry_source_v1', q['id'], e[0]['id']]))
        records, encoding = trajectory(tok, sq['question'], sr['response'], True)
        text, cut = head_tail(tok, q['question'], 1536)
        context = task_context_text(text, text)
        assert len(tok(context, add_special_tokens=False).input_ids) <= 2048
        content = {'id': q['id'], 'input_content_sha256': q['input_content_sha256'],
            'source_id': sq['id'], 'source_response_sha256': sr['response_sha256'],
            'good_records': records, 'good_encoding': encoding,
            'target_context': context, 'target_truncated': cut}
        targets.append({**content, 'binding_sha256': digest(content), 'reviewed_target': True})
    return {'targets': targets, 'targets_sha256': digest(targets),
        'inherited_source_rows': [responses[q['id']] for q in source_inputs],
        'inputs_sha256': file_hash(INPUTS), 'manifest_sha256': base['manifest_sha256'],
        'baseline_sha256': file_hash(BASE), 'source_scores_file_sha256': file_hash(SCORES),
        'checkpoint_sha256': file_hash(CHECKPOINT), 'model_config_sha256': base['model_config_sha256'],
        'model_weight_sha256': base['model_weight_sha256'], 'system_prompt': base['system_prompt'],
        'max_new_tokens': base['max_new_tokens'], 'do_sample': False, 'training_updates': 0,
        'source_selection': 'minimum SHA256([hle_retry_source_v1,target_id,source_id]); disjoint own wins only',
        'dependencies': {n: file_hash(Path(__file__).with_name(n)) for n in DEPENDENCIES},
        'scope': '64 H-first evaluation attempts;64 inherited base source attempts; frozen ALFWorld H; no HLE training; strict MCQ pilot'}


def evaluate(args):
    plan = json.loads(PLAN.read_text())
    for n, h in plan['dependencies'].items():
        assert file_hash(Path(__file__).with_name(n)) == h
    assert file_hash(INPUTS) == plan['inputs_sha256']
    assert file_hash(CHECKPOINT) == plan['checkpoint_sha256']
    assert file_hash(MODEL/'config.json') == plan['model_config_sha256']
    assert {p.name: file_hash(p) for p in sorted(MODEL.glob('*.safetensors'))} == plan['model_weight_sha256']
    assert digest(plan['targets']) == plan['targets_sha256']
    inputs = json.loads(INPUTS.read_text())
    targets = inputs[64:]
    factor_dir = OUTPUT.with_suffix('.factors')
    assert not OUTPUT.exists() and not factor_dir.exists()
    factor_dir.mkdir(parents=True)
    result = {k: plan[k] for k in ('inputs_sha256', 'manifest_sha256', 'model_config_sha256',
        'model_weight_sha256', 'system_prompt', 'max_new_tokens', 'do_sample', 'training_updates', 'scope')}
    result.update(plan_sha256=file_hash(PLAN), runner_sha256=file_hash(Path(__file__)),
        adapter='frozen_hypernetwork', complete=False, failures=[],
        rows=plan['inherited_source_rows'].copy(), reference_answers_read=False,
        inherited_source_attempts=64, new_evaluation_attempts=0)
    save(OUTPUT, result)
    agent, tok = load_agent(MODEL, CHECKPOINT, args.device, .45)
    agent.requires_grad_(False)
    for i, (q, t) in enumerate(zip(targets, plan['targets'], strict=True)):
        try:
            assert q['partition'] == 'evaluation' and q['id'] == t['id']
            assert q['input_content_sha256'] == t['input_content_sha256']
            query = contextual_text_fields(agent, tok, t['target_context'], args.device, 2048, pooling='both')
            fields = contextual_source_fields(agent, tok, t['good_records'], args.device, 2048, pooling='both')
            good = factors(agent, fields, query)
            fp = factor_dir/f'{i:03d}.pt'
            torch.save({'A': [x.a.detach().cpu().clone() for x in agent.adapters], 'B': good}, fp)
            messages = [{'role': 'system', 'content': plan['system_prompt']},
                        {'role': 'user', 'content': q['question']}]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            encoded = tok(prompt, return_tensors='pt', add_special_tokens=False).to(args.device)
            length = encoded.input_ids.shape[1]
            assert length + plan['max_new_tokens'] <= agent.model.config.max_position_embeddings
            agent.set_source(None)
            for layer, value in zip(agent.adapters, good, strict=True):
                layer.b = value.to(args.device)
            start = time.monotonic()
            with torch.inference_mode():
                output = agent.model.generate(**encoded, do_sample=False,
                    max_new_tokens=plan['max_new_tokens'], pad_token_id=tok.eos_token_id)
            tokens = output[0, length:]
            answer = tok.decode(tokens, skip_special_tokens=True).strip()
            row = {'id': q['id'], 'partition': 'evaluation',
                'input_content_sha256': q['input_content_sha256'], 'binding_sha256': t['binding_sha256'],
                'response': answer, 'response_sha256': digest(answer), 'prompt_sha256': digest(prompt),
                'prompt_tokens': length, 'completion_tokens': len(tokens),
                'hit_token_limit': len(tokens) == plan['max_new_tokens'],
                'elapsed_seconds': time.monotonic()-start, 'B_sha256': factors_hash(good),
                'factors_path': str(fp), 'factors_file_sha256': file_hash(fp)}
            agent.set_source(None)
            result['rows'].append(row)
            result['new_evaluation_attempts'] += 1
            save(OUTPUT, result)
            print(json.dumps({'n': i+1, 'completion_tokens': len(tokens)}), flush=True)
        except Exception as exc:
            result['failures'].append({'id': q['id'], 'error': repr(exc)})
            save(OUTPUT, result)
            raise
    result['complete'] = True
    save(OUTPUT, result)


def audit():
    plan = json.loads(PLAN.read_text())
    assert plan == prepare_content()
    r = json.loads(OUTPUT.read_text())
    assert r['complete'] and not r['failures'] and r['training_updates'] == 0
    assert r['plan_sha256'] == file_hash(PLAN) and r['runner_sha256'] == file_hash(Path(__file__))
    assert r['adapter'] == 'frozen_hypernetwork' and r['rows'][:64] == plan['inherited_source_rows']
    assert len(r['rows']) == 128 and r['new_evaluation_attempts'] == 64
    base = json.loads(BASE.read_text())
    for row, target, original in zip(r['rows'][64:], plan['targets'], base['rows'][64:], strict=True):
        assert row['id'] == target['id'] == original['id']
        assert row['binding_sha256'] == target['binding_sha256']
        assert row['prompt_sha256'] == original['prompt_sha256']
        assert row['response_sha256'] == digest(row['response'])
        assert row['completion_tokens'] <= plan['max_new_tokens']
        assert file_hash(Path(row['factors_path'])) == row['factors_file_sha256']
        state = torch.load(row['factors_path'], map_location='cpu', weights_only=True)
        assert all(torch.isfinite(x).all() for values in state.values() for x in values)
        assert factors_hash(state['B']) == row['B_sha256']
    output = OUTPUT.with_suffix('.audit.json')
    assert not output.exists()
    save(output, {'passed': True, 'responses_sha256': file_hash(OUTPUT), 'plan_sha256': file_hash(PLAN),
        'evaluation_n': 64, 'scope': 'Source partition/content, all-target coverage, prompts and saved factors; no neural/output regeneration'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'evaluate', 'audit'))
    p.add_argument('--device', default='cuda:1')
    args = p.parse_args()
    if args.command == 'prepare':
        assert not PLAN.exists()
        save(PLAN, prepare_content())
        print('Prepared64 H-first targets', flush=True)
    elif args.command == 'evaluate':
        evaluate(args)
    else:
        audit()
