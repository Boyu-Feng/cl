"""Four real retries on the same HLE base-first failure set, without gold text.

Uses a frozen ALFWorld H and its A; this is cross-domain transfer. H is read
once per answer. It is not a token-wise adaptation or an HLE-trained H.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.prepare_hle_retry_controls import ARMS, digest, file_hash
from ttcl.trajectory_hyperlora.collect_hle_base_responses import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_source_fields, contextual_text_fields
from ttcl.trajectory_hyperlora.alfworld_failure_retry_residual_v31b import bounded_residual, effective_norm
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash


def factors(agent, fields, target):
    with torch.no_grad():
        agent.set_source(fields, target_fields=target)
        result = [x.b.detach().cpu().float().clone() for x in agent.adapters]
        agent.set_source(None)
    return result


def matched_random(agent, correction, key):
    seed = int(digest(['hle_retry_random_v1', key])[:16], 16) % (2**63)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    random = [torch.randn(x.shape, generator=generator, dtype=torch.float32) for x in correction]
    expected = effective_norm(agent, correction)
    raw = effective_norm(agent, random)
    if raw <= 0:
        raise ValueError('Degenerate random control')
    result = [x*(expected/raw) for x in random]
    actual = effective_norm(agent, result)
    if abs(actual-expected) > max(1e-7, 1e-5*expected):
        raise ValueError('Random-control effective norm mismatch')
    return result, {'seed':seed, 'expected_norm':expected, 'actual_norm':actual}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--responses', type=Path, required=True)
    p.add_argument('--scores', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    p.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    for name in ('inputs', 'responses', 'scores', 'checkpoint'):
        if plan[name+'_sha256'] != file_hash(getattr(args, name)):
            raise ValueError(f'Changed planned dependency: {name}')
    if plan['runner_sha256'] != file_hash(Path(__file__).with_name('prepare_hle_retry_controls.py')):
        raise ValueError('Changed preparation code')
    if plan['arms'] != list(ARMS) or plan['relative_residual_cap'] != .25:
        raise ValueError('Unexpected planned arms/cap')
    if digest(plan['targets']) != plan['targets_sha256']:
        raise ValueError('Changed target bindings')
    inputs = {r['id']:r for r in json.loads(args.inputs.read_text())}
    baseline = json.loads(args.responses.read_text())
    responses = {r['id']:r for r in baseline['rows']}
    weights = {x.name:file_hash(x) for x in sorted(args.model.glob('*.safetensors'))}
    if (weights != baseline['model_weight_sha256'] or not weights
            or file_hash(args.model/'config.json') != plan['model_config_sha256']):
        raise ValueError('Base model differs from first attempt')
    for target in plan['targets']:
        content = {k:v for k,v in target.items() if k not in ('retry_binding_sha256','reviewed_target')}
        if digest(content) != target['retry_binding_sha256'] or not target['reviewed_target']:
            raise ValueError('Changed retry target content')
        if (inputs[target['id']]['partition'] != 'evaluation'
                or inputs[target['source_id']]['partition'] != 'source_collection'):
            raise ValueError('Source/evaluation partition violation')
    factor_dir = args.output.with_suffix('.factors')
    if args.output.exists() or factor_dir.exists():
        raise FileExistsError('Use fresh retry output and factor paths')
    factor_dir.mkdir(parents=True)
    result = {'plan_sha256':file_hash(args.plan), 'runner_sha256':file_hash(Path(__file__)),
              'dependencies':{name:file_hash(Path(__file__).with_name(name)) for name in (
                  'prepare_hle_retry_controls.py','collect_hle_base_responses.py','alfworld_zero_shot_probe.py',
                  'contextual_alf_source.py','direct_composition_pilot.py','alfworld_failure_retry_residual_v31b.py')},
              'model_weight_sha256':weights, 'rows':[], 'failures':[], 'complete':False,
              'gold_answers_read':False, 'training_updates':0,
              'scope':'Four base-first MCQ retry arms; once-per-answer frozen ALFWorld H transfer'}
    save(args.output, result)
    agent, tok = load_agent(args.model, args.checkpoint, args.device, .45)
    agent.requires_grad_(False)
    for index, target in enumerate(plan['targets']):
        row = {'id':target['id'], 'retry_binding_sha256':target['retry_binding_sha256'], 'retries':{}}
        try:
            query = contextual_text_fields(agent, tok, target['target_context'], args.device, 2048, pooling='both')
            good_fields = contextual_source_fields(agent, tok, target['good_records'], args.device, 2048, pooling='both')
            bad_fields = contextual_source_fields(agent, tok, target['bad_records'], args.device, 2048, pooling='both')
            good, bad = factors(agent, good_fields, query), factors(agent, bad_fields, query)
            toward, norms = bounded_residual(agent, good, bad, max_relative=.25)
            correction = [-x for x in toward]
            random, random_info = matched_random(agent, correction, target['id'])
            candidates = {'unchanged_retry':None, 'success_source_only':good,
                          'failure_residual':[a+b for a,b in zip(good, correction, strict=True)],
                          'matched_norm_random':[a+b for a,b in zip(good, random, strict=True)]}
            path = factor_dir/f'{index:03d}.pt'
            torch.save({'A':[x.a.detach().cpu().clone() for x in agent.adapters],
                        'good':good, 'bad':bad, 'correction':correction, 'random':random}, path)
            row.update(factors_path=str(path), factors_file_sha256=file_hash(path), norms=norms,
                       random_info=random_info, correction_sha256=factors_hash(correction))
            messages = [{'role':'system','content':plan['system_prompt']},
                        {'role':'user','content':inputs[target['id']]['question']}]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if digest(prompt) != responses[target['id']]['prompt_sha256']:
                raise ValueError('Retry actor prompt differs from first attempt')
            encoded = tok(prompt, return_tensors='pt', add_special_tokens=False).to(args.device)
            length = encoded.input_ids.shape[1]
            for arm in ARMS:
                agent.set_source(None)
                if candidates[arm] is not None:
                    for layer, value in zip(agent.adapters, candidates[arm], strict=True):
                        layer.b = value.to(args.device)
                start = time.monotonic()
                with torch.inference_mode():
                    output = agent.model.generate(**encoded, do_sample=False,
                        max_new_tokens=plan['max_new_tokens'], pad_token_id=tok.eos_token_id)
                tokens = output[0, length:]
                answer = tok.decode(tokens, skip_special_tokens=True).strip()
                row['retries'][arm] = {'response':answer, 'response_sha256':digest(answer),
                    'prompt_sha256':digest(prompt), 'prompt_tokens':length,
                    'completion_tokens':len(tokens), 'hit_token_limit':len(tokens)==plan['max_new_tokens'],
                    'elapsed_seconds':time.monotonic()-start, 'B_sha256':factors_hash(candidates[arm])}
                if arm == 'unchanged_retry':
                    row['unchanged_response_identical'] = answer == responses[target['id']]['response']
                save(args.output, {**result, 'in_progress_row':row})
                print(json.dumps({'target':index, 'arm':arm, 'completion_tokens':len(tokens)}), flush=True)
            agent.set_source(None)
            result['rows'].append(row)
            save(args.output, result)
        except Exception as exc:
            result['failures'].append({'id':target['id'], 'error':repr(exc), 'partial_row':row})
            save(args.output, result)
            raise
    result['complete'] = True
    save(args.output, result)


if __name__ == '__main__':
    main()
