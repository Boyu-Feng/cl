"""Collect current-base HLE text-MCQ responses without reading grading keys.

This is a pilot baseline/source collector, not the failure-correction runner.
All questions are independent, with no adapter, optimizer, or cross-question text.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.prepare_hle_failure_retry import digest


SYSTEM = ('Solve the question. Format your response in three sections: '
          'Explanation: explain your reasoning; Answer: give only the chosen option letter; '
          'Confidence: give a percentage from 0% to 100%.')


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-new-tokens', type=int, default=8192)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    inputs = json.loads(args.inputs.read_text())
    manifest = json.loads((args.inputs.parent/'manifest.json').read_text())
    if digest(inputs) != manifest['actor_inputs_sha256']:
        raise ValueError('Actor inputs do not match prepared manifest')
    for row in inputs:
        if set(row) != {'id', 'question', 'partition', 'input_content_sha256'}:
            raise ValueError('Unexpected actor field; grading data forbidden')
        if digest({k:v for k,v in row.items() if k != 'input_content_sha256'}) != row['input_content_sha256']:
            raise ValueError('Input content binding mismatch')
    torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(.45, device=args.device)
    result = {'stage': 'base_response_collection', 'complete': False,
              'runner_sha256': file_hash(Path(__file__)), 'inputs_sha256': file_hash(args.inputs),
              'manifest_sha256': file_hash(args.inputs.parent/'manifest.json'),
              'model': str(args.model), 'model_config_sha256': file_hash(args.model/'config.json'),
              'model_weight_sha256': {x.name:file_hash(x) for x in sorted(args.model.glob('*.safetensors'))},
              'system_prompt': SYSTEM, 'prompt_note': 'Adapted HLE Explanation/Answer/Confidence format; not verbatim official prompt',
              'max_new_tokens': args.max_new_tokens, 'do_sample': False, 'seed': 42,
              'adapter': None, 'training_updates': 0, 'reference_answers_read': False,
              'scope': '64 source and 64 disjoint evaluation text MCQs; not full HLE score',
              'rows': [], 'failures': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, result)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa').to(args.device).eval()
    model.requires_grad_(False)
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 50
    for row in inputs:
        try:
            messages = [{'role':'system','content':SYSTEM}, {'role':'user','content':row['question']}]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            encoded = tok(prompt, return_tensors='pt', add_special_tokens=False).to(args.device)
            length = encoded.input_ids.shape[1]
            if length+args.max_new_tokens > model.config.max_position_embeddings:
                raise ValueError('Prompt and completion exceed model context; no silent truncation')
            start = time.monotonic()
            with torch.inference_mode():
                output = model.generate(**encoded, do_sample=False, max_new_tokens=args.max_new_tokens,
                    pad_token_id=tok.eos_token_id)
            continuation = output[0, length:]
            item = {'id':row['id'], 'partition':row['partition'],
                    'input_content_sha256':row['input_content_sha256'], 'prompt_sha256':digest(prompt),
                    'response':tok.decode(continuation, skip_special_tokens=True).strip(),
                    'prompt_tokens':length, 'completion_tokens':len(continuation),
                    'hit_token_limit':len(continuation)==args.max_new_tokens,
                    'elapsed_seconds':time.monotonic()-start}
            item['response_sha256'] = digest(item['response'])
            result['rows'].append(item)
            save(args.output, result)
            print(json.dumps({'n':len(result['rows']), 'partition':row['partition'],
                              'completion_tokens':item['completion_tokens'],
                              'hit_token_limit':item['hit_token_limit']}), flush=True)
        except Exception as exc:
            result['failures'].append({'id':row['id'], 'error':repr(exc)})
            save(args.output, result)
            raise
    result['complete'] = True
    save(args.output, result)


if __name__ == '__main__':
    main()
