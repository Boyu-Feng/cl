"""Assistant-token-only SFT of a dedicated memory-writer LoRA."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

from ttcl.memory_writer.core import messages


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False))
    temporary.replace(path)


def encode_example(tokenizer, row, limit):
    prompt = messages(row['memory_before'], row['events'], row['schema'])
    prefix = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(prompt + [{'role': 'assistant', 'content': json.dumps(row['target'])}], tokenize=True)
    if full[:len(prefix)] != prefix:
        raise ValueError('Chat template prefix/token boundary mismatch')
    if len(full) > limit:
        return None
    return {'id': row['id'], 'input_ids': full, 'labels': [-100] * len(prefix) + full[len(prefix):]}


def base_fingerprint(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if 'lora_' not in name:
            digest.update(name.encode())
            digest.update(parameter.detach().reshape(-1)[:8].float().cpu().numpy().tobytes())
    return digest.hexdigest()


def train(args):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / 'config.json', vars(args))
    save_json(out / 'progress.json', {'phase': 'loading', 'step': 0})
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenized, skipped = [], []
    for row in read_rows(args.data):
        encoded = encode_example(tokenizer, row, args.max_length)
        if encoded is None:
            skipped.append(row['id'])
        else:
            tokenized.append(encoded)
    if len(tokenized) < 128:
        raise ValueError('Too few examples fit the context budget')
    validation = []
    for path in args.validation:
        for row in read_rows(path)[:16]:
            item = encode_example(tokenizer, row, args.max_length)
            if item is not None:
                validation.append(item)
    save_json(out / 'data_audit.json', {
        'accepted': len(tokenized), 'skipped_overlength': skipped,
        'supervised_tokens': sum(sum(y != -100 for y in x['labels']) for x in tokenized),
        'max_tokens': max(len(x['input_ids']) for x in tokenized),
        'train_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'validation_count': len(validation), 'prompt_labels_masked': True})
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                                                torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0')
    if getattr(args, 'init_adapter', None):
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
                                           target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                                           'gate_proj', 'up_proj', 'down_proj'],
                                           task_type='CAUSAL_LM', bias='none'))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all('lora_' in name for name, _ in trainable)
    before = base_fingerprint(model)
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=args.learning_rate, weight_decay=0.01)
    rng = random.Random(args.seed)
    order = list(range(len(tokenized)))
    rng.shuffle(order)
    cursor = 0

    def loss_for(item):
        ids = torch.tensor([item['input_ids']], device='cuda:0')
        labels = torch.tensor([item['labels']], device='cuda:0')
        return model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False).loss

    def validate():
        model.eval()
        with torch.inference_mode():
            values = [float(loss_for(item)) for item in validation]
        model.train()
        return sum(values) / len(values) if values else None

    start = time.monotonic()
    initial_validation = validate()
    print(json.dumps({'initial_validation_loss': initial_validation, 'train_examples': len(tokenized),
                      'trainable_parameters': sum(p.numel() for _, p in trainable)}), flush=True)
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        learning_rate = args.learning_rate * min(step / 10, 1.0) * (0.1 + 0.9 * (1 - (step - 1) / args.steps))
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        losses = []
        for _ in range(args.accumulation):
            if cursor == len(order):
                rng.shuffle(order)
                cursor = 0
            item = tokenized[order[cursor]]
            cursor += 1
            loss = loss_for(item)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            losses.append(float(loss.detach()))
            (loss / args.accumulation).backward()
        norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0))
        if not math.isfinite(norm):
            raise ValueError('Nonfinite gradient norm')
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        metrics = {'phase': 'training', 'step': step, 'steps': args.steps,
                   'loss': sum(losses) / len(losses), 'gradient_norm': norm, 'learning_rate': learning_rate,
                   'examples_seen': step * args.accumulation, 'elapsed_seconds': time.monotonic() - start,
                   'peak_gpu_bytes': torch.cuda.max_memory_allocated()}
        with (out / 'training.jsonl').open('a') as handle:
            handle.write(json.dumps(metrics) + '\n')
        save_json(out / 'progress.json', metrics)
        if step == 1 or step % 10 == 0:
            print(json.dumps(metrics), flush=True)
    final_validation = validate()
    after = base_fingerprint(model)
    if before != after:
        raise AssertionError('Frozen base weight sample changed')
    model.save_pretrained(out / 'adapter')
    tokenizer.save_pretrained(out / 'adapter')
    metrics.update(phase='complete', initial_validation_loss=initial_validation,
                   final_validation_loss=final_validation, base_weight_sample_unchanged=True,
                   base_fingerprint_before=before, base_fingerprint_after=after,
                   trainable_parameters=sum(p.numel() for _, p in trainable),
                   adapter=str(out / 'adapter'), supervised_role='memory_writer_only')
    save_json(out / 'metrics.json', metrics)
    save_json(out / 'progress.json', metrics)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--init-adapter', type=Path, help='Continue an existing writer LoRA; base stays frozen')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--validation', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=256)
    p.add_argument('--accumulation', type=int, default=8)
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--max-length', type=int, default=2048)
    args = p.parse_args()
    try:
        train(args)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        save_json(args.output / 'failure.json', {'error_type': type(exc).__name__, 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
