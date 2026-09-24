"""Writer-only DPO, using frozen reference log probabilities cached before updates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

from ttcl.memory_writer.train import base_fingerprint, encode_example, read_rows, save_json


def preference_loss(chosen, rejected, reference_chosen, reference_rejected, beta):
    import torch.nn.functional as F
    return -F.logsigmoid(beta * ((chosen - rejected) - (reference_chosen - reference_rejected)))


def orient_pairs(rows, shuffle_labels, seed):
    rows = [dict(row) for row in rows]
    flips = set()
    if shuffle_labels:
        flips = set(random.Random(seed).sample(range(len(rows)), len(rows) // 2))
    for i, row in enumerate(rows):
        row['label_flipped'] = i in flips
        if i in flips:
            row['chosen'], row['rejected'] = row['rejected'], row['chosen']
    return rows


def train(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / 'config.json', vars(args))
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    pairs, dropped = [], []
    source = orient_pairs(read_rows(args.pairs), args.shuffle_labels, args.seed)
    for row in source:
        encoded = [encode_example(tokenizer, dict(row, target=json.loads(row[side])), args.max_length)
                   for side in ('chosen', 'rejected')]
        if any(item is None for item in encoded):
            dropped.append(row['id'])
        else:
            pairs.append((row, *encoded))
    if len(pairs) < 8:
        raise ValueError('Fewer than eight fitting pairs; refusing preference training')
    save_json(out / 'data_audit.json', dict(accepted=len(pairs), skipped=dropped,
                                          pair_sha256=hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
                                          flipped=sum(row['label_flipped'] for row, _, _ in pairs),
                                          assistant_tokens_only=True))
    base = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                                               torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0')
    model = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert parameters and all('lora_' in n for n, _ in parameters)
    # Standard DPO policy/reference comparison must not introduce dropout noise.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    before = base_fingerprint(model)

    def log_probability(item):
        ids = torch.tensor([item['input_ids']], device='cuda:0')
        labels = torch.tensor([item['labels']], device='cuda:0')
        loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False).loss
        return -loss * (labels[:, 1:] != -100).sum()

    references = []
    model.eval()
    with torch.no_grad():
        for i, (_, chosen, rejected) in enumerate(pairs):
            references.append((float(log_probability(chosen)), float(log_probability(rejected))))
            save_json(out / 'progress.json', dict(phase='reference_cache', completed=i + 1, pairs=len(pairs)))
    save_json(out / 'reference_logps.json', [dict(id=row['id'], chosen=ref[0], rejected=ref[1],
                                               label_flipped=row['label_flipped'])
                                           for (row, _, _), ref in zip(pairs, references)])
    model.train()
    optimizer = torch.optim.AdamW([p for _, p in parameters], lr=args.learning_rate, weight_decay=0.01)
    order = list(range(len(pairs)))
    rng = random.Random(args.seed)
    rng.shuffle(order)
    cursor, start = 0, time.monotonic()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        losses, margins = [], []
        lr = args.learning_rate * (0.1 + 0.9 * (1 - (step - 1) / args.steps))
        for group in optimizer.param_groups:
            group['lr'] = lr
        for _ in range(args.accumulation):
            if cursor == len(order):
                rng.shuffle(order)
                cursor = 0
            i = order[cursor]
            cursor += 1
            _, chosen, rejected = pairs[i]
            pc, pr = log_probability(chosen), log_probability(rejected)
            rc, rr = references[i]
            loss = preference_loss(pc, pr, rc, rr, args.beta)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite preference loss')
            losses.append(float(loss.detach()))
            margins.append(float(((pc - pr) - (rc - rr)).detach()))
            (loss / args.accumulation).backward()
        norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in parameters], 1.0))
        if not math.isfinite(norm):
            raise ValueError('Nonfinite preference gradient')
        optimizer.step()
        metrics = dict(phase='training', step=step, steps=args.steps, loss=sum(losses) / len(losses),
                       mean_margin=sum(margins) / len(margins), gradient_norm=norm, learning_rate=lr,
                       pairs_seen=step * args.accumulation, elapsed_seconds=time.monotonic() - start)
        with (out / 'training.jsonl').open('a') as handle:
            handle.write(json.dumps(metrics) + '\n')
        save_json(out / 'progress.json', metrics)
        if step == 1 or step % 8 == 0:
            print(json.dumps(metrics), flush=True)
    after = base_fingerprint(model)
    if after != before:
        raise AssertionError('Frozen base parameter sample changed')
    model.save_pretrained(out / 'adapter')
    tokenizer.save_pretrained(out / 'adapter')
    metrics.update(phase='complete', accepted_pairs=len(pairs), base_weight_sample_unchanged=True,
                   base_fingerprint=after, reader_trained=False, labels_shuffled=args.shuffle_labels,
                   reference_adapter=str(args.init_adapter), algorithm='DPO', beta=args.beta)
    save_json(out / 'metrics.json', metrics)
    save_json(out / 'progress.json', metrics)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--init-adapter', type=Path, required=True)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shuffle-labels', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--max-length', type=int, default=4096)
    p.add_argument('--learning-rate', type=float, default=5e-6)
    p.add_argument('--beta', type=float, default=0.1)
    args = p.parse_args()
    try:
        train(args)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        save_json(args.output / 'failure.json', {'error_type': type(exc).__name__, 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
