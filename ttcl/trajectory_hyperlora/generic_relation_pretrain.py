"""Pretrain an unlabelled trajectory relation matrix for future-action transfer.

No cue-specific slots, rule-ID classifier, or per-step labels are used. A
query-conditioned bilinear head reads a pooled observation/action relation;
only future-query correct actions supervise training. This is a representation
diagnostic before attaching a LoRA hypernetwork, not the final actor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.slot_lora_oracle_pilot import (
    CUES, DEV_RULES, EVAL_NUMBERS, EVAL_QUERY, TEST_RULES,
    TRAIN_QUERY, TRAIN_RULES, action, source_records,
)


class RelationMemory(nn.Module):
    def __init__(self, embedding: nn.Embedding, width: int = 32) -> None:
        super().__init__()
        self.embedding = embedding
        size = embedding.embedding_dim
        self.observation = nn.Sequential(nn.LayerNorm(size),
                                         nn.Linear(size, width), nn.Tanh())
        self.action = nn.Sequential(nn.LayerNorm(size),
                                    nn.Linear(size, width), nn.Tanh())
        self.readout = nn.Linear(width, 2)

    def field(self, tokenizer, strings: list[str], device: str) -> torch.Tensor:
        batch = tokenizer(strings, add_special_tokens=False,
                          padding=True, truncation=True, max_length=80,
                          return_tensors="pt").to(device)
        with torch.no_grad():
            embedded = self.embedding(batch.input_ids).float()
        mask = batch.attention_mask.unsqueeze(-1)
        return (embedded * mask).sum(1) / mask.sum(1)

    def relation(self, tokenizer, source: list[dict], device: str) -> torch.Tensor:
        observation = self.observation(self.field(
            tokenizer, [row["observation"] for row in source], device))
        action = self.action(self.field(
            tokenizer, [row["action"] for row in source], device))
        return torch.einsum("si,sj->ij", observation, action) / len(source)

    def forward(self, tokenizer, source: list[dict], questions: list[str],
                device: str) -> tuple[torch.Tensor, torch.Tensor]:
        relation = self.relation(tokenizer, source, device)
        query = self.observation(self.field(tokenizer, questions, device))
        memory = query @ relation
        return self.readout(memory), relation


def evaluate(model: RelationMemory, tokenizer, device: str,
             split: str) -> dict:
    rows = []
    model.eval()
    with torch.no_grad():
        for rule in DEV_RULES if split == "dev" else TEST_RULES:
            for source_seed in (3001, 3002):
                seed = source_seed + 31 * rule + (0 if split == "dev" else 10000)
                source = source_records(rule, random.Random(seed), test=True)
                wrong = source_records(rule ^ 15, random.Random(seed), test=True)
                if any(left["observation"] != right["observation"]
                       for left, right in zip(source, wrong, strict=True)):
                    raise ValueError("Source swap changed test observations")
                source_hash = hashlib.sha256(json.dumps(
                    source, sort_keys=True).encode()).hexdigest()
                queries = [EVAL_QUERY[split].format(number=number, cue=cue)
                           for cue in CUES for number in EVAL_NUMBERS[split]]
                logits, relation = model(tokenizer, source, queries, device)
                wrong_logits, _ = model(tokenizer, wrong, queries, device)
                for cue_index, cue in enumerate(CUES):
                    for number_index, number in enumerate(EVAL_NUMBERS[split]):
                        index = cue_index * len(EVAL_NUMBERS[split]) + number_index
                        expected = int(action(rule, cue_index) == "RIGHT")
                        rows.append({"rule": rule, "source_seed": source_seed,
                                     "cue": cue, "number": number,
                                     "source_sha256": source_hash,
                                     "query_sha256": hashlib.sha256(
                                         queries[index].encode()).hexdigest(),
                                     "expected": expected,
                                     "correct": int(logits[index].argmax()) == expected,
                                     "wrong": int(wrong_logits[index].argmax()) == expected,
                                     "changed": int(logits[index].argmax()) !=
                                                int(wrong_logits[index].argmax())})
    return {"n": len(rows),
            "correct_source_success": sum(row["correct"] for row in rows),
            "wrong_source_success": sum(row["wrong"] for row in rows),
            "source_swap_action_changes": sum(row["changed"] for row in rows),
            "rows": rows}


def run(args: argparse.Namespace) -> dict:
    if args.output.exists() or args.checkpoint.exists() or args.steps < 1:
        raise ValueError("Fresh output and positive training steps required")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                                   device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    actor = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    memory = RelationMemory(actor.get_input_embeddings(), args.width).to(args.device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in memory.parameters() if parameter.requires_grad],
        lr=args.lr, weight_decay=0)
    losses = []
    for step in range(args.steps):
        rule = TRAIN_RULES[step % len(TRAIN_RULES)]
        source = source_records(rule, rng, test=False)
        questions = [rng.choice(TRAIN_QUERY).format(number=rng.randrange(90, 120),
                                                    cue=cue) for cue in CUES]
        targets = torch.tensor([int(action(rule, index) == "RIGHT")
                                for index in range(len(CUES))], device=args.device)
        logits, _ = memory(tokenizer, source, questions, args.device)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in memory.parameters()
             if parameter.requires_grad], 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 100 == 0:
            print(json.dumps({"step": step + 1,
                              "mean_loss": sum(losses[-100:]) / 100}), flush=True)
    dev = evaluate(memory, tokenizer, args.device, "dev")
    test = evaluate(memory, tokenizer, args.device, "test")
    result = {"protocol": "Generic permutation-invariant observation/action relation matrix trained only by future-query action loss on even-parity rules; no cue slots, rule IDs, per-step labels or LoRA; odd-parity whole-rule holdout",
              "seed": args.seed, "steps": args.steps, "width": args.width,
              "lr": args.lr, "last_100_loss": sum(losses[-100:]) / 100,
              "dev": dev, "test": test}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    torch.save({"trainable_state": {name: parameter.detach().cpu()
                                   for name, parameter in memory.named_parameters()
                                   if parameter.requires_grad},
                "seed": args.seed, "width": args.width, "steps": args.steps},
               args.checkpoint)
    print(json.dumps({"dev": {k: v for k, v in dev.items() if k != "rows"},
                      "test": {k: v for k, v in test.items() if k != "rows"}}),
          flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.35)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
