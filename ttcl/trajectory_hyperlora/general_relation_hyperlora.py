"""Source-agnostic trajectory-to-LoRA pilot on four-rule composition.

The generator has no named cue slots or per-rule expert bank. It reads raw
observation/action/feedback text, pools step relations, and predicts a B factor
for each mounted LoRA layer. Training uses only future-query action loss on
even-parity rules. Whole odd-parity rules remain held out.
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

from ttcl.trajectory_hyperlora.direct_composition_pilot import GeneratedDownProjection
from ttcl.trajectory_hyperlora.pilot import prompt
from ttcl.trajectory_hyperlora.paired_counterfactual_lora import paired_objective
from ttcl.trajectory_hyperlora.relational_router_pilot import tokenize_records
from ttcl.trajectory_hyperlora.slot_lora_oracle_pilot import (
    CUES, DEV_RULES, EVAL_NUMBERS, EVAL_QUERY, TEST_RULES,
    TRAIN_QUERY, TRAIN_RULES, action, source_records,
)


class GenericRelationEncoder(nn.Module):
    """Permutation-invariant relation pool without task-specific slots."""

    def __init__(self, embedding: nn.Embedding, width: int = 64) -> None:
        super().__init__()
        self.embedding = embedding
        size = embedding.embedding_dim
        self.observation = nn.Sequential(nn.LayerNorm(size),
                                         nn.Linear(size, width), nn.Tanh())
        self.action = nn.Sequential(nn.LayerNorm(size),
                                    nn.Linear(size, width), nn.Tanh())
        self.feedback = nn.Sequential(nn.LayerNorm(size),
                                      nn.Linear(size, width), nn.Sigmoid())

    def field_mean(self, pair) -> torch.Tensor:
        ids, mask = pair
        with torch.no_grad():
            values = self.embedding(ids).float()
        weights = mask.unsqueeze(-1)
        return (values * weights).sum(1) / weights.sum(1)

    def forward(self, fields) -> torch.Tensor:
        observation = self.observation(self.field_mean(fields["observation"]))
        action = self.action(self.field_mean(fields["action"]))
        feedback = self.feedback(self.field_mean(fields["feedback"]))
        return (observation * action * feedback).mean(0, keepdim=True)


class GenericRelationHyperLoRA(nn.Module):
    def __init__(self, model: nn.Module, rank: int = 8,
                 layers: int = 2, width: int = 64) -> None:
        super().__init__()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model
        self.encoder = GenericRelationEncoder(model.get_input_embeddings(), width)
        self.adapters = nn.ModuleList()
        self.b_heads = nn.ModuleList()
        for block in model.model.layers[-layers:]:
            adapter = GeneratedDownProjection(block.mlp.down_proj, rank)
            adapter.scale = rank / 2
            block.mlp.down_proj = adapter
            self.adapters.append(adapter)
            head = nn.Linear(width, adapter.base.out_features * rank)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.b_heads.append(head)

    def source_factors(self, fields) -> list[torch.Tensor]:
        latent = self.encoder(fields)
        return [head(latent).reshape(1, adapter.base.out_features, adapter.rank)
                for head, adapter in zip(self.b_heads, self.adapters, strict=True)]

    def mount(self, factors: list[torch.Tensor] | None) -> None:
        if factors is not None and len(factors) != len(self.adapters):
            raise ValueError("One generated factor per mounted layer required")
        for layer, adapter in enumerate(self.adapters):
            adapter.b = factors[layer] if factors is not None else None


def first_logits(agent: GenericRelationHyperLoRA, tokenizer, question: str,
                 device: str) -> torch.Tensor:
    ids = tokenizer(prompt(tokenizer, question), add_special_tokens=False).input_ids
    return agent.model(input_ids=torch.tensor([ids], device=device),
                       use_cache=False).logits[:, -1, :].float()


def evaluate(agent, tokenizer, device: str, split: str) -> dict:
    if split not in ("dev", "test"):
        raise ValueError(split)
    agent.eval()
    action_ids = {label: tokenizer(label, add_special_tokens=False).input_ids[0]
                  for label in ("LEFT", "RIGHT")}
    rows = []
    with torch.no_grad():
        public_sources = [source_records(rule, random.Random(5000 + rule),
                                         test=False) for rule in TRAIN_RULES]
        public_by_source = [agent.source_factors(tokenize_records(
            tokenizer, source, device)) for source in public_sources]
        public = [torch.stack([source[layer] for source in public_by_source]).mean(0)
                  for layer in range(len(agent.adapters))]
        for rule in DEV_RULES if split == "dev" else TEST_RULES:
            for source_seed in (3001, 3002):
                seed = source_seed + 31 * rule + (0 if split == "dev" else 10000)
                source = source_records(rule, random.Random(seed), test=True)
                wrong = source_records(rule ^ 15, random.Random(seed), test=True)
                if any(left["observation"] != right["observation"] or
                       left["feedback"] != right["feedback"]
                       for left, right in zip(source, wrong, strict=True)):
                    raise ValueError("Test source swap changes more than actions")
                correct_factors = agent.source_factors(tokenize_records(
                    tokenizer, source, device))
                wrong_factors = agent.source_factors(tokenize_records(
                    tokenizer, wrong, device))
                source_hash = hashlib.sha256(json.dumps(
                    source, sort_keys=True).encode()).hexdigest()
                wrong_hash = hashlib.sha256(json.dumps(
                    wrong, sort_keys=True).encode()).hexdigest()
                for cue_index, cue in enumerate(CUES):
                    for number in EVAL_NUMBERS[split]:
                        question = EVAL_QUERY[split].format(number=number,
                                                           cue=cue)
                        expected = action(rule, cue_index)
                        row = {"rule": rule, "source_seed": source_seed,
                               "cue": cue, "number": number, "expected": expected,
                               "source_sha256": source_hash,
                               "wrong_source_sha256": wrong_hash,
                               "query_sha256": hashlib.sha256(
                                   question.encode()).hexdigest()}
                        for arm, factors in (("base", None), ("public", public),
                                             ("correct", correct_factors),
                                             ("wrong", wrong_factors)):
                            agent.mount(factors)
                            logits = first_logits(agent, tokenizer, question, device)
                            choice = int(logits.argmax(-1)[0])
                            row[arm] = {"first_token": tokenizer.decode(choice).strip(),
                                        "correct": choice == action_ids[expected]}
                        agent.mount(None)
                        rows.append(row)
    return {"n": len(rows),
            "success": {arm: sum(row[arm]["correct"] for row in rows)
                        for arm in ("base", "public", "correct", "wrong")},
            "rows": rows}


def run(args: argparse.Namespace) -> dict:
    if (args.output.exists() or args.checkpoint.exists() or args.steps < 1 or
            args.margin < 0 or args.rank_weight < 0):
        raise ValueError("Fresh outputs and valid training budget required")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                                   device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    action_ids = {label: tokenizer(label, add_special_tokens=False).input_ids
                  for label in ("LEFT", "RIGHT")}
    if any(len(ids) != 1 for ids in action_ids.values()):
        raise ValueError("Action words must each have one token")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    model.config.use_cache = False
    agent = GenericRelationHyperLoRA(model, args.rank, args.layers,
                                     args.width).to(args.device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in agent.parameters() if parameter.requires_grad],
        lr=args.lr, weight_decay=0)
    losses = []
    for step in range(args.steps):
        rule = TRAIN_RULES[step % len(TRAIN_RULES)]
        if args.paired:
            pair_seed = rng.randrange(2**32)
            source = source_records(rule, random.Random(pair_seed), test=False)
            opposite = source_records(rule ^ 15, random.Random(pair_seed),
                                      test=False)
            if any(left["observation"] != right["observation"] or
                   left["feedback"] != right["feedback"]
                   for left, right in zip(source, opposite, strict=True)):
                raise ValueError("Training source swap changes more than actions")
            opposite_fields = tokenize_records(tokenizer, opposite, args.device)
        else:
            source = source_records(rule, rng, test=False)
        source_fields = tokenize_records(tokenizer, source, args.device)
        optimizer.zero_grad(set_to_none=True)
        for cue_index, cue in enumerate(CUES):
            number = rng.randrange(90, 120)
            question = rng.choice(TRAIN_QUERY).format(number=number, cue=cue)
            agent.mount(agent.source_factors(source_fields))
            logits = first_logits(agent, tokenizer, question, args.device)
            agent.mount(None)
            target = torch.tensor([action_ids[action(rule, cue_index)][0]],
                                  device=args.device)
            if args.paired:
                agent.mount(agent.source_factors(opposite_fields))
                opposite_logits = first_logits(agent, tokenizer, question,
                                               args.device)
                agent.mount(None)
                loss = paired_objective(
                    logits, opposite_logits, int(target[0]),
                    action_ids[action(rule ^ 15, cue_index)][0],
                    args.margin, args.rank_weight) / len(CUES)
            else:
                loss = F.cross_entropy(logits, target) / len(CUES)
            loss.backward()
            losses.append(float(loss.detach()) * len(CUES))
        torch.nn.utils.clip_grad_norm_(
            [p for p in agent.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if (step + 1) % 100 == 0:
            print(json.dumps({"step": step + 1,
                              "mean_loss": sum(losses[-400:]) / min(400, len(losses))}),
                  flush=True)
    agent.mount(None)
    dev = evaluate(agent, tokenizer, args.device, "dev")
    test = evaluate(agent, tokenizer, args.device, "test")
    result = {"protocol": "No named parameter slots or per-rule bank; generic step relation pool predicts LoRA B factors; actor frozen; future-query action supervision on eight even-parity rules; optional same-query opposite-history paired loss; odd-parity dev/test held out; first-token exact scoring; no step labels or RL",
              "seed": args.seed, "steps": args.steps, "lr": args.lr,
              "paired_training": args.paired,
              "margin": args.margin if args.paired else None,
              "rank_weight": args.rank_weight if args.paired else None,
              "rank": args.rank, "layers": args.layers, "width": args.width,
              "train_rules": TRAIN_RULES, "dev_rules": DEV_RULES,
              "test_rules": TEST_RULES,
              "last_100_step_loss": sum(losses[-400:]) / min(400, len(losses)),
              "dev": dev, "test": test}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    torch.save({"trainable_state": {name: p.detach().cpu()
                                   for name, p in agent.named_parameters()
                                   if p.requires_grad},
                "seed": args.seed, "rank": args.rank,
                "layers": args.layers, "width": args.width,
                "steps": args.steps}, args.checkpoint)
    print(json.dumps({"dev": dev["success"], "test": test["success"]}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.35)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--rank-weight", type=float, default=.3)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
