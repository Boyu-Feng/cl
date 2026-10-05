"""Compile a pretrained generic trajectory relation into LoRA factors.

The relation encoder was pretrained only on future-query actions and remains
frozen here. This generator has no named cue slots, per-rule adapters, or
stepwise target labels. It predicts B factors directly from the pooled raw
trajectory relation; Qwen base parameters stay frozen.
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
from ttcl.trajectory_hyperlora.general_relation_hyperlora import first_logits
from ttcl.trajectory_hyperlora.generic_relation_pretrain import RelationMemory
from ttcl.trajectory_hyperlora.feedback_relation_pretrain import (
    FeedbackRelationMemory, add_corrections,
)
from ttcl.trajectory_hyperlora.paired_counterfactual_lora import paired_objective
from ttcl.trajectory_hyperlora.slot_lora_oracle_pilot import (
    CUES, DEV_RULES, TEST_RULES, TRAIN_QUERY, TRAIN_RULES,
    action, source_records,
)


class PretrainedRelationHyperLoRA(nn.Module):
    def __init__(self, model: nn.Module, relation_checkpoint: dict,
                 rank: int = 8, layers: int = 2,
                 latent_width: int = 128) -> None:
        super().__init__()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model
        width = relation_checkpoint["width"]
        relation_class = (FeedbackRelationMemory if relation_checkpoint.get("kind") ==
                          "feedback_relation" else RelationMemory)
        self.relation = relation_class(model.get_input_embeddings(), width)
        parameters = dict(self.relation.named_parameters())
        with torch.no_grad():
            for name, saved in relation_checkpoint["trainable_state"].items():
                if name not in parameters or parameters[name].shape != saved.shape:
                    raise ValueError(f"Relation checkpoint mismatch: {name}")
                parameters[name].copy_(saved.to(parameters[name].device))
        for parameter in self.relation.parameters():
            parameter.requires_grad_(False)
        self.projector = nn.Sequential(nn.LayerNorm(width * width),
                                       nn.Linear(width * width, latent_width),
                                       nn.Tanh())
        self.adapters = nn.ModuleList()
        self.b_heads = nn.ModuleList()
        for block in model.model.layers[-layers:]:
            adapter = GeneratedDownProjection(block.mlp.down_proj, rank)
            adapter.scale = rank / 2
            block.mlp.down_proj = adapter
            self.adapters.append(adapter)
            head = nn.Linear(latent_width, adapter.base.out_features * rank)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.b_heads.append(head)

    def source_factors(self, source) -> list[torch.Tensor]:
        tokenizer, records, device = source
        with torch.no_grad():
            relation = self.relation.relation(tokenizer, records, device)
        latent = self.projector(relation.reshape(1, -1))
        return [head(latent).reshape(1, adapter.base.out_features, adapter.rank)
                for head, adapter in zip(self.b_heads, self.adapters, strict=True)]

    def mount(self, factors: list[torch.Tensor] | None) -> None:
        if factors is not None and len(factors) != len(self.adapters):
            raise ValueError("One generated factor per mounted layer required")
        for layer, adapter in enumerate(self.adapters):
            adapter.b = factors[layer] if factors is not None else None


def evaluate_raw(agent, tokenizer, device: str, split: str) -> dict:
    """Same target and arm grid as the no-pretrain generic generator."""
    from ttcl.trajectory_hyperlora.slot_lora_oracle_pilot import EVAL_NUMBERS, EVAL_QUERY

    if split not in ("dev", "test"):
        raise ValueError(split)
    rows = []
    agent.eval()
    ids = {label: tokenizer(label, add_special_tokens=False).input_ids[0]
           for label in ("LEFT", "RIGHT")}
    with torch.no_grad():
        public_source = [agent.source_factors((tokenizer, source_records(
            rule, random.Random(5000 + rule), test=False), device))
            for rule in TRAIN_RULES]
        public = [torch.stack([factors[layer] for factors in public_source]).mean(0)
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
                correct_factors = agent.source_factors((tokenizer, source, device))
                wrong_factors = agent.source_factors((tokenizer, wrong, device))
                source_hash = hashlib.sha256(json.dumps(
                    source, sort_keys=True).encode()).hexdigest()
                wrong_hash = hashlib.sha256(json.dumps(
                    wrong, sort_keys=True).encode()).hexdigest()
                for cue_index, cue in enumerate(CUES):
                    for number in EVAL_NUMBERS[split]:
                        question = EVAL_QUERY[split].format(number=number, cue=cue)
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
                                        "correct": choice == ids[expected]}
                        agent.mount(None)
                        rows.append(row)
    return {"n": len(rows),
            "success": {arm: sum(row[arm]["correct"] for row in rows)
                        for arm in ("base", "public", "correct", "wrong")},
            "rows": rows}


def run(args: argparse.Namespace) -> dict:
    if (args.output.exists() or args.checkpoint.exists() or args.steps < 1 or
            args.margin < 0 or args.rank_weight < 0 or
            not 0 <= args.correction_prob <= 1):
        raise ValueError("Fresh outputs and valid training budget required")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                                   device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    action_ids = {label: tokenizer(label, add_special_tokens=False).input_ids
                  for label in ("LEFT", "RIGHT")}
    if any(len(value) != 1 for value in action_ids.values()):
        raise ValueError("Action labels must be single tokens")
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    base.config.use_cache = False
    relation_checkpoint = torch.load(args.relation_checkpoint,
                                     map_location="cpu", weights_only=True)
    agent = PretrainedRelationHyperLoRA(
        base, relation_checkpoint, args.rank, args.layers,
        args.latent_width).to(args.device)
    optimizer = torch.optim.AdamW(
        [p for p in agent.parameters() if p.requires_grad], lr=args.lr,
        weight_decay=0)
    losses = []
    for step in range(args.steps):
        rule = TRAIN_RULES[step % len(TRAIN_RULES)]
        if args.paired:
            pair_seed = rng.randrange(2**32)
            source = source_records(rule, random.Random(pair_seed), test=False)
            opposite = source_records(rule ^ 15, random.Random(pair_seed),
                                      test=False)
        else:
            source = source_records(rule, rng, test=False)
        if args.correction_prob > 0 and rng.random() < args.correction_prob:
            source = add_corrections(source, test=False)
            if args.paired:
                opposite = add_corrections(opposite, test=False)
        if args.diverse_feedback:
            for index, row in enumerate(source):
                word = rng.choice(
                    ("success", "confirmed correct", "accepted")
                    if row["feedback"] == "success" else
                    ("rejected", "invalid attempt", "incorrect"))
                row["feedback"] = word
                if args.paired:
                    opposite[index]["feedback"] = word
        if args.paired and any(
                left["observation"] != right["observation"] or
                left["feedback"] != right["feedback"]
                for left, right in zip(source, opposite, strict=True)):
            raise ValueError("Training source swap changes more than actions")
        optimizer.zero_grad(set_to_none=True)
        for cue_index, cue in enumerate(CUES):
            question = rng.choice(TRAIN_QUERY).format(number=rng.randrange(90, 120),
                                                       cue=cue)
            agent.mount(agent.source_factors((tokenizer, source, args.device)))
            logits = first_logits(agent, tokenizer, question, args.device)
            agent.mount(None)
            target = torch.tensor([action_ids[action(rule, cue_index)][0]],
                                  device=args.device)
            if args.paired:
                agent.mount(agent.source_factors((tokenizer, opposite,
                                                  args.device)))
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
    dev = evaluate_raw(agent, tokenizer, args.device, "dev")
    test = evaluate_raw(agent, tokenizer, args.device, "test")
    result = {"protocol": "Generic pretrained relation matrix from raw trajectory to generated LoRA B factors; no named semantic slots, rule IDs or step labels; relation encoder/Qwen frozen; future-query action supervision with optional same-query opposite-history paired loss on even-parity rules; odd-parity whole-rule holdout",
              "seed": args.seed, "steps": args.steps, "lr": args.lr,
              "paired_training": args.paired,
              "correction_prob": args.correction_prob,
              "diverse_feedback": args.diverse_feedback,
              "margin": args.margin if args.paired else None,
              "rank_weight": args.rank_weight if args.paired else None,
              "rank": args.rank, "layers": args.layers,
              "latent_width": args.latent_width,
              "relation_checkpoint_sha256": hashlib.sha256(
                  args.relation_checkpoint.read_bytes()).hexdigest(),
              "last_100_step_loss": sum(losses[-400:]) / min(400, len(losses)),
              "dev": dev, "test": test}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    torch.save({"trainable_state": {name: p.detach().cpu()
                                   for name, p in agent.named_parameters()
                                   if p.requires_grad},
                "relation_checkpoint_sha256": result["relation_checkpoint_sha256"],
                "rank": args.rank, "layers": args.layers,
                "latent_width": args.latent_width,
                "seed": args.seed, "steps": args.steps}, args.checkpoint)
    print(json.dumps({"dev": dev["success"], "test": test["success"]}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.35)
    parser.add_argument("--relation-checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--rank-weight", type=float, default=.3)
    parser.add_argument("--correction-prob", type=float, default=0.0)
    parser.add_argument("--diverse-feedback", action="store_true")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--latent-width", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
