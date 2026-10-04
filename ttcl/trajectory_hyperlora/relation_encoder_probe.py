"""Isolate trajectory relation extraction before training generated LoRA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.direct_composition_pilot import (
    DirectRelationHyperLoRA, HELDOUT_POLICIES, TRAIN_POLICIES,
    oracle_relation_bits, records,
)
from ttcl.trajectory_hyperlora.relational_router_pilot import tokenize_records


def evaluate(agent, tokenizer, device: str) -> dict:
    agent.eval()
    rows = []
    for split, policies in (("seen", (1, 6)), ("unseen", HELDOUT_POLICIES)):
        for policy in policies:
            for source_seed in (2001, 2002):
                source, _ = records(policy,
                                    random.Random(source_seed + 31 * policy),
                                    test=True)
                fields = tokenize_records(tokenizer, source, device)
                with torch.no_grad():
                    probabilities = torch.sigmoid(
                        agent.predict_relation(fields))[0].tolist()
                target = oracle_relation_bits(source, device)[0].tolist()
                rows.append({"split": split, "policy": policy,
                             "source_seed": source_seed,
                             "target": [int(value > 0) for value in target],
                             "probabilities": probabilities,
                             "correct": [int(prob >= .5) == int(value > 0)
                                         for prob, value in zip(probabilities,
                                                                target, strict=True)]})
    return {"bit_accuracy": {split: sum(sum(row["correct"]) for row in rows
                                        if row["split"] == split) /
                             (3 * sum(row["split"] == split for row in rows))
                             for split in ("seen", "unseen")},
            "whole_policy_accuracy": {split: sum(all(row["correct"])
                                                    for row in rows if row["split"] == split) /
                                      sum(row["split"] == split for row in rows)
                                      for split in ("seen", "unseen")},
            "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.40)
    parser.add_argument("--encoder-kind", choices=("covariance", "slots",
                                                    "token_slots", "factorized"),
                        required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                                  device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    agent = DirectRelationHyperLoRA(model,
                                    encoder_kind=args.encoder_kind).to(args.device)
    agent.train()
    optimizer = torch.optim.AdamW(
        [param for name, param in agent.named_parameters()
         if param.requires_grad and (name.startswith("encoder.") or
                                     name.startswith("latent.") or
                                     name.startswith("slot_") or
                                     name.startswith("relation_head.") or
                                     name.startswith("cue_head.") or
                                     name.startswith("action_head."))],
        lr=.001)
    losses = []
    for step in range(args.steps):
        policy = TRAIN_POLICIES[step % len(TRAIN_POLICIES)]
        source, _ = records(policy, rng, test=False)
        fields = tokenize_records(tokenizer, source, args.device)
        target = oracle_relation_bits(source, args.device)
        loss = (agent.factorized_step_loss(fields, source)
                if args.encoder_kind == "factorized" else
                agent.relation_loss(fields, target))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
    result = evaluate(agent, tokenizer, args.device)
    result.update({"seed": args.seed, "steps": args.steps,
                   "encoder_kind": args.encoder_kind,
                   "last_100_loss": sum(losses[-100:]) / min(100, len(losses)),
                   "train_policies": TRAIN_POLICIES,
                   "heldout_policies": HELDOUT_POLICIES,
                   "protocol": "Synthetic relation extraction only; source-derived supervision in training; unseen whole-policy combinations and new source wording in evaluation"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"bit_accuracy": result["bit_accuracy"],
                      "whole_policy_accuracy": result["whole_policy_accuracy"],
                      "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
