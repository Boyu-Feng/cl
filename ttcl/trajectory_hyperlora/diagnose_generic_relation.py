"""Read-only probe of relation information in a generic trajectory latent."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.general_relation_hyperlora import GenericRelationHyperLoRA
from ttcl.trajectory_hyperlora.relational_router_pilot import tokenize_records
from ttcl.trajectory_hyperlora.slot_lora_oracle_pilot import (
    DEV_RULES, TEST_RULES, TRAIN_RULES, source_records,
)


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                                   device=args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    agent = GenericRelationHyperLoRA(
        model, checkpoint["rank"], checkpoint["layers"],
        checkpoint["width"]).to(args.device)
    params = dict(agent.named_parameters())
    with torch.no_grad():
        for name, saved in checkpoint["trainable_state"].items():
            if name not in params or params[name].shape != saved.shape:
                raise ValueError(f"Checkpoint mismatch: {name}")
            params[name].copy_(saved.to(args.device))
    agent.eval()

    def encode(rule: int, seed: int, test: bool) -> torch.Tensor:
        source = source_records(rule, random.Random(seed), test=test)
        fields = tokenize_records(tokenizer, source, args.device)
        with torch.no_grad():
            return agent.encoder(fields).squeeze(0).float().cpu()

    train = [(rule, encode(rule, 50000 + 101 * rule + seed, False))
             for rule in TRAIN_RULES for seed in range(8)]
    valid = [(rule, encode(rule, 60000 + 101 * rule + seed, True))
             for rule in (*DEV_RULES, *TEST_RULES) for seed in range(2)]
    x = torch.stack([vector for _, vector in train])
    y = torch.tensor([[(rule >> bit) & 1 for bit in range(4)]
                      for rule, _ in train], dtype=torch.float32)
    center = x.mean(0)
    scale = x.std(0).clamp_min(1e-4)
    standardized = (x - center) / scale
    ones = torch.ones((len(train), 1))
    design = torch.cat([standardized, ones], dim=-1)
    ridge = torch.eye(design.shape[1]) * args.ridge
    ridge[-1, -1] = 0
    weights = torch.linalg.solve(design.T @ design + ridge,
                                 design.T @ (y * 2 - 1))

    def score(rows: list[tuple[int, torch.Tensor]]) -> dict:
        z = torch.stack([vector for _, vector in rows])
        design_z = torch.cat([(z - center) / scale,
                              torch.ones((len(rows), 1))], dim=-1)
        predicted = ((design_z @ weights) >= 0).int()
        expected = torch.tensor([[(rule >> bit) & 1 for bit in range(4)]
                                 for rule, _ in rows])
        return {"bits": int((predicted == expected).sum()),
                "total_bits": int(expected.numel()),
                "exact_rules": int((predicted == expected).all(-1).sum()),
                "sources": len(rows)}

    result = {"protocol": "Read-only ridge probe on frozen generic trajectory latent; probe fitted only to even-parity train-rule source latents; whole odd-parity rules and source wording held out",
              "checkpoint_sha256": hashlib.sha256(
                  args.checkpoint.read_bytes()).hexdigest(),
              "ridge": args.ridge,
              "train": score(train), "heldout": score(valid),
              "latent_coordinate_std_mean": float(x.std(0).mean()),
              "latent_coordinate_std_max": float(x.std(0).max())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.35)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=.1)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
