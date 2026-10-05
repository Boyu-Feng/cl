"""Replace diagnostic rule bits with a learned completed-trajectory readout.

The pretrained slot LoRA and Qwen actor are frozen. Only a small encoder of
raw observation/action strings is trained on even-parity source trajectories.
No dev/test trajectory or actor reward is used for optimization.
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

from ttcl.trajectory_hyperlora.relational_router_pilot import tokenize_records
from ttcl.trajectory_hyperlora.slot_lora_oracle_pilot import (
    CUES, TRAIN_RULES, SlotLoRA, evaluate, oracle_bits, source_records,
)


class SourceReadout(nn.Module):
    def __init__(self, embedding: nn.Embedding, width: int = 64) -> None:
        super().__init__()
        self.embedding = embedding
        size = embedding.embedding_dim
        self.observation = nn.Sequential(nn.LayerNorm(size),
                                         nn.Linear(size, width), nn.Tanh())
        self.action = nn.Sequential(nn.LayerNorm(size),
                                    nn.Linear(size, width), nn.Tanh())
        self.cue_head = nn.Linear(width, len(CUES))
        self.action_head = nn.Linear(width, 1)

    def field_mean(self, pair) -> torch.Tensor:
        ids, mask = pair
        with torch.no_grad():
            vectors = self.embedding(ids).float()
        weights = mask.unsqueeze(-1)
        return (vectors * weights).sum(1) / weights.sum(1)

    def forward(self, fields) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cue_logits = self.cue_head(
            self.observation(self.field_mean(fields["observation"])))
        action_logits = self.action_head(
            self.action(self.field_mean(fields["action"]))).squeeze(-1)
        cue_probs = cue_logits.softmax(-1)
        right_probs = action_logits.sigmoid()
        bits = (cue_probs * right_probs[:, None]).sum(0) / \
            cue_probs.sum(0).clamp_min(1e-6)
        return cue_logits, action_logits, bits


def step_targets(source: list[dict], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    cue_targets, action_targets = [], []
    for row in source:
        cues = [index for index, cue in enumerate(CUES)
                if cue in row["observation"]]
        actions = [int(label == "RIGHT") for label in ("LEFT", "RIGHT")
                   if label in row["action"]]
        if len(cues) != 1 or len(actions) != 1:
            raise ValueError("Ambiguous training source step")
        cue_targets.append(cues[0])
        action_targets.append(actions[0])
    return (torch.tensor(cue_targets, device=device),
            torch.tensor(action_targets, device=device, dtype=torch.float32))


def readout_loss(readout: SourceReadout, fields, source: list[dict],
                 device: str) -> torch.Tensor:
    cue_logits, action_logits, bits = readout(fields)
    cue_targets, action_targets = step_targets(source, device)
    rule_targets = torch.tensor(oracle_bits(source), device=device,
                                dtype=torch.float32)
    return (F.cross_entropy(cue_logits, cue_targets) +
            F.binary_cross_entropy_with_logits(action_logits, action_targets) +
            F.binary_cross_entropy(bits.clamp(1e-6, 1 - 1e-6), rule_targets))


def unique_bit_accuracy(rows: list[dict]) -> dict:
    sources = {row["source_sha256"]: (row["readout_bits"], row["oracle_bits"])
               for row in rows}
    total = sum(len(expected) for _, expected in sources.values())
    correct = sum(sum(a == b for a, b in zip(actual, expected, strict=True))
                  for actual, expected in sources.values())
    return {"correct": correct, "total": total,
            "exact_sources": sum(actual == expected for actual, expected
                                 in sources.values()),
            "sources": len(sources)}


def run(args: argparse.Namespace) -> dict:
    if args.output.exists() or args.checkpoint_output.exists() or args.steps < 1:
        raise ValueError("Fresh outputs and positive step budget required")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                                   device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    model.config.use_cache = False
    saved = torch.load(args.slot_checkpoint, map_location="cpu", weights_only=True)
    agent = SlotLoRA(model, saved["rank_per_cue"], saved["layers"]).to(args.device)
    parameters = dict(agent.named_parameters())
    with torch.no_grad():
        for name, value in saved["trainable_state"].items():
            if name not in parameters or parameters[name].shape != value.shape:
                raise ValueError(f"Slot checkpoint mismatch: {name}")
            parameters[name].copy_(value.to(args.device))
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    readout = SourceReadout(model.get_input_embeddings()).to(args.device)
    optimizer = torch.optim.AdamW(
        [p for p in readout.parameters() if p.requires_grad], lr=args.lr,
        weight_decay=0)
    losses = []
    for step in range(args.steps):
        rule = TRAIN_RULES[step % len(TRAIN_RULES)]
        source = source_records(rule, rng, test=False)
        fields = tokenize_records(tokenizer, source, args.device)
        loss = readout_loss(readout, fields, source, args.device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in readout.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 100 == 0:
            print(json.dumps({"step": step + 1,
                              "mean_loss": sum(losses[-100:]) / 100}), flush=True)
    readout.eval()

    def relation_reader(source: list[dict]) -> tuple[int, ...]:
        fields = tokenize_records(tokenizer, source, args.device)
        return tuple(int(value >= .5) for value in readout(fields)[2].tolist())

    dev = evaluate(agent, tokenizer, args.device, "dev", relation_reader)
    test = evaluate(agent, tokenizer, args.device, "test", relation_reader)
    result = {"protocol": "Four-cue raw observation/action trajectory readout, trained on even-parity source rules; Qwen actor and slot LoRA frozen; heldout whole rules evaluated with same predeclared source/query split; no environment RL",
              "seed": args.seed, "steps": args.steps, "lr": args.lr,
              "slot_checkpoint_sha256": hashlib.sha256(
                  args.slot_checkpoint.read_bytes()).hexdigest(),
              "last_100_loss": sum(losses[-100:]) / min(100, len(losses)),
              "dev": {**dev, "readout": unique_bit_accuracy(dev["rows"])},
              "test": {**test, "readout": unique_bit_accuracy(test["rows"])}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    torch.save({"readout_state": {name: parameter.detach().cpu()
                                 for name, parameter in readout.named_parameters()
                                 if parameter.requires_grad},
                "slot_checkpoint_sha256": result["slot_checkpoint_sha256"]},
               args.checkpoint_output)
    print(json.dumps({"dev": dev["success"], "test": test["success"],
                      "dev_readout": result["dev"]["readout"],
                      "test_readout": result["test"]["readout"]}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.35)
    parser.add_argument("--slot-checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
