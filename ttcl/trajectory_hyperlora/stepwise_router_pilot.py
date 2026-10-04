"""Amortize trajectory-fitted LoRAs with a step-preserving raw-text encoder.

This is a bounded synthetic pilot. Two frozen LoRA basis elements were fitted
from successful source steps in the earlier experiment. A shared encoder reads
new raw trajectory steps, preserving their within-step token order, and emits
mixing weights. It never receives a policy ID or a written rule. It is not yet
an unrestricted hypernetwork that creates a novel adapter for any trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
from torch import nn
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.experience_pilot import (
    TEST_NUMBERS, TEST_SOURCE_TEMPLATES, TRAIN_EVEN, TRAIN_ODD,
    TRAIN_SOURCE_TEMPLATES, action, generate, parity, query,
)
from ttcl.trajectory_hyperlora.pilot import TrajectoryHyperLoRA
from ttcl.trajectory_hyperlora.two_stage_pilot import target_loss


class StepwiseEncoder(nn.Module):
    """Encode complete lines before pooling; no environment action ontology."""

    def __init__(self, embedding: nn.Embedding, experts: int = 2,
                 hidden: int = 96) -> None:
        super().__init__()
        self.embedding = embedding
        self.token_gru = nn.GRU(embedding.embedding_dim, hidden, batch_first=True,
                                bidirectional=True)
        self.step_gru = nn.GRU(hidden * 2, hidden, batch_first=True,
                               bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden * 2),
                                  nn.Linear(hidden * 2, experts))

    def forward(self, token_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or token_mask.shape != token_ids.shape:
            raise ValueError("Expected [steps, tokens] IDs and matching mask")
        lengths = token_mask.sum(-1)
        if not len(token_ids) or torch.any(lengths == 0):
            raise ValueError("Each source step needs nonempty raw text")
        with torch.no_grad():
            embeddings = self.embedding(token_ids).float()
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, last = self.token_gru(packed)
        step_vectors = torch.cat((last[-2], last[-1]), dim=-1).unsqueeze(0)
        _, episode_last = self.step_gru(step_vectors)
        episode = torch.cat((episode_last[-2], episode_last[-1]), dim=-1)
        return torch.softmax(self.head(episode), dim=-1)


def encode_steps(tokenizer: object, lines: list[str], device: str,
                 max_tokens: int = 80) -> tuple[torch.Tensor, torch.Tensor]:
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("A source must contain nonempty completed steps")
    encoded = tokenizer(lines, add_special_tokens=False, padding=True,
                        truncation=True, max_length=max_tokens,
                        return_tensors="pt")
    return encoded.input_ids.to(device), encoded.attention_mask.to(device)


def make_source(policy: int, rng: random.Random, templates: tuple[str, ...],
                support_per_parity: int) -> tuple[list[str], tuple[int, ...]]:
    numbers = rng.sample(TRAIN_ODD, support_per_parity) + rng.sample(
        TRAIN_EVEN, support_per_parity)
    rng.shuffle(numbers)
    lines = [rng.choice(templates).format(number=n, parity=parity(n),
                                          action=action(policy, n)) for n in numbers]
    return lines, tuple(numbers)


def load_expert_bank(agent: TrajectoryHyperLoRA, bank: Path) -> None:
    first_layer = len(agent.model.model.layers) - len(agent.adapters)
    for policy in (0, 1):
        tensors = load_file(str(bank / f"trajectory_adapter_{policy}" /
                                "adapter_model.safetensors"))
        for offset, adapter in enumerate(agent.adapters):
            prefix = (f"base_model.model.model.layers.{first_layer + offset}"
                      ".mlp.down_proj")
            a = tensors[f"{prefix}.lora_A.weight"]
            b = tensors[f"{prefix}.lora_B.weight"]
            rank = adapter.a.shape[1]
            if a.shape != adapter.a[policy].shape or b.shape != adapter.b[policy].shape:
                raise ValueError("Saved expert LoRA does not match the pilot architecture")
            with torch.no_grad():
                adapter.a[policy].copy_(a.to(adapter.a.device))
                adapter.b[policy].copy_((b * rank).to(adapter.b.device))
    for parameter in agent.adapters.parameters():
        parameter.requires_grad_(False)


def mount(agent: TrajectoryHyperLoRA, coefficients: torch.Tensor | None) -> None:
    for adapter in agent.adapters:
        adapter.coefficients = coefficients


def evaluate(agent: TrajectoryHyperLoRA, encoder: StepwiseEncoder,
             tokenizer: object, device: str) -> dict:
    agent.eval()
    encoder.eval()
    rows = []
    for policy in (0, 1):
        for source_seed in (2001, 2002):
            lines, support = make_source(policy, random.Random(source_seed + 17 * policy),
                                         TEST_SOURCE_TEMPLATES, 8)
            wrong_lines, _ = make_source(1 - policy,
                                         random.Random(source_seed + 17 * policy),
                                         TEST_SOURCE_TEMPLATES, 8)
            source = "\n".join(lines)
            with torch.no_grad():
                ids, mask = encode_steps(tokenizer, lines, device)
                coeff = encoder(ids, mask)
                wrong_ids, wrong_mask = encode_steps(tokenizer, wrong_lines, device)
                wrong_coeff = encoder(wrong_ids, wrong_mask)
            for number in TEST_NUMBERS:
                assert number not in support
                question = query(number, train=False)
                expected = action(policy, number)
                row = {"policy": policy, "source_seed": source_seed,
                       "number": number, "expected": expected,
                       "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                       "query_sha256": hashlib.sha256(question.encode()).hexdigest(),
                       "coefficients": coeff.detach().float().cpu().tolist()[0]}
                for arm in ("base", "text", "generated", "wrong_source", "expert"):
                    selected = (coeff if arm == "generated" else wrong_coeff
                                if arm == "wrong_source" else
                                torch.tensor([[float(i == policy) for i in (0, 1)]],
                                             device=device) if arm == "expert" else None)
                    mount(agent, selected)
                    answer = generate(agent, tokenizer, question, device,
                                      history=source if arm == "text" else None)
                    first = answer.upper().split()[0].strip(".,:;!") if answer else ""
                    row[arm] = {"answer": answer, "correct": first == expected}
                mount(agent, None)
                rows.append(row)
    arms = ("base", "text", "generated", "wrong_source", "expert")
    return {"n": len(rows),
            "accuracy": {arm: sum(row[arm]["correct"] for row in rows) / len(rows)
                         for arm in arms},
            "by_policy": {str(policy): {arm: sum(row[arm]["correct"] for row in rows
                                               if row["policy"] == policy) / 8
                                        for arm in arms} for policy in (0, 1)},
            "rows": rows}


def run(args: argparse.Namespace) -> dict:
    if args.steps < 1 or not 1 <= args.support_per_parity <= len(TRAIN_EVEN):
        raise ValueError("Invalid step or support count")
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
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 50
    agent = TrajectoryHyperLoRA(model, experts=2, rank=4, layers=2).to(args.device)
    load_expert_bank(agent, args.bank)
    encoder = StepwiseEncoder(agent.model.get_input_embeddings()).to(args.device)
    optimizer = torch.optim.AdamW((p for p in encoder.parameters() if p.requires_grad),
                                  lr=args.lr)
    losses = []
    for step in range(args.steps):
        policy = step % 2
        lines, support = make_source(policy, rng, TRAIN_SOURCE_TEMPLATES,
                                     args.support_per_parity)
        ids, mask = encode_steps(tokenizer, lines, args.device)
        coefficients = encoder(ids, mask)
        mount(agent, coefficients)
        candidate_odd = rng.choice([n for n in TRAIN_ODD if n not in support])
        candidate_even = rng.choice([n for n in TRAIN_EVEN if n not in support])
        loss = (target_loss(agent, tokenizer, policy, candidate_odd, args.device)
                + target_loss(agent, tokenizer, policy, candidate_even, args.device)) / 2
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in encoder.parameters()
                                        if p.requires_grad), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        mount(agent, None)
        losses.append(float(loss.detach()))
        if (step + 1) % 50 == 0:
            print(json.dumps({"step": step + 1,
                              "mean_loss": sum(losses[-50:]) / 50}), flush=True)
    result = evaluate(agent, encoder, tokenizer, args.device)
    result.update({"seed": args.seed, "steps": args.steps,
                   "support_per_parity": args.support_per_parity,
                   "last_50_loss": sum(losses[-50:]) / min(50, len(losses)),
                   "protocol": "Synthetic raw-step trajectory to frozen LoRA-basis mixture; unseen numbers and source wording; no RL"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"accuracy": result["accuracy"],
                      "by_policy": result["by_policy"],
                      "output": str(args.output)}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--bank", type=Path, default=Path(
        "results/trajectory_hyperlora/experience_pilot_20261004"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--support-per-parity", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/stepwise_router_20261005/seed42.json"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
