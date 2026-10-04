"""Stabilized experience pilot: learn LoRA skill basis, then trajectory routing.

Policy IDs are used only while learning the two basis adapters. The router
receives raw completed trajectories and downstream action loss, never IDs.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.experience_pilot import (
    TEST_NUMBERS, TRAIN_SOURCE_TEMPLATES, action, evaluate, generate, query, trajectory,
)
from ttcl.trajectory_hyperlora.pilot import TrajectoryHyperLoRA, prompt


def target_loss(agent: TrajectoryHyperLoRA, tokenizer: object,
                policy: int, number: int, device: str) -> torch.Tensor:
    prefix = tokenizer(prompt(tokenizer, query(number, train=True)),
                       add_special_tokens=False).input_ids
    suffix = tokenizer(" " + action(policy, number) + tokenizer.eos_token,
                       add_special_tokens=False).input_ids
    ids = torch.tensor([prefix + suffix], device=device)
    labels = torch.tensor([[-100] * len(prefix) + suffix], device=device)
    return agent.model(input_ids=ids, labels=labels, use_cache=False).loss


def oracle_evaluate(agent: TrajectoryHyperLoRA, tokenizer: object, device: str) -> dict:
    rows = []
    for policy in (0, 1):
        coefficients = torch.tensor([[float(index == policy) for index in (0, 1)]],
                                    device=device)
        for adapter in agent.adapters:
            adapter.coefficients = coefficients
        for number in TEST_NUMBERS:
            answer = generate(agent, tokenizer, query(number, train=False), device)
            expected = action(policy, number)
            rows.append({"policy": policy, "number": number, "expected": expected,
                         "answer": answer, "correct": answer.upper().split()[0] == expected
                         if answer else False})
    agent.clear()
    return {"correct": sum(row["correct"] for row in rows), "n": len(rows), "rows": rows}


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa"
    ).to(args.device)
    model.config.use_cache = False
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 50
    agent = TrajectoryHyperLoRA(model, experts=2, rank=args.rank, layers=args.layers,
                                encoder_kind=args.encoder).to(args.device)

    expert_optimizer = torch.optim.AdamW(agent.adapters.parameters(), lr=args.expert_lr)
    for step in range(args.expert_steps):
        policy = step % 2
        number = rng.randrange(1, 90)
        one_hot = torch.tensor([[float(index == policy) for index in (0, 1)]],
                               device=args.device)
        for adapter in agent.adapters:
            adapter.coefficients = one_hot
        loss = target_loss(agent, tokenizer, policy, number, args.device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.adapters.parameters(), 1.0)
        expert_optimizer.step()
        expert_optimizer.zero_grad(set_to_none=True)
        agent.clear()
        if (step + 1) % 200 == 0:
            print(json.dumps({"expert_step": step + 1, "loss": float(loss.detach())}), flush=True)
    oracle = oracle_evaluate(agent, tokenizer, args.device)
    print(json.dumps({"oracle_experts": f"{oracle['correct']}/{oracle['n']}"}), flush=True)

    for parameter in agent.adapters.parameters():
        parameter.requires_grad_(False)
    router_parameters = list(agent.encoder.parameters()) + list(agent.projection.parameters())
    if hasattr(agent, "token_attention"):
        router_parameters += list(agent.token_attention.parameters())
    router_optimizer = torch.optim.AdamW(router_parameters, lr=args.router_lr)
    router_losses = []
    for step in range(args.router_steps):
        policy = step % 2
        source, _, source_numbers = trajectory(policy, rng, TRAIN_SOURCE_TEMPLATES,
                                               ordered_pair=args.ordered_pair)
        number = rng.choice([value for value in range(1, 90) if value not in source_numbers])
        agent.set_source(tokenizer(source, return_tensors="pt").input_ids.to(args.device))
        loss = target_loss(agent, tokenizer, policy, number, args.device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router_parameters, 1.0)
        router_optimizer.step()
        router_optimizer.zero_grad(set_to_none=True)
        agent.clear()
        router_losses.append(float(loss.detach()))
        if (step + 1) % 100 == 0:
            print(json.dumps({"router_step": step + 1,
                              "mean_loss": sum(router_losses[-100:]) / 100}), flush=True)
    metrics = evaluate(agent, tokenizer, args.device, ordered_pair=args.ordered_pair)
    metrics.update({"oracle": oracle, "seed": args.seed, "encoder": args.encoder,
                    "expert_steps": args.expert_steps, "router_steps": args.router_steps,
                    "ordered_pair": args.ordered_pair,
                    "expert_lr": args.expert_lr, "router_lr": args.router_lr,
                    "router_loss_last_100": sum(router_losses[-100:]) / min(100, len(router_losses)),
                    "protocol": "Synthetic parity-to-lever transfer; policy IDs used only for expert pretraining"})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"accuracy": metrics["accuracy"], "output": str(output)}), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder", choices=("attention", "frozen_lm"), default="frozen_lm")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ordered-pair", action="store_true")
    parser.add_argument("--expert-steps", type=int, default=800)
    parser.add_argument("--router-steps", type=int, default=600)
    parser.add_argument("--expert-lr", type=float, default=0.003)
    parser.add_argument("--router-lr", type=float, default=0.0003)
    parser.add_argument("--output", default="results/trajectory_hyperlora/experience_pilot_20261004/two_stage_seed42.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
