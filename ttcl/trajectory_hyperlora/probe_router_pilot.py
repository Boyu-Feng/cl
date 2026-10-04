"""Trajectory-conditioned LoRA with latent action probes on a toy rule task.

Two probes are a toy-environment-specific encoder. They produce model logits,
not a textual summary, and no probe result is given to the future actor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.experience_pilot import (
    TEST_NUMBERS, TEST_SOURCE_TEMPLATES, TRAIN_SOURCE_TEMPLATES,
    action, generate, query, trajectory,
)
from ttcl.trajectory_hyperlora.pilot import TrajectoryHyperLoRA, prompt
from ttcl.trajectory_hyperlora.two_stage_pilot import oracle_evaluate, target_loss


class ProbeRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(2), nn.Linear(2, 16),
                                 nn.Tanh(), nn.Linear(16, 2))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(features), dim=-1)


def action_probe_features(agent: TrajectoryHyperLoRA, tokenizer: object,
                          source: str, device: str) -> torch.Tensor:
    """Read two generic panel states from history into numeric, detached logits."""
    agent.clear()
    token_ids = {
        answer: [tokenizer(answer, add_special_tokens=False).input_ids[0],
                 tokenizer(" " + answer, add_special_tokens=False).input_ids[0]]
        for answer in ("LEFT", "RIGHT")
    }
    gaps = []
    for number in (91, 92):  # unseen source readings, one odd and one even
        inputs = tokenizer(prompt(tokenizer, query(number, train=False), source),
                           return_tensors="pt").to(device)
        with torch.no_grad():
            logits = agent.model(**inputs).logits[0, -1].float()
            left = torch.logsumexp(logits[token_ids["LEFT"]], dim=0)
            right = torch.logsumexp(logits[token_ids["RIGHT"]], dim=0)
            gaps.append(((left - right) / 10.0).clamp(-3, 3))
    return torch.stack(gaps).unsqueeze(0).detach()


def mount(agent: TrajectoryHyperLoRA, coefficients: torch.Tensor | None) -> None:
    for adapter in agent.adapters:
        adapter.coefficients = coefficients


def evaluate(agent: TrajectoryHyperLoRA, router: ProbeRouter,
             tokenizer: object, device: str) -> dict:
    rows = []
    feature_cache = {}
    for policy in (0, 1):
        for source_seed in (2001, 2002):
            source, _, source_numbers = trajectory(
                policy, random.Random(source_seed + 17 * policy),
                TEST_SOURCE_TEMPLATES, ordered_pair=True)
            wrong_source, _, _ = trajectory(
                1 - policy, random.Random(source_seed + 17 * policy),
                TEST_SOURCE_TEMPLATES, ordered_pair=True)
            for raw in (source, wrong_source):
                if raw not in feature_cache:
                    feature_cache[raw] = action_probe_features(agent, tokenizer, raw, device)
            for number in TEST_NUMBERS:
                assert number not in source_numbers
                question = query(number, train=False)
                expected = action(policy, number)
                row = {"policy": policy, "number": number, "expected": expected,
                       "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                       "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                       "source": source, "question": question}
                for arm in ("base", "text", "hyper", "shuffled"):
                    if arm in ("hyper", "shuffled"):
                        raw = source if arm == "hyper" else wrong_source
                        mount(agent, router(feature_cache[raw]))
                    else:
                        mount(agent, None)
                    answer = generate(agent, tokenizer, question, device,
                                      history=source if arm == "text" else None)
                    first = answer.upper().split()[0].strip(".,:;!") if answer else ""
                    row[arm] = {"answer": answer, "correct": first == expected}
                mount(agent, None)
                rows.append(row)
    arms = ("base", "text", "hyper", "shuffled")
    return {"n": len(rows),
            "accuracy": {arm: sum(row[arm]["correct"] for row in rows) / len(rows)
                         for arm in arms},
            "by_policy": {str(p): {arm: sum(row[arm]["correct"] for row in rows
                                           if row["policy"] == p) / 8 for arm in arms}
                          for p in (0, 1)},
            "rows": rows}


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
    agent = TrajectoryHyperLoRA(model, experts=2, rank=4, layers=2).to(args.device)
    expert_optimizer = torch.optim.AdamW(agent.adapters.parameters(), lr=args.expert_lr)
    for step in range(args.expert_steps):
        policy = step % 2
        number = rng.randrange(1, 90)
        one_hot = torch.tensor([[float(index == policy) for index in (0, 1)]],
                               device=args.device)
        mount(agent, one_hot)
        loss = target_loss(agent, tokenizer, policy, number, args.device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.adapters.parameters(), 1.0)
        expert_optimizer.step()
        expert_optimizer.zero_grad(set_to_none=True)
        mount(agent, None)
        if (step + 1) % 200 == 0:
            print(json.dumps({"expert_step": step + 1, "loss": float(loss.detach())}), flush=True)
    oracle = oracle_evaluate(agent, tokenizer, args.device)
    print(json.dumps({"oracle_experts": f"{oracle['correct']}/{oracle['n']}"}), flush=True)
    for parameter in agent.adapters.parameters():
        parameter.requires_grad_(False)
    router = ProbeRouter().to(args.device)
    router_optimizer = torch.optim.AdamW(router.parameters(), lr=args.router_lr)
    losses = []
    for step in range(args.router_steps):
        policy = step % 2
        source, _, source_numbers = trajectory(policy, rng, TRAIN_SOURCE_TEMPLATES,
                                               ordered_pair=True)
        number = rng.choice([value for value in range(1, 90) if value not in source_numbers])
        features = action_probe_features(agent, tokenizer, source, args.device)
        mount(agent, router(features))
        loss = target_loss(agent, tokenizer, policy, number, args.device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        router_optimizer.step()
        router_optimizer.zero_grad(set_to_none=True)
        mount(agent, None)
        losses.append(float(loss.detach()))
        if (step + 1) % 100 == 0:
            print(json.dumps({"router_step": step + 1,
                              "mean_loss": sum(losses[-100:]) / 100}), flush=True)
    metrics = evaluate(agent, router, tokenizer, args.device)
    metrics.update({"oracle": oracle, "seed": args.seed,
                    "expert_steps": args.expert_steps, "router_steps": args.router_steps,
                    "router_loss_last_100": sum(losses[-100:]) / min(100, len(losses)),
                    "protocol": "Toy native-action probes encode history; future actor receives only generated LoRA"})
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
    parser.add_argument("--expert-steps", type=int, default=800)
    parser.add_argument("--router-steps", type=int, default=400)
    parser.add_argument("--expert-lr", type=float, default=0.003)
    parser.add_argument("--router-lr", type=float, default=0.001)
    parser.add_argument("--output", default="results/trajectory_hyperlora/experience_pilot_20261004/probe_seed42.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
