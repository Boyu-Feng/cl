"""Existence test: turn successful trajectory actions into a transferable LoRA.

This deliberately does not train a hypernetwork. It asks the narrower question
whether parameter experience extracted from completed actions can transfer to
new states. Each LoRA is fitted only to its own source trajectory; future
readings, alternate-policy actions, and test feedback are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, TaskType
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.experience_pilot import (
    TEST_NUMBERS, TRAIN_EVEN, TRAIN_ODD, action, generate, parity, query,
)
from ttcl.trajectory_hyperlora.pilot import TrajectoryHyperLoRA
from ttcl.trajectory_hyperlora.two_stage_pilot import target_loss


def mount(agent: TrajectoryHyperLoRA, policy: int | None, device: str) -> None:
    coefficients = None
    if policy is not None:
        coefficients = torch.tensor([[float(index == policy) for index in (0, 1)]],
                                    device=device)
    for adapter in agent.adapters:
        adapter.coefficients = coefficients


def source_text(policy: int, numbers: list[int]) -> str:
    return "\n".join(
        f"Completed step: reading {number} ({parity(number)}); "
        f"agent pressed {action(policy, number)}; environment feedback: success."
        for number in numbers
    )


def export_expert(agent: TrajectoryHyperLoRA, policy: int, model_path: str,
                  output_dir: Path) -> None:
    """Save one trajectory-fitted expert as a standard, reloadable PEFT LoRA."""
    rank = agent.adapters[0].a.shape[1]
    first_layer = len(agent.model.model.layers) - len(agent.adapters)
    targets = []
    tensors = {}
    for offset, adapter in enumerate(agent.adapters):
        layer_index = first_layer + offset
        name = f"model.layers.{layer_index}.mlp.down_proj"
        targets.append(name)
        prefix = f"base_model.model.{name}"
        tensors[f"{prefix}.lora_A.weight"] = adapter.a[policy].detach().float().cpu().contiguous()
        tensors[f"{prefix}.lora_B.weight"] = (
            adapter.b[policy].detach().float().cpu() / rank).contiguous()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = LoraConfig(
        r=rank, lora_alpha=rank, lora_dropout=0.0,
        target_modules=targets, bias="none", task_type=TaskType.CAUSAL_LM,
        inference_mode=True, base_model_name_or_path=str(Path(model_path).resolve()),
    )
    config.save_pretrained(output_dir)
    save_file(tensors, output_dir / "adapter_model.safetensors")


def run(args: argparse.Namespace) -> dict:
    if args.steps <= 0 or args.support_per_parity < 1:
        raise ValueError("steps and support_per_parity must be positive")
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
    agent = TrajectoryHyperLoRA(model, experts=2, rank=args.rank,
                                layers=args.layers).to(args.device)
    support = {
        policy: rng.sample(TRAIN_ODD, args.support_per_parity)
        + rng.sample(TRAIN_EVEN, args.support_per_parity)
        for policy in (0, 1)
    }
    assert not any(number in TEST_NUMBERS for numbers in support.values() for number in numbers)
    optimizer = torch.optim.AdamW(agent.adapters.parameters(), lr=args.lr)
    losses = []
    for step in range(args.steps):
        policy = step % 2
        number = rng.choice(support[policy])
        mount(agent, policy, args.device)
        loss = target_loss(agent, tokenizer, policy, number, args.device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.adapters.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        mount(agent, None, args.device)
        losses.append(float(loss.detach()))
        if (step + 1) % 300 == 0:
            print(json.dumps({"step": step + 1,
                              "mean_loss": sum(losses[-300:]) / 300}), flush=True)

    rows = []
    for policy in (0, 1):
        history = source_text(policy, support[policy])
        for number in TEST_NUMBERS:
            question = query(number, train=False)
            expected = action(policy, number)
            row = {"policy": policy, "number": number, "expected": expected,
                   "source_sha256": hashlib.sha256(history.encode()).hexdigest(),
                   "question_sha256": hashlib.sha256(question.encode()).hexdigest()}
            for arm in ("base", "text", "trajectory_lora", "wrong_lora"):
                selected = policy if arm == "trajectory_lora" else (1 - policy if arm == "wrong_lora" else None)
                mount(agent, selected, args.device)
                answer = generate(agent, tokenizer, question, args.device,
                                  history=history if arm == "text" else None)
                first = answer.upper().split()[0].strip(".,:;!") if answer else ""
                row[arm] = {"answer": answer, "correct": first == expected}
            mount(agent, None, args.device)
            rows.append(row)
    arms = ("base", "text", "trajectory_lora", "wrong_lora")
    metrics = {"n": len(rows), "seed": args.seed, "steps": args.steps,
               "support_per_parity": args.support_per_parity,
               "support": {str(policy): {"numbers": numbers,
                                         "source_sha256": hashlib.sha256(
                                             source_text(policy, numbers).encode()).hexdigest()}
                           for policy, numbers in support.items()},
               "accuracy": {arm: sum(row[arm]["correct"] for row in rows) / len(rows)
                            for arm in arms},
               "rows": rows,
               "protocol": "LoRA fitted only on accepted source actions; unseen target readings"}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n")
    if args.export_adapters:
        for policy in (0, 1):
            export_expert(agent, policy, args.model,
                          output.parent / f"trajectory_adapter_{policy}")
    print(json.dumps({"accuracy": metrics["accuracy"], "output": str(output)}), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--support-per-parity", type=int, default=8)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--export-adapters", action="store_true")
    parser.add_argument("--output", default="results/trajectory_hyperlora/experience_pilot_20261004/support_seed42.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
