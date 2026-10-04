"""Test whether a generated LoRA transfers a learned action rule to new states.

The environment has two hidden policies. A completed trajectory reveals which
lever was accepted for odd/even readings; a later query uses a never-seen
reading. The policy name and rule are never supplied to the actor or encoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.pilot import TrajectoryHyperLoRA, prompt


TRAIN_SOURCE_TEMPLATES = (
    "Observation: panel shows {number} ({parity}). Agent action: {action}. Environment feedback: accepted.",
    "Step: reading={number}; indicator={parity}; command={action}; result=success.",
)
TEST_SOURCE_TEMPLATES = (
    "Past episode: at reading {number}, marked {parity}, the agent pressed {action}; the panel confirmed success.",
    "Completed trial: sensor {number} reported {parity}. Selected lever {action}. Feedback: correct.",
)
TRAIN_QUERY = (
    "Panel reading {number} is {parity}. Which lever should the agent press "
    "under this panel's convention? Reply with exactly LEFT or RIGHT."
)
TEST_QUERY = (
    "New panel reading: {number} ({parity}). Choose the correct lever. "
    "Reply with only LEFT or RIGHT."
)
TRAIN_ODD = tuple(range(1, 90, 2))
TRAIN_EVEN = tuple(range(2, 90, 2))
TEST_NUMBERS = (101, 102, 113, 114)


def parity(number: int) -> str:
    return "ODD" if number % 2 else "EVEN"


def action(policy: int, number: int) -> str:
    """Private synthetic environment label; never put the policy in model input."""
    left = (number % 2 == 0) == (policy == 0)
    return "LEFT" if left else "RIGHT"


def trajectory(policy: int, rng: random.Random,
               templates: tuple[str, ...],
               ordered_pair: bool = False) -> tuple[str, str, tuple[int, ...]]:
    if ordered_pair:
        numbers = [rng.choice(TRAIN_ODD), rng.choice(TRAIN_EVEN)]
    else:
        numbers = rng.sample(TRAIN_ODD, 2) + rng.sample(TRAIN_EVEN, 2)
        rng.shuffle(numbers)
    lines = [rng.choice(templates).format(number=n, parity=parity(n), action=action(policy, n))
             for n in numbers]
    source = "\n".join(lines)
    return source, action(policy, numbers[-1]), tuple(numbers)


def query(number: int, *, train: bool) -> str:
    template = TRAIN_QUERY if train else TEST_QUERY
    return template.format(number=number, parity=parity(number))


def generate(agent: TrajectoryHyperLoRA, tokenizer: object, question: str,
             device: str, history: str | None = None) -> str:
    inputs = tokenizer(prompt(tokenizer, question, history), return_tensors="pt").to(device)
    with torch.no_grad():
        output = agent.model.generate(**inputs, do_sample=False, max_new_tokens=5,
                                      pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def evaluate(agent: TrajectoryHyperLoRA, tokenizer: object, device: str,
             ordered_pair: bool = False) -> dict:
    agent.eval()
    rows = []
    for policy in (0, 1):
        for source_seed in (2001, 2002):
            source, _, source_numbers = trajectory(
                policy, random.Random(source_seed + 17 * policy), TEST_SOURCE_TEMPLATES,
                ordered_pair=ordered_pair)
            wrong_source, _, _ = trajectory(
                1 - policy, random.Random(source_seed + 17 * policy), TEST_SOURCE_TEMPLATES,
                ordered_pair=ordered_pair)
            for number in TEST_NUMBERS:
                assert number not in source_numbers
                question = query(number, train=False)
                expected = action(policy, number)
                row = {"source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                       "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                       "policy": policy, "number": number, "expected": expected,
                       "source": source, "question": question}
                for arm in ("base", "text", "hyper", "shuffled"):
                    memory = source if arm == "hyper" else wrong_source if arm == "shuffled" else None
                    agent.set_source(tokenizer(memory, return_tensors="pt").input_ids.to(device)
                                     if memory is not None else None)
                    answer = generate(agent, tokenizer, question, device,
                                      history=source if arm == "text" else None)
                    first = answer.upper().split()[0].strip(".,:;!") if answer else ""
                    row[arm] = {"answer": answer, "correct": first == expected}
                agent.clear()
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
    if args.steps <= 0 or args.warmup_steps < 0:
        raise ValueError("steps must be positive and warmup_steps nonnegative")
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
    agent = TrajectoryHyperLoRA(model, experts=args.experts, rank=args.rank,
                                layers=args.layers, encoder_kind=args.encoder,
                                pretrain_tokens=args.warmup_steps > 0).to(args.device)
    optimizer = torch.optim.AdamW((p for p in agent.parameters() if p.requires_grad), lr=args.lr)
    warmup_losses = []
    for step in range(args.warmup_steps):
        policy = step % 2
        source, last_action, _ = trajectory(policy, rng, TRAIN_SOURCE_TEMPLATES,
                                            ordered_pair=args.ordered_pair)
        agent.set_source(tokenizer(source, return_tensors="pt").input_ids.to(args.device))
        token_id = tokenizer(" " + last_action, add_special_tokens=False).input_ids[0]
        loss = agent.reconstruction_loss(token_id)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in agent.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        agent.clear()
        warmup_losses.append(float(loss.detach()))
        if (step + 1) % 50 == 0:
            print(json.dumps({"warmup_step": step + 1,
                              "mean_loss": sum(warmup_losses[-50:]) / 50}), flush=True)
    losses = []
    for step in range(args.steps):
        policy = step % 2
        source, _, source_numbers = trajectory(policy, rng, TRAIN_SOURCE_TEMPLATES,
                                               ordered_pair=args.ordered_pair)
        number = rng.choice([value for value in range(1, 90) if value not in source_numbers])
        expected = action(policy, number)
        agent.set_source(tokenizer(source, return_tensors="pt").input_ids.to(args.device))
        prefix = tokenizer(prompt(tokenizer, query(number, train=True)),
                           add_special_tokens=False).input_ids
        suffix = tokenizer(" " + expected + tokenizer.eos_token,
                           add_special_tokens=False).input_ids
        inputs = torch.tensor([prefix + suffix], device=args.device)
        labels = torch.tensor([[-100] * len(prefix) + suffix], device=args.device)
        loss = agent.model(input_ids=inputs, labels=labels, use_cache=False).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in agent.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        agent.clear()
        losses.append(float(loss.detach()))
        if (step + 1) % 50 == 0:
            print(json.dumps({"step": step + 1, "mean_loss": sum(losses[-50:]) / 50}),
                  flush=True)
    metrics = evaluate(agent, tokenizer, args.device, ordered_pair=args.ordered_pair)
    metrics.update({"seed": args.seed, "steps": args.steps, "encoder": args.encoder,
                    "warmup_steps": args.warmup_steps, "rank": args.rank,
                    "experts": args.experts, "layers": args.layers,
                    "loss_last_50": sum(losses[-50:]) / min(50, len(losses)),
                    "ordered_pair": args.ordered_pair,
                    "protocol": "Synthetic hidden parity-to-lever rule; new readings and held-out wording"})
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
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ordered-pair", action="store_true")
    parser.add_argument("--encoder", choices=("attention", "frozen_lm"),
                        default="attention")
    parser.add_argument("--output", default="results/trajectory_hyperlora/experience_pilot_20261004/seed42.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
