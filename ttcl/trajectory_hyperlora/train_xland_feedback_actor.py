"""Self-distill a generic explorer from environment-confirmed own successes.

Only training-split model-generated episodes with positive reward supply action
labels. Changed-view steps in the last 64 actions before success are retained;
no hidden rule, expert action, or developer-defined task solution is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.collect_xland_official_histories import digest
from ttcl.trajectory_hyperlora.evaluate_xland_official_live import selection_prompt
from ttcl.trajectory_hyperlora.train_xland_official_history_hyperlora import (
    QwenRawHyperLoRA, padded,
)


def examples(rows, tokenizer, choice_count, feedback_window, success_only):
    output = []
    for item in rows:
        episode = item["source"]
        if digest(episode) != item["source_sha256"]:
            raise ValueError("Source episode changed")
        if success_only and episode["positive_rewards"] < 1:
            continue
        steps = episode["steps"]
        positive = next((i for i, step in enumerate(steps)
                         if step["reward"] > 0), len(steps) - 1)
        start = max(0, positive - 63)
        for index in range(start, positive + 1):
            step = steps[index]
            if (step["action"] >= choice_count or
                    step["state"]["observation"] ==
                    step["next_state"]["observation"] and
                    step["reward"] <= 0):
                continue
            feedback = steps[max(0, index - feedback_window):index]
            prompt = selection_prompt(tokenizer, choice_count,
                step["state"]["observation"], feedback)
            output.append({"task_id": item["task_id"],
                "ids": tokenizer(prompt,
                    add_special_tokens=False).input_ids,
                "label": step["action"],
                "reward": step["reward"]})
    return output


def evaluate(agent, static_b, data, choice_ids, pad_id, device, batch_size):
    agent.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]
            tokens, mask = padded([x["ids"] for x in batch], pad_id, device)
            agent.mount([b.unsqueeze(0).expand(len(batch), -1, -1)
                         for b in static_b])
            output = agent.base(input_ids=tokens,
                attention_mask=mask, use_cache=False).logits[:, -1,
                                                              choice_ids].float()
            truth = torch.tensor([x["label"] for x in batch], device=device)
            correct += int((output.argmax(-1) == truth).sum())
    agent.mount(None)
    return {"correct": correct, "n": len(data)}


def run(args):
    if args.output.exists() or args.checkpoint.exists():
        raise FileExistsError("Fresh result/checkpoint required")
    raw = args.sources.read_bytes()
    sources = json.loads(raw)
    if sources["failures"]:
        raise ValueError("Collect failures must be resolved before training")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                               device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    choice_ids = [tokenizer(str(i), add_special_tokens=False).input_ids[0]
                  for i in range(args.choice_count)]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    data = {split: examples(sources["split"][split], tokenizer,
        args.choice_count, args.feedback_window, True)
        for split in ("train", "dev", "test")}
    if len(data["train"]) < args.batch_size:
        raise ValueError("Too few successful own-source examples")
    base = AutoModelForCausalLM.from_pretrained(args.model,
        torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    base.config.use_cache = False
    agent = QwenRawHyperLoRA(base, args.rank, args.layers, 64, False).to(
        args.device)
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    static_b = nn.ParameterList([nn.Parameter(torch.zeros(
        adapter.base.out_features, adapter.rank, device=args.device))
        for adapter in agent.adapters])
    for adapter in agent.adapters:
        adapter.a.requires_grad_(True)
    parameters = list(static_b.parameters()) + [adapter.a
        for adapter in agent.adapters]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    history = []
    best_dev, best_step, best_state = -1., 0, None
    for step in range(1, args.steps + 1):
        agent.train()
        batch = rng.sample(data["train"], k=args.batch_size)
        tokens, mask = padded([x["ids"] for x in batch], pad_id, args.device)
        agent.mount([b.unsqueeze(0).expand(len(batch), -1, -1)
                     for b in static_b])
        output = agent.base(input_ids=tokens, attention_mask=mask,
            use_cache=False).logits[:, -1, choice_ids].float()
        truth = torch.tensor([x["label"] for x in batch], device=args.device)
        loss = F.cross_entropy(output, truth)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        agent.mount(None)
        if step % args.eval_every == 0 or step == args.steps:
            dev = evaluate(agent, static_b, data["dev"], choice_ids,
                pad_id, args.device, args.batch_size)
            history.append({"step": step, "loss": float(loss.detach()),
                            "dev": dev})
            print(json.dumps(history[-1]), flush=True)
            score = dev["correct"] / dev["n"] if dev["n"] else -1.
            if best_state is None or score > best_dev:
                best_dev, best_step = score, step
                best_state = {"static_b": [b.detach().cpu().clone()
                    for b in static_b],
                    "lora_a": [a.a.detach().cpu().clone()
                    for a in agent.adapters]}
    if best_state is None:
        raise RuntimeError("No checkpoint evaluation")
    with torch.no_grad():
        for b, value in zip(static_b, best_state["static_b"], strict=True):
            b.copy_(value.to(b.device))
        for adapter, value in zip(agent.adapters, best_state["lora_a"],
                                  strict=True):
            adapter.a.copy_(value.to(adapter.a.device))
    final = {split: evaluate(agent, static_b, data[split], choice_ids,
        pad_id, args.device, args.batch_size)
        for split in ("dev", "test")}
    result = {"protocol": "Generic static LoRA explorer self-distilled only from changed-view steps in successful model-generated train episodes; task-disjoint held-out action agreement; no live reward claim",
              "sources_sha256": hashlib.sha256(raw).hexdigest(),
              "model_config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
              "seed": args.seed, "steps": args.steps, "lr": args.lr,
              "best_step": best_step,
              "rank": args.rank, "layers": args.layers,
              "feedback_window": args.feedback_window,
              "choice_count": args.choice_count,
              "batch_size": args.batch_size,
              "examples": {split: len(rows) for split, rows in data.items()},
              "history": history, **final}
    state = {"static_b": best_state["static_b"],
             "lora_a": best_state["lora_a"],
             "sources_sha256": result["sources_sha256"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    torch.save(state, args.checkpoint)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=.0001)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--choice-count", type=int, default=5)
    parser.add_argument("--feedback-window", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
