"""Adapt a reviewed-history hypernetwork to reward-free source prefixes.

The source is an earlier official history. Expert actions from a separate
history supervise only the later action choice. Existing checkpoints are read
only; this script writes a new result and checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.train_xland_qwen_hyperlora import QwenRawHyperLoRA
from ttcl.trajectory_hyperlora.train_xland_official_history_hyperlora import (
    evaluate, logits, prepare, source_tensor,
)
from ttcl.trajectory_hyperlora.train_xland_raw_hyperlora import checked_query


def add_reward_free_sources(raw_items: list[dict], items: list[dict],
                            device: str) -> list[dict]:
    if len(raw_items) != len(items):
        raise ValueError("Prepared source/task alignment changed")
    without_reward = []
    for raw_item, item in zip(raw_items, items):
        if raw_item["task_id"] != item["task_id"]:
            raise ValueError("Prepared source/task order changed")
        content, _ = checked_query(raw_item["queries"][0])
        episode = content["source_episodes"][0]
        steps = episode["steps"]
        if (len(steps) < 2 or steps[-1]["reward"] <= 0 or
                any(step["reward"] > 0 for step in steps[:-1])):
            raise ValueError("Expected reviewed first-success history")
        partial = {**content, "source_episodes": [{**episode,
                    "steps": steps[:-1]}]}
        source = source_tensor(partial, device, 16)
        item["reward_free_source"] = source
        without_reward.append({**item, "source": source})
    return without_reward


def run(args: argparse.Namespace) -> dict:
    if args.output.exists() or args.checkpoint.exists():
        raise FileExistsError("Use fresh v10 output paths")
    raw = args.annotations.read_bytes()
    annotations = json.loads(raw)
    original_sha = hashlib.sha256(raw).hexdigest()
    previous = torch.load(args.initial_checkpoint, map_location="cpu",
                          weights_only=True)
    if previous["annotations_sha256"] != original_sha:
        raise ValueError("Initial checkpoint/annotation content mismatch")
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                                    device=args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    choices = [tokenizer(str(i), add_special_tokens=False).input_ids
               for i in range(6)]
    if any(len(tokens) != 1 for tokens in choices):
        raise ValueError("Action digits must be single tokens")
    choice_ids = [tokens[0] for tokens in choices]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    split = {name: prepare(annotations["split"][name], tokenizer,
                           args.device, action_count=6)
             for name in ("train", "dev", "test")}
    reward_free = {name: add_reward_free_sources(annotations["split"][name],
                        split[name], args.device)
                   for name in ("train", "dev", "test")}
    train_labels = sorted({target["label"] for item in split["train"]
                           for target in item["targets"]})
    base = AutoModelForCausalLM.from_pretrained(args.model,
        torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    base.config.use_cache = False
    agent = QwenRawHyperLoRA(base, args.rank, args.layers, args.width,
        order_invariant_source=False, max_source_length=16).to(args.device)
    parameters = dict(agent.named_parameters())
    with torch.no_grad():
        for name, value in previous["trainable_state"].items():
            if name not in parameters or parameters[name].shape != value.shape:
                raise ValueError(f"Initial checkpoint architecture mismatch: {name}")
            parameters[name].copy_(value.to(parameters[name].device))
    optimizer = torch.optim.AdamW(
        (p for p in agent.parameters() if p.requires_grad), lr=args.lr)
    baseline = {name: evaluate(agent, split[name], choice_ids, pad_id,
                               args.device, args.batch_size)
                for name in ("dev", "test")}
    baseline_reward_free = {name: evaluate(agent, reward_free[name],
                choice_ids, pad_id, args.device, args.batch_size)
                for name in ("dev", "test")}
    print(json.dumps({"step": 0, "dev": baseline["dev"]["correct"],
                      "test": baseline["test"]["correct"],
                      "dev_reward_free": baseline_reward_free["dev"]["correct"]}),
          flush=True)
    best_score = (baseline["dev"]["correct"] +
                  args.reward_free_selection_weight *
                  baseline_reward_free["dev"]["correct"])
    best_step = 0
    best_state = {name: p.detach().cpu().clone()
                  for name, p in agent.named_parameters() if p.requires_grad}
    history = []
    for step in range(1, args.steps + 1):
        agent.train()
        item = rng.choice(split["train"])
        batch = rng.sample(item["targets"],
                           min(args.batch_size, len(item["targets"])))
        source = (item["reward_free_source"] if
                  rng.random() < args.reward_free_probability else
                  item["source"])
        factors = agent.compile_adapters(source)
        output = logits(agent, batch, factors, choice_ids, pad_id, args.device)
        truth = torch.tensor([target["label"] for target in batch],
                             device=args.device)
        correct_ce = F.cross_entropy(output, truth)
        loss = correct_ce
        if args.pair_weight:
            other = rng.choice([candidate for candidate in split["train"]
                                if candidate["task_id"] != item["task_id"]])
            wrong = logits(agent, batch,
                           agent.compile_adapters(other["source"]),
                           choice_ids, pad_id, args.device)
            wrong_ce = F.cross_entropy(wrong, truth)
            loss = loss + args.pair_weight * F.softplus(
                args.pair_margin + correct_ce - wrong_ce)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in agent.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        agent.mount(None)
        if step % args.eval_every == 0 or step == args.steps:
            dev = evaluate(agent, split["dev"], choice_ids, pad_id,
                           args.device, args.batch_size)
            dev_reward_free = evaluate(agent, reward_free["dev"], choice_ids,
                pad_id, args.device, args.batch_size)
            row = {"step": step, "loss": float(loss.detach()),
                   "dev": {key: dev[key] for key in
                           ("n", "correct", "wrong", "none",
                            "source_swap_changes")},
                   "dev_reward_free": {key: dev_reward_free[key] for key in
                            ("n", "correct", "wrong", "none",
                             "source_swap_changes")}}
            history.append(row)
            print(json.dumps(row), flush=True)
            score = (dev["correct"] + args.reward_free_selection_weight *
                     dev_reward_free["correct"])
            if score > best_score:
                best_score, best_step = score, step
                best_state = {name: p.detach().cpu().clone()
                              for name, p in agent.named_parameters()
                              if p.requires_grad}
    assert best_state is not None
    with torch.no_grad():
        for name, value in best_state.items():
            parameters[name].copy_(value.to(parameters[name].device))
    final = {name: evaluate(agent, split[name], choice_ids, pad_id,
                            args.device, args.batch_size)
             for name in ("dev", "test")}
    reward_free_final = {name: evaluate(agent, reward_free[name], choice_ids,
        pad_id, args.device, args.batch_size)
        for name in ("dev", "test")}
    result = {"protocol": "Continue mixed reviewed official XLand hypernetwork using reward-free prefix versus first-success source variants; six-way choice; frozen trivial dev/test",
              "annotations_sha256": original_sha,
              "initial_annotations_sha256": annotations["initial_annotations_sha256"],
              "medium_annotations_sha256": annotations["medium_annotations_sha256"],
              "initial_checkpoint_sha256": hashlib.sha256(
                  args.initial_checkpoint.read_bytes()).hexdigest(),
              "seed": args.seed, "steps": args.steps,
              "best_step": best_step, "lr": args.lr,
              "rank": args.rank, "layers": args.layers, "width": args.width,
              "action_count": 6, "max_source_length": 16,
              "train_labels_present": train_labels,
              "reward_free_probability": args.reward_free_probability,
              "reward_free_selection_weight": args.reward_free_selection_weight,
              "pair_weight": args.pair_weight,
              "pair_margin": args.pair_margin,
              "contrast_selection_weight": args.contrast_selection_weight,
              "history": history, "baseline": baseline,
              "baseline_reward_free": baseline_reward_free,
              "reward_free_final": reward_free_final, **final}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    torch.save({"trainable_state": best_state,
                "annotations_sha256": original_sha,
                "seed": args.seed}, args.checkpoint)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=.0001)
    parser.add_argument("--pair-weight", type=float, default=0.)
    parser.add_argument("--pair-margin", type=float, default=1.)
    parser.add_argument("--reward-free-probability", type=float, default=.5)
    parser.add_argument("--reward-free-selection-weight", type=float, default=1.)
    parser.add_argument("--contrast-selection-weight", type=float, default=0.)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if (args.steps < 1 or args.eval_every < 1 or args.batch_size < 1 or
            args.lr <= 0 or args.pair_weight < 0 or
            not 0 <= args.reward_free_probability <= 1 or
            args.reward_free_selection_weight < 0 or
            args.contrast_selection_weight < 0 or
            not 0 < args.gpu_fraction <= 1):
        parser.error("Invalid training budget")
    run(args)


if __name__ == "__main__":
    main()
