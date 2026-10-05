"""Real XLand-100B history -> LoRA -> expert action on another history.

This is an offline, task-held-out expert-action diagnostic. It does not claim
official environment success or reuse future/expert actions as source input.
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
from ttcl.trajectory_hyperlora.train_xland_raw_hyperlora import (
    checked_query, encode_query,
)


ACTIONS = tuple(range(5))


def source_tensor(content: dict, device: str,
                  max_source_length: int = 16) -> dict[str, torch.Tensor]:
    before, after, goals, actions, rewards, dones, state, goal, \
        episodes, positions = encode_query(content, max_source_length)
    n = len(before)
    return {key: torch.tensor(value, device=device).unsqueeze(0) for key, value in {
        "before": before, "after": after, "goals": goals,
        "actions": actions, "rewards": rewards, "dones": dones,
        "episode_index": episodes, "step_index": positions,
        "mask": [True] * n,
        "query_state": state, "query_goal": goal}.items()}


def question(content: dict, action_count: int = 5) -> str:
    state = content["target_initial_state"]
    rows = [" ".join(f"({tile[0]},{tile[1]})" for tile in row)
            for row in state["observation"]]
    choice_text = ("0=forward, 1=turn right, 2=turn left, 3=pick up, "
                   "4=put down" + (", 5=toggle" if action_count == 6 else "") + ". ")
    return ("XLand-MiniGrid action prediction. The hidden task is conveyed "
            "by learned parameters. From the current local 5x5 observation, "
            "select the expert's next action. " + choice_text +
            "Reply with one digit only.\nObservation:\n" +
            "\n".join(rows) + "\nAction:")


def prompt_ids(tokenizer, content: dict, action_count: int = 5) -> list[int]:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question(content, action_count)}], tokenize=False,
        add_generation_prompt=True)
    return tokenizer(text, add_special_tokens=False).input_ids


def prepare(items: list[dict], tokenizer, device: str,
            prefix_augmentation: bool = False,
            action_count: int = 5,
            max_source_length: int = 16) -> list[dict]:
    prepared = []
    for item in items:
        targets = []
        for query in item["queries"]:
            content, label = checked_query(query)
            if label not in range(action_count):
                raise ValueError("Official expert action outside pilot choices")
            targets.append({"ids": prompt_ids(tokenizer, content, action_count), "label": label})
        if not targets:
            raise ValueError("Empty task")
        content, _ = checked_query(item["queries"][0])
        source_variants = [source_tensor(content, device,
                                         max_source_length)]
        if prefix_augmentation:
            steps = content["source_episodes"][0]["steps"]
            for length in (1, 2, 4, 8):
                if length >= len(steps):
                    continue
                partial = {**content, "source_episodes": [{
                    **content["source_episodes"][0],
                    "steps": steps[:length]}]}
                source_variants.append(source_tensor(partial, device,
                                                      max_source_length))
        prepared.append({"task_id": item["task_id"],
                         "ruleset_id": item["ruleset_id"],
                         "source": source_variants[0],
                         "source_variants": source_variants,
                         "targets": targets})
    return prepared


def padded(ids: list[list[int]], pad_id: int,
           device: str) -> tuple[torch.Tensor, torch.Tensor]:
    length = max(map(len, ids))
    tokens = torch.full((len(ids), length), pad_id, dtype=torch.long,
                        device=device)
    mask = torch.zeros_like(tokens)
    for i, row in enumerate(ids):
        tokens[i, -len(row):] = torch.tensor(row, device=device)
        mask[i, -len(row):] = 1
    return tokens, mask


def logits(agent: QwenRawHyperLoRA, targets: list[dict],
           factors: list[torch.Tensor] | None,
           choice_ids: list[int], pad_id: int, device: str) -> torch.Tensor:
    tokens, mask = padded([target["ids"] for target in targets], pad_id, device)
    agent.mount(None if factors is None else
                [factor.expand(len(targets), -1, -1) for factor in factors])
    return agent.base(input_ids=tokens, attention_mask=mask,
                      use_cache=False).logits[:, -1, choice_ids].float()


def evaluate(agent: QwenRawHyperLoRA, items: list[dict],
             choice_ids: list[int], pad_id: int, device: str,
             batch_size: int,
             shared_source: dict[str, torch.Tensor] | None = None) -> dict:
    agent.eval()
    counts = {kind: 0 for kind in ("correct", "wrong", "none")}
    changes = 0
    confusion = {kind: {str(i): [0] * len(choice_ids)
                        for i in range(len(choice_ids))}
                 for kind in counts}
    with torch.no_grad():
        for item_index, item in enumerate(items):
            other = items[(item_index + 1) % len(items)]
            if other["task_id"] == item["task_id"]:
                raise ValueError("Wrong source must be another held-out task")
            factors = agent.compile_adapters(shared_source or item["source"])
            wrong_factors = agent.compile_adapters(
                shared_source or other["source"])
            targets = item["targets"]
            for start in range(0, len(targets), batch_size):
                batch = targets[start:start + batch_size]
                truth = torch.tensor([x["label"] for x in batch], device=device)
                output = {}
                for kind, adapter in (("correct", factors),
                                      ("wrong", wrong_factors),
                                      ("none", None)):
                    output[kind] = logits(agent, batch, adapter, choice_ids,
                                          pad_id, device).argmax(-1)
                    counts[kind] += int((output[kind] == truth).sum())
                    for actual, prediction in zip(truth.tolist(),
                                                  output[kind].tolist()):
                        confusion[kind][str(actual)][prediction] += 1
                changes += int((output["correct"] != output["wrong"]).sum())
    agent.mount(None)
    return {"n": sum(len(x["targets"]) for x in items),
            **counts, "source_swap_changes": changes,
            "confusion": confusion}


def run(args: argparse.Namespace) -> dict:
    if args.output.exists() or args.checkpoint.exists():
        raise FileExistsError("Fresh result paths required")
    raw = args.annotations.read_bytes()
    annotations = json.loads(raw)
    action_count = annotations.get("budget", {}).get("action_count", 5)
    if action_count not in (5, 6):
        raise ValueError("Unsupported official action count")
    max_source_length = annotations.get("source_limit",
        annotations.get("budget", {}).get("source_length", 16))
    if max_source_length not in (16, 64):
        raise ValueError("Unsupported source window length")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                                   device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    choice_tokens = [tokenizer(str(i), add_special_tokens=False).input_ids
                     for i in range(action_count)]
    if any(len(x) != 1 for x in choice_tokens):
        raise ValueError("Action labels are not single tokens")
    choice_ids = [x[0] for x in choice_tokens]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    items = {name: annotations["split"][name]
             for name in ("train", "dev", "test")}
    if args.positive_only_train:
        if not all("online_source_positive" in item
                   for group in items.values() for item in group):
            raise ValueError("Positive-only training requires reviewed online sources")
        items["train"] = [item for item in items["train"]
                          if item["online_source_positive"] > 0]
    split = {name: prepare(items[name], tokenizer,
                           args.device, args.prefix_augmentation, action_count,
                           max_source_length)
             for name in ("train", "dev", "test")}
    if not split["train"] or (args.pair_weight and len(split["train"]) < 2):
        raise ValueError("Too few train tasks after source filtering")
    positive_dev_ids = {item["task_id"] for item in items["dev"]
        if item.get("online_source_positive", 0) > 0}
    if args.select_positive_dev and len(positive_dev_ids) < 2:
        raise ValueError("Need two positive online sources on dev")
    shared_source = split["train"][0]["source"] if args.shared_source else None
    base = AutoModelForCausalLM.from_pretrained(args.model,
        torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    base.config.use_cache = False
    agent = QwenRawHyperLoRA(base, args.rank, args.layers, args.width,
        order_invariant_source=False,
        max_source_length=max_source_length).to(args.device)
    optimizer = torch.optim.AdamW(
        (p for p in agent.parameters() if p.requires_grad), lr=args.lr)
    best_dev, best_step, best_state, history = float("-inf"), 0, None, []
    for step in range(1, args.steps + 1):
        agent.train()
        task = rng.choice(split["train"])
        batch = rng.sample(task["targets"], k=min(args.batch_size,
                                                   len(task["targets"])))
        source = (shared_source or rng.choice(task["source_variants"]))
        factors = agent.compile_adapters(source)
        output = logits(agent, batch, factors, choice_ids, pad_id, args.device)
        truth = torch.tensor([x["label"] for x in batch], device=args.device)
        correct_ce = F.cross_entropy(output, truth, reduction="none")
        loss = correct_ce.mean()
        if args.pair_weight:
            wrong_task = rng.choice([other for other in split["train"]
                                     if other["task_id"] != task["task_id"]])
            wrong_source = (rng.choice(wrong_task["source_variants"]) if
                            args.prefix_augmentation else wrong_task["source"])
            wrong_factors = agent.compile_adapters(wrong_source)
            wrong_output = logits(agent, batch, wrong_factors, choice_ids,
                                  pad_id, args.device)
            wrong_ce = F.cross_entropy(wrong_output, truth,
                                       reduction="none")
            # A task's own source should explain its later expert actions
            # better than an independently sampled task's source. No rule
            # names or future expert actions enter the source encoder.
            loss = loss + args.pair_weight * F.softplus(
                args.pair_margin + correct_ce - wrong_ce).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in agent.parameters() if p.requires_grad], 1.)
        optimizer.step()
        agent.mount(None)
        if step % args.eval_every == 0 or step == args.steps:
            dev_items = ([item for item in split["dev"]
                if item["task_id"] in positive_dev_ids]
                if args.select_positive_dev else split["dev"])
            dev = evaluate(agent, dev_items, choice_ids, pad_id,
                           args.device, args.batch_size, shared_source)
            history.append({"step": step, "loss": float(loss.detach()),
                            "dev": {key: dev[key] for key in
                                    ("n", "correct", "wrong", "none",
                                     "source_swap_changes")}})
            print(json.dumps(history[-1]), flush=True)
            selection_score = (dev["correct"] +
                args.contrast_selection_weight *
                (dev["correct"] - dev["wrong"]))
            if selection_score > best_dev:
                best_dev, best_step = selection_score, step
                best_state = {name: p.detach().cpu().clone()
                              for name, p in agent.named_parameters()
                              if p.requires_grad}
    if best_state is None:
        raise RuntimeError("No dev checkpoint")
    parameters = dict(agent.named_parameters())
    with torch.no_grad():
        for name, value in best_state.items():
            parameters[name].copy_(value.to(parameters[name].device))
    final = {name: evaluate(agent, split[name], choice_ids, pad_id,
                            args.device, args.batch_size, shared_source)
             for name in ("dev", "test")}
    positive_source_final = {}
    for name in ("dev", "test"):
        ids = {item["task_id"] for item in items[name]
               if item.get("online_source_positive", 0) > 0}
        if len(ids) >= 2:
            subset = [item for item in split[name]
                      if item["task_id"] in ids]
            positive_source_final[name] = evaluate(agent, subset,
                choice_ids, pad_id, args.device, args.batch_size,
                shared_source)
    source_kind = ("reviewed model-generated live source" if
                   all("online_source_positive" in item
                       for group in items.values() for item in group)
                   else "official XLand-100B source history")
    result = {"protocol": source_kind + " -> generated LoRA on frozen Qwen; target expert-action agreement on task-held-out independent official history; no target environment rollout during training",
              "annotations_sha256": hashlib.sha256(raw).hexdigest(),
              "model_config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
              "seed": args.seed, "steps": args.steps,
              "best_step": best_step, "batch_size": args.batch_size,
              "lr": args.lr, "rank": args.rank, "layers": args.layers,
              "pair_weight": args.pair_weight,
              "pair_margin": args.pair_margin,
              "contrast_selection_weight": args.contrast_selection_weight,
              "best_dev_selection_score": best_dev,
              "action_count": action_count,
              "max_source_length": max_source_length,
              "prefix_augmentation": args.prefix_augmentation,
              "positive_only_train": args.positive_only_train,
              "select_positive_dev": args.select_positive_dev,
              "train_task_count": len(split["train"]),
              "positive_source_final": positive_source_final,
              "shared_source": args.shared_source,
              "shared_source_task_id": split["train"][0]["task_id"]
                  if args.shared_source else None,
              "history": history, **final}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    torch.save({"trainable_state": best_state,
                "annotations_sha256": result["annotations_sha256"],
                "seed": args.seed}, args.checkpoint)
    print(json.dumps({name: {key: final[name][key] for key in
          ("n", "correct", "wrong", "none", "source_swap_changes")}
          for name in final}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=.0003)
    parser.add_argument("--pair-weight", type=float, default=0.)
    parser.add_argument("--pair-margin", type=float, default=1.)
    parser.add_argument("--contrast-selection-weight", type=float, default=0.)
    parser.add_argument("--shared-source", action="store_true")
    parser.add_argument("--prefix-augmentation", action="store_true")
    parser.add_argument("--positive-only-train", action="store_true")
    parser.add_argument("--select-positive-dev", action="store_true")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if (args.steps < 1 or args.eval_every < 1 or args.batch_size < 1 or
            args.lr <= 0 or args.pair_weight < 0 or
            args.contrast_selection_weight < 0 or
            (args.shared_source and args.pair_weight) or
            not 0 < args.gpu_fraction <= 1):
        parser.error("Invalid training budget")
    run(args)


if __name__ == "__main__":
    main()
