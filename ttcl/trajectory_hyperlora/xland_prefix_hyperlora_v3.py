"""Separate v3 XLand hypernetwork: train on partial, audited trajectories.

The frozen v1 checkpoints remain untouched. This trainer starts from one of
them, uses only independently re-reviewed train groups for optimization, and
selects a new checkpoint with development prefixes. No rule kind or expected
action is supplied to the model as a feature.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.collect_xland_crossed_rules import digest
from ttcl.trajectory_hyperlora.review_xland_multi_mechanism import validate
from ttcl.trajectory_hyperlora.train_xland_qwen_hyperlora import (
    ACTIONS, ARMS, QwenRawHyperLoRA, evaluate, group, queries,
)
from ttcl.trajectory_hyperlora.train_xland_raw_hyperlora import make_split


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args: argparse.Namespace) -> dict:
    if args.review.exists():
        raise FileExistsError(args.review)
    raw = args.candidates.read_bytes()
    original = json.loads(args.reference_annotations.read_text())
    verified = validate(json.loads(raw))
    if original["candidates_sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError("Original candidate source changed")
    new_split = {}
    for split in ("train", "dev", "test"):
        if verified[split] != original["split"][split]:
            raise ValueError("Original independent review changed")
        expanded = []
        for item in verified[split]:
            for length in (1, 2, 3):
                for subset in itertools.combinations(range(3), length):
                    arms = {}
                    for kind in ARMS:
                        full = item["arms"][kind]["queries"][0]
                        content = dict(full["model_input"])
                        content["source_episodes"] = [
                            content["source_episodes"][index]
                            for index in subset]
                        arms[kind] = {"queries": [{
                            "model_input": content,
                            "input_sha256": digest(content),
                            "target_action": full["target_action"],
                            "reviewed_target": True,
                            "review_basis": "Fresh prefix input hash and target rederived from independently audited candidate environment transitions"}]}
                    expanded.append({"item_id": f"{item['item_id']}|{subset}",
                                     "rule_sha256": item["rule_sha256"],
                                     "source_subset": list(subset),
                                     "arms": arms})
        new_split[split] = expanded
    review = {
        "protocol": "Freshly rederived source-prefix action targets with exact input-content hashes; same real XLand transitions and isolated rule-content split as original review",
        "candidates_sha256": hashlib.sha256(raw).hexdigest(),
        "reference_annotations_sha256": sha(args.reference_annotations),
        "split": new_split,
    }
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({name: len(items) for name, items in new_split.items()}),
          flush=True)
    return review


def train(args: argparse.Namespace) -> dict:
    if args.output.exists() or args.save_checkpoint.exists():
        raise FileExistsError("Fresh v3 outputs required")
    review_raw = args.review.read_bytes()
    review = json.loads(review_raw)
    if (review["candidates_sha256"] != sha(args.candidates) or
            review["reference_annotations_sha256"] !=
            sha(args.reference_annotations) or
            len(review["split"]["train"]) != 108 * 7 or
            len(review["split"]["dev"]) != 49 * 7):
        raise ValueError("Prefix review or train/dev partition changed")
    original = torch.load(args.start_checkpoint, map_location="cpu",
                          weights_only=True)
    if (original["annotations_sha256"] != sha(args.reference_annotations) or
            original.get("order_invariant_source") is not True):
        raise ValueError("v1 checkpoint lineage changed")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                                   device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    ids = [tokenizer(str(action), add_special_tokens=False).input_ids
           for action in ACTIONS]
    if any(len(value) != 1 for value in ids):
        raise ValueError("Action label is not a single token")
    choice_ids = [value[0] for value in ids]
    train_items = [item for item in review["split"]["train"]
                   if len(item["source_subset"]) >= 2]
    dev_two = [item for item in review["split"]["dev"]
               if len(item["source_subset"]) == 2]
    dev_full = [item for item in review["split"]["dev"]
                if len(item["source_subset"]) == 3]
    train_data = make_split(train_items, args.device)
    dev_two_data = make_split(dev_two, args.device)
    dev_full_data = make_split(dev_full, args.device)
    train_prompts = queries(train_items, tokenizer)
    dev_two_prompts = queries(dev_two, tokenizer)
    dev_full_prompts = queries(dev_full, tokenizer)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    base.config.use_cache = False
    agent = QwenRawHyperLoRA(base, order_invariant_source=True).to(args.device)
    parameters = dict(agent.named_parameters())
    trainable = {name: parameter for name, parameter in parameters.items()
                 if parameter.requires_grad}
    if set(trainable) != set(original["trainable_state"]):
        raise ValueError("v1 trainable architecture changed")
    with torch.no_grad():
        for name, value in original["trainable_state"].items():
            trainable[name].copy_(value.to(trainable[name].device))
    anchor = {name: parameter.detach().clone()
              for name, parameter in trainable.items()}
    optimizer = torch.optim.AdamW(trainable.values(), lr=args.lr,
                                  weight_decay=0)
    by_length = {length: [index for index, item in enumerate(train_items)
                          if len(item["source_subset"]) == length]
                 for length in (2, 3)}
    history = []
    best = None
    for step in range(1, args.steps + 1):
        agent.train()
        length = 3 if random.random() < args.full_fraction else 2
        index = random.choice(by_length[length])
        data = group(train_data, index)
        agent.mount(agent.compile_adapters(data))
        logits = agent.choice_logits(train_prompts[index], choice_ids,
                                     3, args.device)
        target = torch.tensor([ACTIONS.index(int(value))
                               for value in data["target"]],
                              device=args.device)
        competing = target.roll(-1)
        margin = logits.gather(1, target[:, None]).squeeze(1) - \
            logits.gather(1, competing[:, None]).squeeze(1)
        ce = F.cross_entropy(logits, target)
        contrast = F.softplus(1.0 - margin).mean()
        anchor_loss = sum((parameter - anchor[name]).square().mean()
                          for name, parameter in trainable.items())
        loss = ce + args.pair_weight * contrast + \
            args.anchor_weight * anchor_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
        optimizer.step()
        agent.mount(None)
        if step % args.eval_every == 0 or step == args.steps:
            agent.eval()
            two = evaluate(agent, dev_two_data, dev_two_prompts,
                           choice_ids, args.device, limit=args.dev_probe)
            full = evaluate(agent, dev_full_data, dev_full_prompts,
                            choice_ids, args.device, limit=args.dev_probe)
            row = {"step": step, "ce": float(ce.detach()),
                   "two_trial_dev_probe": two, "full_dev_probe": full}
            history.append(row)
            print(json.dumps(row), flush=True)
            selection = (two["correct"] + full["correct"],
                         full["correct"], -step)
            if best is None or selection > best[0]:
                best = (selection, step, {name: parameter.detach().cpu().clone()
                    for name, parameter in trainable.items()})
    if best is None:
        raise RuntimeError("No v3 checkpoint selected")
    with torch.no_grad():
        for name, value in best[2].items():
            trainable[name].copy_(value.to(trainable[name].device))
    final_dev = {
        "two_trial": evaluate(agent, dev_two_data, dev_two_prompts,
                              choice_ids, args.device),
        "full": evaluate(agent, dev_full_data, dev_full_prompts,
                         choice_ids, args.device),
    }
    report = {
        "protocol": "v3 partial-prefix trajectory-to-Qwen-LoRA continuation; frozen Qwen, original order-invariant hypernetwork initialization; only independently re-reviewed train prefixes optimize; dev probe selects checkpoint; no test prefix or environment target used in training",
        "review_sha256": hashlib.sha256(review_raw).hexdigest(),
        "start_checkpoint_sha256": sha(args.start_checkpoint),
        "seed": args.seed, "steps": args.steps, "lr": args.lr,
        "pair_weight": args.pair_weight,
        "anchor_weight": args.anchor_weight,
        "full_fraction": args.full_fraction,
        "dev_probe": args.dev_probe,
        "selected_step": best[1], "history": history,
        "final_dev": final_dev,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    torch.save({"trainable_state": best[2],
                "annotations_sha256": sha(args.reference_annotations),
                "order_invariant_source": True,
                "prefix_review_sha256": report["review_sha256"],
                "start_checkpoint_sha256": report["start_checkpoint_sha256"],
                "seed": args.seed}, args.save_checkpoint)
    print(json.dumps({"selected_step": best[1], "final_dev": final_dev}),
          flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "train"))
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/xland_multi_mechanism_candidates_v1_20261005.json"))
    parser.add_argument("--reference-annotations", type=Path, default=Path(
        "data/annotations/xland_multi_mechanism_reviewed_v1_20261005.json"))
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--start-checkpoint", type=Path)
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.55)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--dev-probe", type=int, default=24)
    parser.add_argument("--lr", type=float, default=0.00003)
    parser.add_argument("--pair-weight", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=0.05)
    parser.add_argument("--full-fraction", type=float, default=.25)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-checkpoint", type=Path)
    args = parser.parse_args()
    if (args.steps < 1 or args.eval_every < 1 or args.dev_probe < 1 or
            args.lr <= 0 or not 0 <= args.full_fraction <= 1 or
            not 0 < args.gpu_fraction <= 1):
        parser.error("Invalid v3 training budget")
    if args.command == "prepare":
        prepare(args)
    else:
        if args.start_checkpoint is None or args.output is None or \
                args.save_checkpoint is None:
            parser.error("train needs start checkpoint and fresh output paths")
        train(args)


if __name__ == "__main__":
    main()
