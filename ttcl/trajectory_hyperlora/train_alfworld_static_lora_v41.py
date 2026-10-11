"""Direct static B-LoRA SFT on the exact 33 reviewed hypernetwork tasks.

The frozen base and rank-4 A factors come from the old 68/134 checkpoint.
Only two static B factors (last two down projections) are optimized. This
matches the generated factor shape while removing trajectory conditioning.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.train_alf_next_task import query_ids, target_loss
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save


def checked_labels(args):
    dataset = json.loads(args.labels.read_text())
    review = json.loads(args.source_review.read_text())
    source_training = json.loads(args.source_training.read_text())
    if (source_training["labels_sha256"] != file_hash(args.labels)
            or source_training["label_review_sha256"] != file_hash(args.source_review)
            or source_training["selected_train_tasks"] != 33
            or source_training["train_actions"] != 269
            or review["labels_sha256"] != file_hash(args.labels)
            or len(dataset["tasks"]) != 33 or len(review["annotations"]) != 269):
        raise ValueError("Changed exact old hypernetwork training data")
    labels = [row for task in dataset["tasks"] for row in task["labels"]]
    annotations = {(row["target_game"], row["input_content_sha256"]): row
                   for row in review["annotations"]}
    if len(labels) != 269 or len(annotations) != 269:
        raise ValueError("Missing or duplicate action labels")
    for task in dataset["tasks"]:
        if ("/train/" not in "/" + task["target_game"] or
                file_hash(args.data_root / task["target_game"]) != task["target_game_sha256"]):
            raise ValueError("Changed official train game")
        for row in task["labels"]:
            annotation = annotations.get((row["target_game"], row["input_content_sha256"]))
            if (annotation is None or annotation["approved"] is not True
                    or row["target_game"] != task["target_game"]
                    or row["source_episode_sha256"] != task["source_episode_sha256"]):
                raise ValueError("Changed reviewed action target")
    return dataset


def fixed(args, dataset):
    value = {"method": "v41 direct static B-LoRA SFT, same 33 hypernetwork train tasks",
        "labels_sha256": file_hash(args.labels),
        "source_review_sha256": file_hash(args.source_review),
        "source_training_report_sha256": file_hash(args.source_training),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "trainer_sha256": file_hash(Path(__file__)),
        "actor_sha256": file_hash(Path(__file__).with_name("alfworld_zero_shot_probe.py")),
        "rank": 4, "layers": 2, "target_module": "last_two_mlp.down_proj",
        "trainable_factors": "B_only", "steps": args.steps, "lr": args.lr,
        "weight_decay": args.weight_decay, "seed": args.seed,
        "history_turns": 2, "max_prompt_tokens": 3000}
    targets = []
    for i, task in enumerate(dataset["tasks"]):
        content = {"index": i, "game": task["target_game"],
            "game_sha256": task["target_game_sha256"],
            "action_content_sha256": [x["input_content_sha256"] for x in task["labels"]],
            "source_episode_sha256": task["source_episode_sha256"],
            "trainer_sha256": value["trainer_sha256"],
            "checkpoint_sha256": value["checkpoint_sha256"]}
        targets.append({**content, "input_content_sha256": digest(content),
                        "reviewed_target": True})
    return value, targets


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    dataset = checked_labels(args)
    value, targets = fixed(args, dataset)
    save(args.review, {"protocol": "Exact 33 official train tasks and 269 reviewed successful action targets used by old hypernetwork; static rank-4 B on same two modules and frozen common A; no valid split reward in training",
        **value, "targets": targets, "targets_sha256": digest(targets)})
    print(json.dumps({"tasks": len(targets), "actions": 269}), flush=True)


def static_agent(args):
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if len(agent.adapters) != 2 or any(layer.rank != 4 for layer in agent.adapters):
        raise ValueError("Changed old generator factor geometry")
    agent.requires_grad_(False)
    factors = [torch.nn.Parameter(torch.zeros((1, layer.base.out_features,
        layer.rank), device=args.device, dtype=torch.float32)) for layer in agent.adapters]
    for layer, factor in zip(agent.adapters, factors, strict=True):
        layer.b = factor
    return agent, tokenizer, factors


def train(args):
    if args.model_output.exists() or args.report_output.exists():
        raise FileExistsError("Use new static LoRA output paths")
    dataset = checked_labels(args)
    value, targets = fixed(args, dataset)
    review = json.loads(args.review.read_text())
    if (review.get("targets") != targets or review.get("targets_sha256") != digest(targets)
            or any(review.get(k) != v for k, v in value.items())):
        raise ValueError("Changed reviewed direct LoRA inputs")
    agent, tokenizer, factors = static_agent(args)
    tasks = dataset["tasks"]
    prefixes = {row["input_content_sha256"]: query_ids(tokenizer, row,
        history_turns=2, max_prompt_tokens=3000)
        for task in tasks for row in task["labels"]}
    optimizer = torch.optim.AdamW(factors, lr=args.lr,
                                  weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    losses = []
    for step in range(args.steps):
        task = rng.choice(tasks)
        row = rng.choice(task["labels"])
        optimizer.zero_grad(set_to_none=True)
        loss = target_loss(agent, tokenizer, row,
                           prefixes[row["input_content_sha256"]], args.device)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite direct LoRA SFT loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(factors, 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 25 == 0 or step + 1 == args.steps:
            print(json.dumps({"step": step + 1,
                "loss_last25": sum(losses[-25:]) / len(losses[-25:])}), flush=True)
    final = [x.detach().cpu().float().clone() for x in factors]
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"factors": final, "factor_shapes": [list(x.shape) for x in final],
        "review_sha256": file_hash(args.review),
        "checkpoint_sha256": value["checkpoint_sha256"],
        "trainer_sha256": value["trainer_sha256"],
        "rank": 4, "layers": 2, "steps": args.steps, "lr": args.lr,
        "seed": args.seed}, args.model_output)
    report = {"protocol": review["protocol"], **value,
        "review_sha256": file_hash(args.review),
        "model_sha256": file_hash(args.model_output),
        "factor_sha256": factors_hash(final), "actions": 269,
        "loss_first25": sum(losses[:25]) / 25,
        "loss_last25": sum(losses[-25:]) / 25,
        "loss_history": losses}
    save(args.report_output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "loss_history"}), flush=True)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "train"))
    defaults = {
        "labels": "results/trajectory_hyperlora/alf_sibling_simple_train40_onpolicy_best_labels_20261006.json",
        "source_review": "data/annotations/alf_sibling_simple_train40_onpolicy_best_reviewed_20261006.json",
        "source_training": "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.json",
        "checkpoint": "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt",
        "model": "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507",
        "data_root": "ttcl/data/alfworld_delta",
        "review": "data/annotations/alfworld_static_lora_sft_v41_train33_20261010.json",
        "model_output": "results/trajectory_hyperlora/alfworld_static_lora_sft_v41_train33_20261010.pt",
        "report_output": "results/trajectory_hyperlora/alfworld_static_lora_sft_v41_train33_20261010.json"}
    for name, value in defaults.items():
        p.add_argument("--" + name.replace("_", "-"), type=Path, default=Path(value))
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--gpu-fraction", type=float, default=.65)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    (prepare if args.command == "prepare" else train)(args)
