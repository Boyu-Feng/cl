"""Distill reviewed ALFWorld winning rollouts into the trajectory LoRA head.

This is reward-filtered future-action supervision, not on-policy RL. The
frozen actor's task-conditioned hypernetwork is updated; old weights remain
untouched and all targets come from train-only official wins.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch

from ttcl.trajectory_hyperlora.alfworld_same_game_retry import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import (
    contextual_source_fields, contextual_text_fields, task_context_text,
)
from ttcl.trajectory_hyperlora.train_alf_next_task import query_ids, target_loss
from ttcl.trajectory_hyperlora.train_alfworld_retry_reward import label_content


def run(args):
    if args.output.exists() or args.save_checkpoint.exists():
        raise FileExistsError("Use fresh reward-selected result/checkpoint")
    dataset = json.loads(args.labels.read_text())
    review = json.loads(args.label_review.read_text())
    parent = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (dataset["checkpoint_sha256"] != file_hash(args.checkpoint) or
            review["labels_sha256"] != file_hash(args.labels) or
            len(dataset["tasks"]) < 12 or args.steps < 1 or args.lr <= 0):
        raise ValueError("Reward-selected training lineage or budget changed")
    notes = {note["input_content_sha256"]: note
             for note in review["annotations"]}
    examples = [row for task in dataset["tasks"] for row in task["labels"]]
    if len(notes) != len(examples):
        raise ValueError("Missing or duplicate reviewed action targets")
    for task in dataset["tasks"]:
        if task["teacher_arm"] not in ("lora", "base"):
            raise ValueError("Invalid selected winning arm")
        for row in task["labels"]:
            note = notes.get(row["input_content_sha256"])
            if (row["input_content_sha256"] != digest(label_content(row)) or
                    note is None or note["approved"] is not True or
                    note["target_game"] != task["target_game"] or
                    note["source_episode_sha256"] !=
                        task["source_episode_sha256"]):
                raise ValueError("Unreviewed or changed action target")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != "contextual" or
            agent.task_conditioned is not True or
            agent.task_context_scope != "current" or
            agent.task_pair_pooling != "mean"):
        raise ValueError("Expected current-context trajectory hypernetwork")
    all_trainable = {name for name, p in agent.named_parameters()
                     if p.requires_grad}
    if not any(name.startswith("task_pair_latent.") for name in all_trainable):
        raise ValueError("Missing task-conditioned LoRA module")
    for name, p in agent.named_parameters():
        p.requires_grad_(name.startswith("task_pair_latent."))
    params = [p for p in agent.parameters() if p.requires_grad]
    source_fields, target_fields, prefixes, anchor_factors = {}, {}, {}, {}
    for task in dataset["tasks"]:
        game = task["target_game"]
        first = task["labels"][0]["target_messages"][1]["content"].split(
            "\nAvailable commands:\n", 1)[0]
        source_fields[game] = contextual_source_fields(
            agent, tokenizer, task["labels"][0]["source_records"],
            args.device, 2048, pooling="both")
        for label in task["labels"]:
            key = label["input_content_sha256"]
            current = label["target_messages"][-1]["content"].split(
                "\nAvailable commands:\n", 1)[0]
            target_fields[key] = contextual_text_fields(agent, tokenizer,
                task_context_text(first, current), args.device, 2048,
                pooling="both")
            prefixes[key] = query_ids(tokenizer, label, history_turns=2,
                                     max_prompt_tokens=args.max_prompt_tokens)
            with torch.no_grad():
                agent.set_source(source_fields[game],
                                 target_fields=target_fields[key])
                anchor_factors[key] = [layer.b.detach().clone()
                                       for layer in agent.adapters]
                agent.set_source(None)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0)
    log = []
    for step in range(1, args.steps + 1):
        task = rng.choice(dataset["tasks"])
        label = rng.choice(task["labels"])
        key = label["input_content_sha256"]
        optimizer.zero_grad(set_to_none=True)
        agent.set_source(source_fields[task["target_game"]],
                         target_fields=target_fields[key])
        ce = target_loss(agent, tokenizer, label, prefixes[key], args.device)
        anchor = sum((layer.b - old).float().square().mean()
                     for layer, old in zip(agent.adapters,
                                           anchor_factors[key], strict=True))
        # Official reward supplies the preference; both-win routes get an
        # ordinary imitation step, unilateral wins receive extra weight.
        advantage = abs(task["old_lora_reward"] - task["base_reward"])
        loss = (1.0 + args.reward_bonus * advantage) * ce + \
               args.anchor_weight * anchor
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        agent.set_source(None)
        if step % args.log_every == 0 or step == args.steps:
            record = {"step": step, "loss": float(loss.detach()),
                      "ce": float(ce.detach()), "anchor": float(anchor.detach())}
            log.append(record)
            print(json.dumps(record), flush=True)
    state = {name: p.detach().cpu() for name, p in agent.named_parameters()
             if name in all_trainable}
    if set(state) != set(parent["trainable_state"]):
        raise ValueError("Saved trainable parameter set changed")
    checkpoint = {**parent, "trainable_state": state,
        "source_checkpoint_sha256": file_hash(args.checkpoint),
        "reward_selected_labels_sha256": file_hash(args.labels),
        "reward_selected_review_sha256": file_hash(args.label_review)}
    result = {"protocol": "Continue frozen Qwen trajectory-to-LoRA model by distilling separately replay-reviewed, train-only official winning routes; update task-pair hypernetwork only; reward-disagreement weighting plus per-action LoRA anchor; no dev/test reward or target labels",
        "source_checkpoint_sha256": file_hash(args.checkpoint),
        "labels_sha256": file_hash(args.labels),
        "label_review_sha256": file_hash(args.label_review),
        "seed": args.seed, "steps": args.steps, "lr": args.lr,
        "reward_bonus": args.reward_bonus,
        "anchor_weight": args.anchor_weight,
        "max_prompt_tokens": args.max_prompt_tokens,
        "train_tasks": len(dataset["tasks"]),
        "train_actions": len(examples), "history": log}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    torch.save(checkpoint, args.save_checkpoint)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_reward_selected_train48_v1_20261007.json"))
    parser.add_argument("--label-review", type=Path, default=Path(
        "data/annotations/alf_reward_selected_train48_v1_reviewed_20261007.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--reward-bonus", type=float, default=2.0)
    parser.add_argument("--anchor-weight", type=float, default=.3)
    parser.add_argument("--max-prompt-tokens", type=int, default=3072)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-checkpoint", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
