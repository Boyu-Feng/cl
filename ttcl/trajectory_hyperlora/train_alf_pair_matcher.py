"""Generic train-only compatibility probe for completed ALFWorld histories.

Frozen Qwen contextual vectors represent a completed source trajectory and
the next task's public initial observation. A regularized linear classifier
learns which reviewed source belongs to that task. Development pairs are used
only after training and threshold selection; no target walkthrough is read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ttcl.experience_evolution.environment import make_env
from ttcl.trajectory_hyperlora.alfworld_same_game_retry import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.calibrate_alf_sibling_gate import checked_tasks
from ttcl.trajectory_hyperlora.contextual_alf_source import (
    contextual_source_fields, contextual_text_fields,
)


def pair_feature(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source = F.layer_norm(source.float(), (source.shape[-1],))
    target = F.layer_norm(target.float(), (target.shape[-1],))
    return torch.cat((source * target, (source - target).abs()), dim=-1).squeeze(0).cpu()


def dev_pairs(args):
    candidates = json.loads(args.dev_candidates.read_text())
    review = json.loads(args.dev_source_review.read_text())
    if (candidates.get("split", "dev") != "dev" or
            review["candidates_sha256"] != file_hash(args.dev_candidates) or
            len(candidates["rows"]) != 9 or
            len(review["annotations"]) != 9):
        raise ValueError("Frozen development source lineage changed")
    approved = {note["input_content_sha256"]: note
                for note in review["annotations"]}
    pairs = []
    for row in candidates["rows"]:
        note = approved.get(row["input_content_sha256"])
        if (note is None or note["approved"] is not True or
                file_hash(args.data_root / row["target_game"]) !=
                    row["target_game_sha256"] or
                digest(note["source_records"]) !=
                    note["source_records_sha256"] or
                digest(note["wrong_records"]) !=
                    note["wrong_records_sha256"]):
            raise ValueError("Development pair content changed")
        pairs.append((row, note))
    return pairs


def threshold(scores, labels):
    values = sorted(set(float(value) for value in scores))
    candidates = ([values[0] - 1.] +
        [(left + right) / 2 for left, right in zip(values, values[1:])] +
        [values[-1] + 1.])
    return max(candidates, key=lambda value: (
        sum((float(score) >= value) == bool(label)
            for score, label in zip(scores, labels, strict=True)),
        -abs(value)))


def run(args):
    if args.output.exists() or args.save_checkpoint.exists():
        raise FileExistsError("Fresh matcher outputs required")
    tasks = checked_tasks(args)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if agent.encoder_kind != "contextual":
        raise ValueError("Matcher expects a contextual source checkpoint")
    train_features = []
    for task in tasks:
        label = task["labels"][0]
        observation = label["target_messages"][1]["content"].split(
            "\nAvailable commands:\n", 1)[0]
        target = contextual_text_fields(agent, tokenizer,
            "Current task observation:\n" + observation,
            args.device, args.max_source_tokens,
            pooling=args.pooling)["contextual"]
        own = contextual_source_fields(agent, tokenizer,
            label["source_records"], args.device,
            args.max_source_tokens, pooling=args.pooling)["contextual"]
        wrong = contextual_source_fields(agent, tokenizer,
            task["wrong_records"], args.device,
            args.max_source_tokens, pooling=args.pooling)["contextual"]
        train_features.append((pair_feature(own, target),
                               pair_feature(wrong, target)))
    development = dev_pairs(args)
    dev_features = []
    for row, note in development:
        env = make_env(args.data_root / row["target_game"])
        try:
            observation = str(env.reset()["feedback"])
        finally:
            env.close()
        target = contextual_text_fields(agent, tokenizer,
            "Current task observation:\n" + observation,
            args.device, args.max_source_tokens,
            pooling=args.pooling)["contextual"]
        own = contextual_source_fields(agent, tokenizer,
            note["source_records"], args.device,
            args.max_source_tokens, pooling=args.pooling)["contextual"]
        wrong = contextual_source_fields(agent, tokenizer,
            note["wrong_records"], args.device,
            args.max_source_tokens, pooling=args.pooling)["contextual"]
        dev_features.append((pair_feature(own, target),
                             pair_feature(wrong, target)))

    # Four of each task family are held out from fitting. This split is fixed
    # by source game content; the development tasks are never consulted here.
    by_family = {}
    for index, task in enumerate(tasks):
        family = task["target_game"].split("/")[2].split("-", 1)[0]
        by_family.setdefault(family, []).append(index)
    fit, holdout = [], []
    for family, indices in sorted(by_family.items()):
        ordered = sorted(indices, key=lambda index: digest([
            "pair_matcher_split", tasks[index]["target_game"]]))
        if len(ordered) != 20:
            raise ValueError(f"Matcher family is incomplete: {family}")
        fit.extend(ordered[:16])
        holdout.extend(ordered[16:])
    fit_rows = torch.stack([feature for index in fit
        for feature in train_features[index]])
    feature_mean = fit_rows.mean(0)
    feature_scale = fit_rows.std(0).clamp_min(.05)

    def matrix(indices, pairs):
        return torch.stack([(feature - feature_mean) / feature_scale
            for index in indices for feature in pairs[index]])

    fit_x = matrix(fit, train_features)
    hold_x = matrix(holdout, train_features)
    dev_x = matrix(list(range(len(dev_features))), dev_features)
    fit_y = torch.tensor([1., 0.] * len(fit))
    hold_y = torch.tensor([1., 0.] * len(holdout))
    torch.manual_seed(args.seed)
    classifier = nn.Linear(fit_x.shape[-1], 1)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    best = None
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        logits = classifier(fit_x).flatten()
        loss = F.binary_cross_entropy_with_logits(logits, fit_y) + \
            args.l2 * classifier.weight.square().sum()
        loss.backward()
        optimizer.step()
        if (step + 1) % 20 == 0:
            with torch.no_grad():
                held_loss = float(F.binary_cross_entropy_with_logits(
                    classifier(hold_x).flatten(), hold_y))
            if best is None or held_loss < best[0]:
                best = (held_loss, step + 1,
                    {name: value.detach().clone()
                     for name, value in classifier.state_dict().items()})
    classifier.load_state_dict(best[2])
    with torch.no_grad():
        fit_scores = classifier(fit_x).flatten().tolist()
        hold_scores = classifier(hold_x).flatten().tolist()
        dev_scores = classifier(dev_x).flatten().tolist()
    cut = threshold(fit_scores, fit_y.tolist())

    def summary(scores, count):
        return {"pairs": count,
            "own_above_wrong": sum(scores[2*i] > scores[2*i+1]
                                   for i in range(count)),
            "own_mounted": sum(scores[2*i] >= cut for i in range(count)),
            "wrong_rejected": sum(scores[2*i+1] < cut for i in range(count))}

    result = {"protocol": "Frozen-Qwen raw contextual pair features, regularized train-only linear matcher; 16/4 task directories per family for fit/internal holdout, threshold fit-only; development source pairs scored after model selection; no development rewards or walkthroughs",
        "source_checkpoint_sha256": file_hash(args.checkpoint),
        "labels_sha256": file_hash(args.labels),
        "label_review_sha256": file_hash(args.label_review),
        "source_review_sha256": file_hash(args.source_review),
        "dev_candidates_sha256": file_hash(args.dev_candidates),
        "dev_source_review_sha256": file_hash(args.dev_source_review),
        "max_source_tokens": args.max_source_tokens,
        "pooling": args.pooling,
        "seed": args.seed, "steps": args.steps,
        "best_step": best[1], "threshold": cut,
        "fit": summary(fit_scores, len(fit)),
        "internal_holdout": summary(hold_scores, len(holdout)),
        "development": summary(dev_scores, len(dev_features)),
        "development_scores": [
            {"target_game": row["target_game"],
             "own": dev_scores[2 * index],
             "wrong": dev_scores[2 * index + 1]}
            for index, (row, _) in enumerate(development)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False,
                                      indent=2) + "\n")
    torch.save({"state": {name: value.detach().cpu()
                           for name, value in classifier.state_dict().items()},
                "feature_mean": feature_mean,
                "feature_scale": feature_scale,
                "threshold": cut,
                "source_checkpoint_sha256": result["source_checkpoint_sha256"],
                "labels_sha256": result["labels_sha256"]},
               args.save_checkpoint)
    print(json.dumps({key: result[key] for key in (
        "best_step", "threshold", "fit", "internal_holdout",
        "development")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_contextual120_match800_20261006.pt"))
    parser.add_argument("--labels", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train120_labels_20261006.json"))
    parser.add_argument("--label-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train120_labels_reviewed_20261006.json"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train120_reviewed_20261006.json"))
    parser.add_argument("--dev-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_dev_candidates_20261006.json"))
    parser.add_argument("--dev-source-review", type=Path, default=Path(
        "data/annotations/alf_sibling_dev_reviewed_20261006.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.6)
    parser.add_argument("--max-source-tokens", type=int, default=2048)
    parser.add_argument("--pooling", choices=("last", "mean"), default="last")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=.01)
    parser.add_argument("--weight-decay", type=float, default=.1)
    parser.add_argument("--l2", type=float, default=.0001)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if (args.steps < 20 or args.steps % 20 or args.lr <= 0 or
            args.l2 < 0 or args.weight_decay < 0 or
            args.max_source_tokens < 2 or not 0 < args.gpu_fraction <= 1):
        parser.error("Invalid matcher training budget")
    run(args)


if __name__ == "__main__":
    main()
