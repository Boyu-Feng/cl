"""Train-only task/source compatibility threshold for a generated LoRA.

The gate compares learned contextual latents of a completed source trajectory
and the next task's initial public observation. It uses reviewed train pairs,
never development rewards or target walkthroughs, to set one threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_same_game_retry import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import (
    contextual_source_fields, contextual_text_fields,
)
from ttcl.trajectory_hyperlora.train_alfworld_retry_reward import label_content


def checked_tasks(args, expected_tasks: int = 120):
    dataset = json.loads(args.labels.read_text())
    review = json.loads(args.label_review.read_text())
    if (dataset["source_scope"] != "sibling_expert" or
            dataset.get("split") != "train_large" or
            dataset["sibling_source_review_sha256"] !=
                file_hash(args.source_review) or
            review["labels_sha256"] != file_hash(args.labels) or
            len(dataset["tasks"]) != expected_tasks or
            expected_tasks < 12 or expected_tasks % 6):
        raise ValueError("Gate calibration requires reviewed large train tasks")
    approved = {row["input_content_sha256"]: row
                for row in review["annotations"]}
    for task in dataset["tasks"]:
        for label in task["labels"]:
            note = approved.get(label["input_content_sha256"])
            if (digest(label_content(label)) != label["input_content_sha256"] or
                    note is None or note["approved"] is not True or
                    note["target_game"] != task["target_game"] or
                    note["source_episode_sha256"] !=
                        task["source_episode_sha256"]):
                raise ValueError("Unreviewed gate calibration target")
    return dataset["tasks"]


def pair_score(agent, source_fields, target_fields):
    with torch.no_grad():
        source = agent.encode(source_fields)
        target = agent.encode(target_fields)
        return float((source * target).mean())


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    tasks = checked_tasks(args)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if agent.encoder_kind != "contextual":
        raise ValueError("Task gate needs a contextual LoRA generator")
    scores = []
    for task in tasks:
        label = task["labels"][0]
        observation = label["target_messages"][1]["content"].split(
            "\nAvailable commands:\n", 1)[0]
        target = contextual_text_fields(agent, tokenizer,
            "Current task observation:\n" + observation,
            args.device, args.max_source_tokens)
        own = contextual_source_fields(agent, tokenizer,
            label["source_records"], args.device, args.max_source_tokens)
        wrong = contextual_source_fields(agent, tokenizer,
            task["wrong_records"], args.device, args.max_source_tokens)
        scores.append({"target_game": task["target_game"],
            "own": pair_score(agent, own, target),
            "wrong": pair_score(agent, wrong, target)})
    values = sorted({row[arm] for row in scores
                     for arm in ("own", "wrong")})
    thresholds = ([values[0] - 1.] +
        [(left + right) / 2 for left, right in zip(values, values[1:])] +
        [values[-1] + 1.])
    threshold = max(thresholds, key=lambda value: (
        sum(row["own"] >= value for row in scores) +
        sum(row["wrong"] < value for row in scores), -abs(value)))
    result = {"protocol": "One train-only threshold for learned source/target latent dot compatibility; positive is another expert trial of same task, negative is same-family different task; no development target or reward",
        "checkpoint_sha256": file_hash(args.checkpoint),
        "labels_sha256": file_hash(args.labels),
        "label_review_sha256": file_hash(args.label_review),
        "source_review_sha256": file_hash(args.source_review),
        "max_source_tokens": args.max_source_tokens,
        "threshold": threshold,
        "summary": {"pairs": len(scores),
            "own_above_threshold": sum(row["own"] >= threshold
                                       for row in scores),
            "wrong_below_threshold": sum(row["wrong"] < threshold
                                         for row in scores),
            "own_score_above_wrong": sum(row["own"] > row["wrong"]
                                         for row in scores)},
        "scores": scores}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False,
                                      indent=2) + "\n")
    print(json.dumps({"threshold": threshold, **result["summary"]}),
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train120_labels_20261006.json"))
    parser.add_argument("--label-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train120_labels_reviewed_20261006.json"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train120_reviewed_20261006.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.6)
    parser.add_argument("--max-source-tokens", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_source_tokens < 2 or not 0 < args.gpu_fraction <= 1:
        parser.error("Invalid gate calibration budget")
    run(args)


if __name__ == "__main__":
    main()
