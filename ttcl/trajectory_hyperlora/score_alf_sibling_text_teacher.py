"""Train-only text-teacher action evidence for sibling-history LoRA distillation.

The teacher sees the reviewed prior trajectory as text. Scores are computed
only for freshly bound ALFWorld train expert actions; no development or unseen
target is read. A later student can use these scores while receiving only a
trajectory-generated LoRA at inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_same_game_retry import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_sibling_transfer import sibling_memory_text
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.train_alf_next_task import query_ids, target_loss
from ttcl.trajectory_hyperlora.train_alfworld_retry_reward import label_content


def checked_labels(args):
    dataset = json.loads(args.labels.read_text())
    review = json.loads(args.label_review.read_text())
    if (dataset["source_scope"] != "sibling_expert" or
            dataset["teacher_mode"] != "walkthrough" or
            dataset["sibling_source_review_sha256"] !=
                file_hash(args.source_review) or
            review["labels_sha256"] != file_hash(args.labels) or
            len(dataset["tasks"]) != 42):
        raise ValueError("Text-teacher train lineage changed")
    approved = {row["input_content_sha256"]: row
                for row in review["annotations"]}
    labels = [label for task in dataset["tasks"] for label in task["labels"]]
    if len(approved) != len(labels):
        raise ValueError("Text-teacher action annotations incomplete")
    for task in dataset["tasks"]:
        for label in task["labels"]:
            note = approved.get(label["input_content_sha256"])
            if (digest(label_content(label)) != label["input_content_sha256"] or
                    note is None or note["approved"] is not True or
                    note["target_game"] != task["target_game"] or
                    note["source_episode_sha256"] !=
                        task["source_episode_sha256"]):
                raise ValueError("Text-teacher action content changed")
    return dataset


def score(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    dataset = checked_labels(args)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    agent.set_source(None)
    output = {"protocol": "Train-only frozen actor action CE with correct sibling text, wrong same-family text, and no history; expert target labels already independently reviewed; no LoRA mounted; no development targets",
        "labels_sha256": file_hash(args.labels),
        "label_review_sha256": file_hash(args.label_review),
        "source_review_sha256": dataset["sibling_source_review_sha256"],
        "checkpoint_sha256": file_hash(args.checkpoint),
        "history_turns": args.history_turns,
        "max_prompt_tokens": args.max_prompt_tokens,
        "rows": []}
    for task in dataset["tasks"]:
        labels = (task["labels"] if args.all_actions else
                  task["labels"][:1])
        for label in labels:
            arms = {}
            for arm, records in (("base", None),
                                 ("own_text", label["source_records"]),
                                 ("wrong_text", task["wrong_records"])):
                messages = [dict(message) for message in label["target_messages"]]
                if records is not None:
                    messages[0]["content"] += ("\n\nPrior attempt record:\n" +
                                               sibling_memory_text(records))
                row = {**label, "target_messages": messages}
                prefix = query_ids(tokenizer, row,
                    history_turns=args.history_turns,
                    max_prompt_tokens=args.max_prompt_tokens)
                with torch.no_grad():
                    arms[arm] = float(target_loss(agent, tokenizer, row,
                        prefix, args.device))
            output["rows"].append({
                "target_game": task["target_game"],
                "input_content_sha256": label["input_content_sha256"],
                "action_index": task["labels"].index(label),
                "action": label["target_action"],
                "ce": arms})
        output["summary"] = {"tasks": len({row["target_game"]
            for row in output["rows"]}), "actions": len(output["rows"]),
            "own_better_than_wrong": sum(
                row["ce"]["own_text"] < row["ce"]["wrong_text"]
                for row in output["rows"]),
            "own_better_than_base": sum(
                row["ce"]["own_text"] < row["ce"]["base"]
                for row in output["rows"])}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(output, ensure_ascii=False,
                                        indent=2) + "\n")
        temporary.replace(args.output)
        print(json.dumps(output["summary"]), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_warmstart_20261005/seed42_v2.pt"))
    parser.add_argument("--labels", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train42_labels_20261006.json"))
    parser.add_argument("--label-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train42_labels_reviewed_20261006.json"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train42_reviewed_20261006.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.6)
    parser.add_argument("--history-turns", type=int, default=2)
    parser.add_argument("--max-prompt-tokens", type=int, default=3000)
    parser.add_argument("--all-actions", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.history_turns < 0 or args.max_prompt_tokens < 1 or
            not 0 < args.gpu_fraction <= 1):
        parser.error("Invalid text-teacher budget")
    score(args)


if __name__ == "__main__":
    main()
