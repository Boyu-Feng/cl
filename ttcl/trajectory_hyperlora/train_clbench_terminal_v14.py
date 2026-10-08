"""Add reviewed terminal-action Cohort targets to the successful CLBench v2 mix.

This keeps the v2 training objective and checkpoint. Only official train/dev
indices 0-7 enter labels; held-out indices 8+ remain evaluation-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v2 import (
    ROOT, compact_actor_messages, target_text,
)
from ttcl.trajectory_hyperlora.train_clbench_hyperlora_v1 import clean_records, train


COHORT = ROOT / "results/trajectory_hyperlora/clbench_cohort_train_collect0_7_20261008_episodes/cohort_studies"
BASE = ROOT / "data/annotations/clbench_hyperlora_v2_candidates_20261008.json"


def episode(index: int):
    path = COHORT / f"episode_{index+1:03}"
    row = json.loads((path / "row.json").read_text())
    trajectory = json.loads((path / "trajectory.json").read_text())
    events = [json.loads(line) for line in (path / "responses.jsonl").read_text().splitlines()]
    if (row["status"] != "complete" or not trajectory["completed"] or
            float(row["reward"]) != float(trajectory["reward"])):
        raise ValueError("Incomplete official Cohort training episode")
    return row, trajectory, events, path


def build_candidates():
    old = json.loads(BASE.read_text())
    labels = list(old["labels"])
    provenance = {"previous_candidates_sha256": file_hash(BASE),
                  "cohort_targets": []}
    for source_index, target_index, split in ((4, 5, "train"), (6, 7, "dev")):
        _, source, _, source_path = episode(source_index)
        target_row, target, events, target_path = episode(target_index)
        if float(target_row["reward"]) <= 0 or not source["steps"] or not target["steps"]:
            raise ValueError("Terminal target has no positive official reward")
        action = target["steps"][-1]["action"]
        matches = [event for event in events if event.get("action") == action and
                   event.get("parse_error") is None]
        if not isinstance(action, dict) or len(matches) != 1:
            raise ValueError("Terminal action must match one schema-valid generation")
        messages = compact_actor_messages(matches[0]["messages"], 2)
        if messages[-1]["role"] != "user":
            raise ValueError("Final action prompt changed")
        content = {"domain": "cohort_studies", "source_index": source_index,
            "target_index": target_index,
            "source_trajectory_sha256": file_hash(source_path / "trajectory.json"),
            "target_trajectory_sha256": file_hash(target_path / "trajectory.json"),
            "target_responses_sha256": file_hash(target_path / "responses.jsonl"),
            "source_records": clean_records(source),
            "source_context": target_text(source),
            "target_context": target_text(target),
            "target_messages": messages,
            "target_action": json.dumps(action, ensure_ascii=False, sort_keys=True),
            "target_reward": float(target_row["reward"]), "split": split}
        labels.append({**content, "input_content_sha256": digest(content)})
        provenance["cohort_targets"].append({
            "source_index": source_index, "target_index": target_index,
            "source_hash": content["source_trajectory_sha256"],
            "target_hash": content["target_trajectory_sha256"],
            "reward": target_row["reward"]})
    return {"protocol": "CLBench v2 mix plus positive Cohort final-action targets; test 8+ excluded",
            "provenance": provenance, "labels": labels,
            "input_content_sha256": digest({"provenance": provenance, "labels": labels})}


def prepare(args):
    if args.candidates.exists() or args.review.exists():
        raise FileExistsError("Fresh annotation paths required")
    candidate = build_candidates()
    args.candidates.parent.mkdir(parents=True, exist_ok=True)
    args.candidates.write_text(json.dumps(candidate, ensure_ascii=False, indent=2) + "\n")
    review = {"candidates_sha256": file_hash(args.candidates),
              "annotations": [{"input_content_sha256": row["input_content_sha256"],
                               "approved": False, "review_basis": "pending"}
                              for row in candidate["labels"]]}
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"train": sum(x["split"] == "train" for x in candidate["labels"]),
                      "dev": sum(x["split"] == "dev" for x in candidate["labels"])}), flush=True)


def checked(args):
    candidate = json.loads(args.candidates.read_text())
    review = json.loads(args.review.read_text())
    if (candidate != build_candidates() or
            review["candidates_sha256"] != file_hash(args.candidates) or
            [x["input_content_sha256"] for x in candidate["labels"]] !=
            [x["input_content_sha256"] for x in review["annotations"]] or
            not all(x["approved"] and x["review_basis"] != "pending"
                    for x in review["annotations"])):
        raise ValueError("Unreviewed terminal-action training data")
    return candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "train"))
    parser.add_argument("--candidates", type=Path,
        default=ROOT / "data/annotations/clbench_terminal_v14_candidates_20261008.json")
    parser.add_argument("--review", type=Path,
        default=ROOT / "data/annotations/clbench_terminal_v14_reviewed_20261008.json")
    parser.add_argument("--checkpoint", type=Path,
        default=ROOT / "results/trajectory_hyperlora/clbench_hyperlora_v2_20261008.pt")
    parser.add_argument("--checkpoint-out", type=Path,
        default=ROOT / "results/trajectory_hyperlora/clbench_terminal_v14_20261008.pt")
    parser.add_argument("--output", type=Path,
        default=ROOT / "results/trajectory_hyperlora/clbench_terminal_v14_train_20261008.json")
    parser.add_argument("--model", type=Path,
        default=ROOT / "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--gpu-fraction", type=float, default=.85)
    parser.add_argument("--source-tokens", type=int, default=2048)
    parser.add_argument("--context-limit", type=int, default=32768)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    (prepare(args) if args.command == "prepare" else
     train(args, include_thinking=True, reviewed_data=checked(args)))


if __name__ == "__main__":
    main()
