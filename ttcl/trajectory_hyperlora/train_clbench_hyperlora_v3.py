"""Train a shared hyper-LoRA from official paired reward at first divergence.

A matched positive delta teaches the winning LoRA action; a negative delta
teaches the stronger no-LoRA action. For poker, CALL amounts are ignored when
locating divergence because the official schema uses amount only for RAISE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v2 import ROOT, compact_actor_messages, target_text
from ttcl.trajectory_hyperlora.train_clbench_hyperlora_v1 import (
    clean_records, load_episode, train,
)

PAIR = ROOT / "results/trajectory_hyperlora/clbench_hyperlora_v2_counterfactual_20261008.json"


def executed_action(action):
    result = {key: value for key, value in action.items() if key != "thinking"}
    if result.get("action") != "RAISE":
        result.pop("amount", None)
    return result


def accepted_events(events):
    return [event for event in events if event.get("action") is not None and
            event.get("parse_error") is None]


def build_candidates():
    report = json.loads(PAIR.read_text())
    labels = []
    provenance = []
    for item in report["rows"]:
        target = item["target"]
        domain, index = target["domain"], target["index"]
        base_row, base_episode, base_events, base_dir = load_episode(domain, index)
        source_row, source, _, source_dir = load_episode(domain, index - 1)
        online_dir = PAIR.parent / (PAIR.stem + "_episodes") / domain / f"episode_{index+1:03}"
        online_episode = json.loads((online_dir / "trajectory.json").read_text())
        online_events = [json.loads(line) for line in (online_dir / "responses.jsonl").read_text().splitlines()]
        if (base_row["reward"] != target["base_reward"] or
                file_hash(base_dir / "trajectory.json") != target["base_trajectory_sha256"] or
                file_hash(source_dir / "trajectory.json") != target["source_trajectory_sha256"] or
                file_hash(online_dir / "trajectory.json") != item["online_trajectory_sha256"] or
                source_row["status"] != "complete" or
                item["online"]["status"] != "complete" or
                float(online_episode["reward"]) != float(item["online"]["reward"])):
            raise ValueError("Counterfactual lineage or reward changed")
        delta = float(item["reward_delta"])
        if abs(delta - (float(online_episode["reward"]) - float(base_episode["reward"]))) > 1e-8:
            raise ValueError("Official paired reward delta changed")
        provenance.append({"domain": domain, "index": index,
                           "base_sha256": target["base_trajectory_sha256"],
                           "online_sha256": item["online_trajectory_sha256"],
                           "reward_delta": delta})
        if delta == 0:
            continue
        first = next((i for i, (base_step, online_step) in enumerate(zip(
            base_episode["steps"], online_episode["steps"]))
            if executed_action(base_step["action"]) !=
               executed_action(online_step["action"])), None)
        if first is None:
            raise ValueError("Nonzero reward delta has no comparable action divergence")
        if base_episode["steps"][first]["query"] != online_episode["steps"][first]["query"]:
            raise ValueError("First divergent actions saw different public states")
        better = "online" if delta > 0 else "base"
        episode, events = ((online_episode, online_events) if delta > 0 else
                           (base_episode, base_events))
        approved = accepted_events(events)
        if len(approved) < first + 1 or \
           approved[first]["action"] != episode["steps"][first]["action"]:
            raise ValueError("First divergent action cannot bind to actor prompt")
        messages = compact_actor_messages(approved[first]["messages"], 2)
        if messages[-1]["role"] != "user":
            raise ValueError("First divergent target prompt invalid")
        content = {"domain": domain, "source_index": index - 1,
            "target_index": index, "first_divergence": first,
            "teacher_arm": better, "reward_delta": delta,
            "source_trajectory_sha256": file_hash(source_dir / "trajectory.json"),
            "target_trajectory_sha256": file_hash((online_dir if delta > 0 else base_dir) / "trajectory.json"),
            "target_responses_sha256": file_hash((online_dir if delta > 0 else base_dir) / "responses.jsonl"),
            "source_records": clean_records(source),
            "source_context": target_text(source),
            "target_context": target_text(episode),
            "target_messages": messages,
            "target_action": json.dumps(episode["steps"][first]["action"], ensure_ascii=False, sort_keys=True),
            "target_reward": float(episode["reward"]),
            "split": target["split"]}
        labels.append({**content, "input_content_sha256": digest(content)})
    return {"protocol": "Official matched train/dev reward delta; first executed-action divergence target; indices 8+ excluded",
            "counterfactual_report_sha256": file_hash(PAIR),
            "provenance": provenance, "labels": labels,
            "input_content_sha256": digest({"provenance": provenance, "labels": labels})}


def prepare(args):
    if args.candidates.exists() or args.review.exists():
        raise FileExistsError("Fresh counterfactual target paths required")
    value = build_candidates()
    args.candidates.parent.mkdir(parents=True, exist_ok=True)
    args.candidates.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    review = {"candidates_sha256": file_hash(args.candidates),
        "annotations": [{"input_content_sha256": x["input_content_sha256"],
                         "approved": False, "review_basis": "pending"}
                        for x in value["labels"]]}
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"labels": len(value["labels"]),
        "train": sum(x["split"] == "train" for x in value["labels"]),
        "dev": sum(x["split"] == "dev" for x in value["labels"])}), flush=True)


def checked(args):
    candidate = json.loads(args.candidates.read_text())
    review = json.loads(args.review.read_text())
    if candidate != build_candidates() or \
       review["candidates_sha256"] != file_hash(args.candidates) or \
       [x["input_content_sha256"] for x in candidate["labels"]] != \
       [x["input_content_sha256"] for x in review["annotations"]] or \
       not all(x["approved"] and x["review_basis"] != "pending" for x in review["annotations"]):
        raise ValueError("Unreviewed counterfactual training target")
    return candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "train"))
    parser.add_argument("--candidates", type=Path, default=ROOT / "data/annotations/clbench_hyperlora_v3_candidates_20261008.json")
    parser.add_argument("--review", type=Path, default=ROOT / "data/annotations/clbench_hyperlora_v3_reviewed_20261008.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results/trajectory_hyperlora/clbench_hyperlora_v2_20261008.pt")
    parser.add_argument("--checkpoint-out", type=Path, default=ROOT / "results/trajectory_hyperlora/clbench_hyperlora_v3_20261008.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "results/trajectory_hyperlora/clbench_hyperlora_v3_train_20261008.json")
    parser.add_argument("--model", type=Path, default=ROOT / "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.85)
    parser.add_argument("--source-tokens", type=int, default=2048)
    parser.add_argument("--context-limit", type=int, default=16384)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    (prepare(args) if args.command == "prepare" else
     train(args, include_thinking=True, reviewed_data=checked(args)))


if __name__ == "__main__":
    main()
