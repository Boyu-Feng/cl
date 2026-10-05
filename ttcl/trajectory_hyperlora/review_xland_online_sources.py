"""Rebind official expert targets to newly observed model source episodes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ttcl.trajectory_hyperlora.collect_xland_official_histories import digest
from ttcl.trajectory_hyperlora.evaluate_xland_official_live import (
    selected_source_steps, sha256,
)
from ttcl.trajectory_hyperlora.review_xland_official_histories import valid_state
from ttcl.trajectory_hyperlora.train_xland_raw_hyperlora import checked_query


PUBLIC_KEYS = ("state", "action", "next_state", "reward", "done")


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    original = json.loads(args.annotations.read_text())
    collected = json.loads(args.sources.read_text())
    if collected["annotations_sha256"] != sha256(args.annotations):
        raise ValueError("Original annotation lineage changed")
    if args.source_limit not in (16, 64):
        raise ValueError("Supported source limits are 16 and 64")
    output = {"protocol": "New reviewed official expert targets with exact model-generated source-input bindings; original task-disjoint split preserved",
              "source_limit": args.source_limit,
              "original_annotations_sha256": sha256(args.annotations),
              "collected_sources_sha256": sha256(args.sources),
              "dataset_identity": original["dataset_identity"],
              "benchmark_id": original["benchmark_id"],
              "environment_id": original["environment_id"],
              "budget": original["budget"],
              "source_protocol": {key: collected[key] for key in
                  ("checkpoint_sha256", "benchmark_sha256", "source_steps",
                   "source_epsilon", "source_feedback_window",
                   "stream_source_updates")},
              "split": {name: [] for name in ("train", "dev", "test")},
              "failures": collected["failures"]}
    for split in output["split"]:
        originals = {item["task_id"]: item
                     for item in original["split"][split]}
        seen = set()
        for row in collected["split"][split]:
            task_id = row["task_id"]
            if task_id not in originals or task_id in seen:
                raise ValueError("Source task missing or duplicated in split")
            seen.add(task_id)
            item = originals[task_id]
            if row["ruleset_id"] != item["ruleset_id"] or \
                    digest(row["source"]) != row["source_sha256"]:
                raise ValueError("Source task/ruleset binding changed")
            steps = row["source"]["steps"]
            if not steps or len(steps) > collected["source_steps"]:
                raise ValueError("Source episode outside budget")
            for step in steps:
                if (step["action"] not in range(6) or
                        not math.isfinite(step["reward"]) or
                        not valid_state(step["state"]) or
                        not valid_state(step["next_state"])):
                    raise ValueError("Invalid online transition")
            original_public, original_window = selected_source_steps(steps)
            if (original_window != tuple(row["selected_window"]) or
                    digest(original_public) != row["selected_public_sha256"]):
                raise ValueError("Selected public source changed")
            public, window = selected_source_steps(steps, args.source_limit)
            if any(set(step) != set(PUBLIC_KEYS) for step in public):
                raise ValueError("Source contains non-public fields")
            source_episodes = [{"goal": [0, 0], "steps": public}]
            queries = []
            for old_query in item["queries"]:
                content, label = checked_query(old_query)
                new_content = {"source_episodes": source_episodes,
                    "target_initial_state": content["target_initial_state"],
                    "goal": content["goal"]}
                queries.append({"model_input": new_content,
                    "input_sha256": digest(new_content),
                    "target_action": label, "reviewed_target": True,
                    "review_basis": "Official expert action rechecked against unchanged target state/index and newly reviewed model source episode",
                    "original_input_sha256": old_query["input_sha256"],
                    "target_history_id": old_query["target_history_id"],
                    "target_transition_index": old_query[
                        "target_transition_index"]})
            output["split"][split].append({"task_id": task_id,
                "ruleset_id": item["ruleset_id"],
                "online_source_sha256": row["source_sha256"],
                "online_source_positive": row["source"]["positive_rewards"],
                "selected_window": list(window),
                "selected_source_sha256": digest(source_episodes),
                "queries": queries})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({name: {"n": len(rows), "positive": sum(
        item["online_source_positive"] > 0 for item in rows)}
        for name, rows in output["split"].items()}), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-limit", type=int, default=16)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
