"""Review official-history expert-action targets and bind exact model inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from ttcl.trajectory_hyperlora.collect_xland_official_histories import (
    ACTIONS, digest,
)


def valid_state(value: dict) -> bool:
    rows = value.get("observation")
    return (value.get("pocket") == [0, 0] and
            isinstance(rows, list) and len(rows) == 5 and
            all(isinstance(row, list) and len(row) == 5 and
                all(isinstance(tile, list) and len(tile) == 2 and
                    all(isinstance(x, int) and 0 <= x <= 63 for x in tile)
                    for tile in row) for row in rows))


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.candidates.read_bytes()
    candidate = json.loads(raw)
    budget = candidate["budget"]
    actions = tuple(range(budget.get("action_count", len(ACTIONS))))
    ids, rulesets = set(), set()
    tasks = []
    for item in candidate["tasks"]:
        task_id, ruleset_id = item["task_id"], item["ruleset_id"]
        if task_id in ids or ruleset_id in rulesets or \
                not 0 <= task_id < budget["max_task_id"]:
            raise ValueError("Duplicate or out-of-budget official task")
        ids.add(task_id)
        rulesets.add(ruleset_id)
        source, queries = item["source_steps"], item["queries"]
        if (digest(source) != item["source_sha256"] or
                digest(queries) != item["target_sha256"] or
                not 1 <= len(source) <= budget["source_length"] or
                len(queries) != len(actions) * budget["per_action"] or
                item["source_history_id"] == item["target_history_id"]):
            raise ValueError("Source or target provenance changed")
        source_indices = [step["transition_index"] for step in source]
        if (source_indices != list(range(source_indices[0],
                                       source_indices[0] + len(source))) or
                source_indices[-1] >= budget["max_source"] or
                source[-1]["reward"] <= 0):
            raise ValueError("Source is not a bounded first-success segment")
        public_steps = []
        for step in source:
            if (step["history_index"] != item["source_history_id"] or
                    step["action"] not in range(6) or
                    not math.isfinite(step["reward"]) or
                    not valid_state(step["state"]) or
                    not valid_state(step["next_state"])):
                raise ValueError("Invalid official source transition")
            public_steps.append({key: step[key] for key in
                ("state", "action", "next_state", "reward", "done")})
        source_episodes = [{"goal": [0, 0], "steps": public_steps}]
        seen_states, seen_indices = set(), set()
        labels = []
        reviewed_queries = []
        for row in queries:
            action = row["expert_action"]
            label = int(action)
            index = row["transition_index"]
            if (row["history_index"] != item["target_history_id"] or
                    not budget["target_start"] <= index < budget["target_stop"] or
                index in seen_indices or label not in actions or
                    not valid_state(row["target_state"])):
                raise ValueError("Invalid official expert target")
            seen_indices.add(index)
            state_hash = digest(row["target_state"])
            if state_hash in seen_states:
                raise ValueError("Repeated target observation")
            seen_states.add(state_hash)
            labels.append(label)
            model_input = {"source_episodes": source_episodes,
                "target_initial_state": row["target_state"],
                "goal": [0, 0]}
            reviewed_queries.append({"model_input": model_input,
                "input_sha256": digest(model_input),
                "target_action": label,
                "reviewed_target": True,
                "review_basis": "Official expert_actions at distinct held-out history indices",
                "target_history_id": item["target_history_id"],
                "target_transition_index": index})
        if any(labels.count(action) != budget["per_action"]
               for action in actions):
            raise ValueError("Targets are not action-balanced")
        tasks.append({"task_id": task_id, "ruleset_id": ruleset_id,
                      "source_history_id": item["source_history_id"],
                      "source_last_index": source_indices[-1],
                      "source_sha256": digest(source_episodes),
                      "queries": reviewed_queries})
    failed_ids = {row["task_id"] for row in candidate["failed_candidates"]}
    if ids & failed_ids or len(ids | failed_ids) != budget["max_task_id"]:
        raise ValueError("Missing or duplicated scanned task failures")
    random.Random(args.seed).shuffle(tasks)
    n = len(tasks)
    n_test = max(1, round(n * .15))
    n_dev = max(1, round(n * .15))
    if n - n_test - n_dev < 4:
        raise ValueError("Too few task-disjoint training tasks")
    split = {"train": tasks[:n - n_dev - n_test],
             "dev": tasks[n - n_dev - n_test:n - n_test],
             "test": tasks[n - n_test:]}
    result = {"protocol": "New reviewed official XLand expert-action targets with exact input-content bindings; task/ruleset-disjoint split",
              "candidates_sha256": hashlib.sha256(raw).hexdigest(),
              "dataset_identity": candidate["dataset_identity"],
              "benchmark_id": candidate.get("benchmark_id", "medium-1m"),
              "environment_id": candidate.get("environment_id",
                                               "XLand-MiniGrid-R1-13x13"),
              "budget": budget, "split_seed": args.seed,
              "failed_candidates": candidate["failed_candidates"],
              "split": split}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"split": {k: len(v) for k, v in split.items()},
                      "failed": len(failed_ids)}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202610052)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
