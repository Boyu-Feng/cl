"""Audit v4 live XLand reward records and fresh source/target bindings."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ttcl.trajectory_hyperlora.evaluate_xland_official_live import digest, sha256
from ttcl.trajectory_hyperlora.train_xland_raw_hyperlora import checked_query


def audit_episode(episode, budget):
    steps = episode["steps"]
    if not 1 <= len(steps) <= budget:
        raise ValueError("Missing or overbudget episode")
    for i, step in enumerate(steps):
        if step["action"] not in range(6) or not math.isfinite(step["reward"]):
            raise ValueError("Invalid action or reward")
        if i and steps[i - 1]["next_state"] != step["state"]:
            raise ValueError("Broken trajectory continuity")
        if i < len(steps) - 1 and step["done"]:
            raise ValueError("Episode continued after done")
    if not math.isclose(episode["return"], sum(x["reward"] for x in steps),
                        abs_tol=1e-6):
        raise ValueError("Incorrect episode return")
    if episode["positive"] != any(x["reward"] > 0 for x in steps):
        raise ValueError("Incorrect positive-reward flag")
    if episode["ended"] != bool(steps[-1]["done"]):
        raise ValueError("Incorrect episode termination")


def audit_reviewed(result, annotation):
    items = annotation["split"]["test"][:len(result["test_task_ids"])]
    if ([item["task_id"] for item in items] != result["test_task_ids"] or
            [item["ruleset_id"] for item in items] != result["test_ruleset_ids"]):
        raise ValueError("Test task list changed")
    expected = {(item["task_id"], episode)
                for item in items for episode in range(result["target_episodes"])}
    seen = set()
    sources = [checked_query(item["queries"][0])[0]["source_episodes"]
               for item in items]
    for row in result["rows"]:
        key = (row["task_id"], row["target_episode"])
        if key not in expected or key in seen:
            raise ValueError("Repeated or unexpected target episode")
        seen.add(key)
        index = result["test_task_ids"].index(row["task_id"])
        if row["ruleset_id"] != items[index]["ruleset_id"]:
            raise ValueError("Wrong ruleset ID")
        initial = row["arms"]["none"]["steps"][0]["state"]
        if row["target_initial_sha256"] != digest(initial):
            raise ValueError("Target reset hash changed")
        for arm in ("none", "correct", "wrong"):
            episode = row["arms"][arm]
            audit_episode(episode, result["budget"])
            if episode["steps"][0]["state"] != initial:
                raise ValueError("Arms use different target reset")
        for arm, source in (("correct", sources[index]),
                            ("wrong", sources[(index + 1) % len(items)])):
            content = {"source_episodes": source,
                       "target_initial_state": initial, "goal": [0, 0]}
            if (row["source_input_sha256"][arm] != digest(content) or
                    row["source_history_sha256"][arm] != digest(source)):
                raise ValueError("Source/target input binding changed")
    if seen != expected or result["failures"]:
        raise ValueError("Incomplete or failed reviewed-source evaluation")
    return len(seen)


def audit_cross_episode(result):
    prefixes = ([result["source_episodes"]] if
                result.get("only_final_target") else
                range(result["source_episodes"] + 1))
    expected = {(rid, prefix) for rid in result["rule_ids"]
                for prefix in prefixes}
    seen = set()
    for row in result["rows"]:
        key = (row["ruleset_id"], row["source_prefix"])
        if key not in expected or key in seen:
            raise ValueError("Repeated or unexpected source prefix")
        seen.add(key)
        initial = row["targets"]["none"]["steps"][0]["state"]
        if row["target_initial_sha256"] != digest(initial):
            raise ValueError("Target reset hash changed")
        sources = row["source_episodes"]
        if (len(sources) != row["source_prefix"] or
                row["source_sha256"] != digest(sources) or
                row["source_positive_so_far"] != sum(
                    source["positive"] for source in sources)):
            raise ValueError("Source prefix record changed")
        for source in sources:
            audit_episode(source, result["budget"])
        for arm in ("none", "all_history", "reward_gated"):
            episode = row["targets"][arm]
            audit_episode(episode, result["budget"])
            if episode["steps"][0]["state"] != initial:
                raise ValueError("Arms use different target reset")
        if (row["source_prefix"] == 0) != (
                row["all_history_input_sha256"] is None):
            raise ValueError("Unexpected empty/full source binding")
        if sources:
            limit = result.get("source_window", 16)
            flat = [step for source in sources for step in source["steps"]][-limit:]
            public = [{key: step[key] for key in
                       ("state", "action", "next_state", "reward", "done")}
                      for step in flat]
            content = {"source_episodes": [{"goal": [0, 0],
                "steps": public}], "target_initial_state": initial,
                "goal": [0, 0]}
            if row["all_history_input_sha256"] != digest(content):
                raise ValueError("All-history content binding changed")
            positive = [source for source in sources if source["positive"]]
            if positive:
                flat = []
                for source in positive:
                    end = max(i for i, step in enumerate(source["steps"])
                              if step["reward"] > 0) + 1
                    flat.extend(source["steps"][:end])
                public = [{key: step[key] for key in
                           ("state", "action", "next_state", "reward", "done")}
                          for step in flat[-limit:]]
                content["source_episodes"][0]["steps"] = public
                if row["reward_gated_input_sha256"] != digest(content):
                    raise ValueError("Reward-gated content binding changed")
        if row["source_positive_so_far"] == 0 and \
                row["reward_gated_input_sha256"] is not None:
            raise ValueError("Reward gate used unsuccessful source")
    if seen != expected or result["failures"]:
        raise ValueError("Incomplete or failed cross-episode evaluation")
    return len(seen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = json.loads(args.result.read_text())
    if args.annotations and result["annotations_sha256"] != sha256(args.annotations):
        raise ValueError("Annotations SHA changed")
    if "test_task_ids" in result:
        if not args.annotations:
            parser.error("Reviewed-source result needs --annotations")
        n = audit_reviewed(result, json.loads(args.annotations.read_text()))
    else:
        n = audit_cross_episode(result)
    record = {"result_sha256": sha256(args.result),
              "annotations_sha256": (sha256(args.annotations) if args.annotations else None),
              "audited_rows": n, "passed": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
