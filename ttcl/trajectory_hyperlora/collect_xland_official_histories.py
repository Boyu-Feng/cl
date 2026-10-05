"""Stream a bounded, task-disjoint pilot from official XLand-100B histories.

No expert action, future state, or future reward enters the source history.
The generated JSON is an unreviewed candidate in an ignored results path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import fsspec
import h5py
import numpy as np
import requests


DEFAULT_URL = "https://fors3toevodomain.s3.cloud.ru/medium-30k.hdf5"
NUM_COLORS = 12
ACTIONS = tuple(range(5))


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def state(packed: np.ndarray) -> dict:
    # Dataset observations omit the pocket. A zero sentinel is explicit and
    # is used only to reuse the 52-value source encoder from the pilot.
    return {"observation": np.stack(np.divmod(packed, NUM_COLORS),
                                    axis=-1).astype(int).tolist(),
            "pocket": [0, 0]}


def source_steps(group: h5py.Group, max_source: int,
                 source_length: int) -> tuple[list[dict] | None, str | None]:
    rewards = group["rewards"][0, :max_source]
    positive = np.flatnonzero(rewards > 0)
    if not len(positive):
        return None, "no_positive_reward_in_source_budget"
    last = int(positive[0])
    first = max(0, last - source_length + 1)
    observations = group["states"][0, first:last + 2]
    actions = group["actions"][0, first:last + 1]
    dones = group["dones"][0, first:last + 1]
    rows = []
    for offset, action in enumerate(actions):
        rows.append({"history_index": 0, "transition_index": first + offset,
                     "state": state(observations[offset]),
                     "action": int(action),
                     "next_state": state(observations[offset + 1]),
                     "reward": float(rewards[first + offset]),
                     "done": bool(dones[offset])})
    return rows, None


def target_queries(group: h5py.Group, task_id: int,
                   start: int, stop: int, per_action: int,
                   seed: int, action_count: int = len(ACTIONS)) -> tuple[list[dict] | None, str | None]:
    packed = group["states"][1, start:stop]
    experts = group["expert_actions"][1, start:stop]
    rng = random.Random(seed ^ (task_id * 1103515245))
    indices = list(range(len(experts)))
    rng.shuffle(indices)
    actions = tuple(range(action_count))
    selected, seen = {action: [] for action in actions}, set()
    for offset in indices:
        action = int(experts[offset])
        key = packed[offset].tobytes()
        if action not in selected or key in seen or \
                len(selected[action]) >= per_action:
            continue
        seen.add(key)
        selected[action].append({"history_index": 1,
                                 "transition_index": start + offset,
                                 "target_state": state(packed[offset]),
                                 "expert_action": action})
        if all(len(rows) == per_action for rows in selected.values()):
            break
    if any(len(rows) != per_action for rows in selected.values()):
        return None, "insufficient_unique_expert_actions"
    return [row for action in actions for row in selected[action]], None


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.max_task_id < 1 or args.source_length < 1 or \
            args.source_length > 16 or args.per_action < 1 or \
            args.target_start < args.max_source or \
            args.target_stop <= args.target_start or \
            args.action_count not in (5, 6):
        raise ValueError("Invalid bounded pilot budget")
    head = requests.head(args.url, allow_redirects=True, timeout=30)
    head.raise_for_status()
    identity = {"final_url": head.url,
                "content_length": int(head.headers["Content-Length"]),
                "etag": head.headers.get("ETag"),
                "last_modified": head.headers.get("Last-Modified")}
    result = {"protocol": "Bounded remote official XLand histories; candidates, not reviewed labels",
              "dataset_identity": identity,
              "benchmark_id": args.benchmark_id,
              "environment_id": args.environment_id,
              "budget": {"max_task_id": args.max_task_id,
                         "max_source": args.max_source,
                         "source_length": args.source_length,
                         "target_start": args.target_start,
                         "target_stop": args.target_stop,
                         "per_action": args.per_action,
                         "action_count": args.action_count,
                         "seed": args.seed},
              "tasks": [], "failed_candidates": []}
    with fsspec.open(head.url, "rb", block_size=args.block_size,
                     cache_type="readahead") as stream:
        with h5py.File(stream, "r") as h5:
            for task_id in range(args.max_task_id):
                group = h5[str(task_id)]
                ruleset_id = int(group.attrs["ruleset-id"])
                if group.attrs["benchmark-id"] != args.benchmark_id:
                    raise ValueError("Unexpected official benchmark")
                if group.attrs["env-id"] != args.environment_id:
                    raise ValueError("Unexpected official environment")
                source, reason = source_steps(group, args.max_source,
                                              args.source_length)
                target, target_reason = target_queries(group, task_id,
                    args.target_start, args.target_stop, args.per_action,
                    args.seed, args.action_count)
                if reason or target_reason:
                    result["failed_candidates"].append({"task_id": task_id,
                        "ruleset_id": ruleset_id,
                        "reason": reason or target_reason})
                else:
                    result["tasks"].append({"task_id": task_id,
                        "ruleset_id": ruleset_id,
                        "source_history_id": 0, "target_history_id": 1,
                        "source_steps": source, "queries": target,
                        "source_sha256": digest(source),
                        "target_sha256": digest(target)})
                if (task_id + 1) % 5 == 0:
                    print(json.dumps({"scanned": task_id + 1,
                        "valid": len(result["tasks"]),
                        "failed": len(result["failed_candidates"])}),
                          flush=True)
    final_head = requests.head(head.url, timeout=30)
    final_head.raise_for_status()
    if (int(final_head.headers["Content-Length"]) != identity["content_length"] or
            final_head.headers.get("ETag") != identity["etag"]):
        raise RuntimeError("Remote official HDF5 changed during collection")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"valid": len(result["tasks"]),
                      "failed": len(result["failed_candidates"])}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--benchmark-id", default="medium-1m")
    parser.add_argument("--environment-id", default="XLand-MiniGrid-R1-13x13")
    parser.add_argument("--max-task-id", type=int, default=64)
    parser.add_argument("--max-source", type=int, default=4096)
    parser.add_argument("--source-length", type=int, default=16)
    parser.add_argument("--target-start", type=int, default=8192)
    parser.add_argument("--target-stop", type=int, default=12288)
    parser.add_argument("--per-action", type=int, default=4)
    parser.add_argument("--action-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--block-size", type=int, default=2097152)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
