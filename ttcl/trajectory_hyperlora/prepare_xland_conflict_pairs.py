"""Review official same-observation, opposing-expert-action pairs.

Every target is inherited from a content-bound reviewed official query. Pair
membership is additionally bound to the exact two query payloads. No labels
are inferred from task IDs or environment rules.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

from ttcl.trajectory_hyperlora.train_xland_raw_hyperlora import checked_query


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.annotations.read_bytes()
    annotation = json.loads(raw)
    split = {}
    for name in ("train", "dev", "test"):
        by_state = defaultdict(list)
        for task in annotation["split"][name]:
            for query in task["queries"]:
                content, label = checked_query(query)
                if content["goal"] != [0, 0]:
                    raise ValueError("Unexpected official goal sentinel")
                by_state[digest(content["target_initial_state"])].append({
                    "task_id": task["task_id"],
                    "ruleset_id": task["ruleset_id"],
                    "query": query, "label": label})
        pairs = []
        for state_hash, candidates in sorted(by_state.items()):
            for left, right in itertools.combinations(candidates, 2):
                if left["task_id"] == right["task_id"] or \
                        left["label"] == right["label"]:
                    continue
                if left["query"]["model_input"]["target_initial_state"] != \
                        right["query"]["model_input"]["target_initial_state"]:
                    raise ValueError("Observation hash collision")
                row = {"observation_sha256": state_hash,
                       "left": left, "right": right,
                       "reviewed_target": True,
                       "review_basis": "Two independently reviewed official expert actions for exactly the same target observation, from disjoint rulesets"}
                row["pair_sha256"] = digest({key: row[key] for key in
                    ("observation_sha256", "left", "right")})
                pairs.append(row)
        split[name] = pairs
    result = {"protocol": "Official content-bound opposing-action pairs; no new trajectories or labels",
              "source_annotations_sha256": hashlib.sha256(raw).hexdigest(),
              "split": split,
              "counts": {name: len(rows) for name, rows in split.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False,
                                      indent=2) + "\n")
    print(json.dumps(result["counts"]), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=Path(
        "data/annotations/xland_official_history_reviewed_64_v1_20261005.json"))
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
