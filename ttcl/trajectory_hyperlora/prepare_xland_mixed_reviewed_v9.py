"""Merge already-reviewed XLand train tasks without moving frozen dev/test."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.train_xland_raw_hyperlora import checked_query


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trivial", type=Path, required=True)
    parser.add_argument("--medium", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    trivial = json.loads(args.trivial.read_text())
    medium = json.loads(args.medium.read_text())
    for record in (trivial, medium):
        if record["budget"]["source_length"] != 16:
            raise ValueError("Mixed run expects the same source length")
        for group in record["split"].values():
            for item in group:
                for query in item["queries"]:
                    checked_query(query)
    medium_train = []
    for item in medium["split"]["train"]:
        entry = copy.deepcopy(item)
        # Only grouping identifiers are namespaced. Reviewed target inputs,
        # exact content hashes, and labels remain unchanged.
        entry["task_id"] = f"medium:{item['task_id']}"
        entry["ruleset_id"] = f"medium:{item['ruleset_id']}"
        medium_train.append(entry)
    train = copy.deepcopy(trivial["split"]["train"]) + medium_train
    ids = [item["task_id"] for item in train]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate merged task ID")
    result = {"protocol": "Mixed official XLand-100B reviewed training histories; trivial frozen dev/test unchanged; no new target labels",
              "initial_annotations_sha256": sha(args.trivial),
              "medium_annotations_sha256": sha(args.medium),
              "source_dataset_identity": {
                  "trivial": trivial["dataset_identity"],
                  "medium": medium["dataset_identity"]},
              "budget": {"source_length": 16, "action_count": 6},
              "split": {"train": train,
                        "dev": copy.deepcopy(trivial["split"]["dev"]),
                        "test": copy.deepcopy(trivial["split"]["test"])}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"split": {name: len(group)
        for name, group in result["split"].items()},
        "input_sha256": sha(args.output)}), flush=True)


if __name__ == "__main__":
    main()
