"""Train the typed operator gate from reviewed, paired public-action deltas.

The initial available supervision supports array_union only. Other operators
remain registered but abstain until independently reviewed utility labels exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v2 import ROOT
from ttcl.trajectory_hyperlora.train_structured_gate_v16 import CHAINS, DOMAIN
from ttcl.trajectory_hyperlora.typed_evidence_v17 import (
    extract, features, fit, proposals,
)


def reviewed_rows(candidates_path: Path, review_path: Path):
    candidates = json.loads(candidates_path.read_text())
    review = json.loads(review_path.read_text())
    if review["candidates_sha256"] != file_hash(candidates_path):
        raise ValueError("Reviewed candidate file changed")
    approvals = {x["input_content_sha256"]: x for x in review["annotations"]}
    if len(approvals) != len(candidates["labels"]):
        raise ValueError("Missing or duplicate review")
    for label in candidates["labels"]:
        item = approvals[label["input_content_sha256"]]
        if not item["approved"] or item["review_basis"] == "pending":
            raise ValueError("Unreviewed utility label")
        yield label


def training_rows(candidates_path: Path, review_path: Path):
    paths = {}
    for stem, start, stop, split in CHAINS:
        for index in range(start, stop):
            paths[index] = (stem, split)
    rows = []
    for label in reviewed_rows(candidates_path, review_path):
        target_index = label["target_index"]
        source_index = label["source_index"]
        stem, split = paths[target_index]
        if paths[source_index][0] != stem or label["split"] != split:
            raise ValueError("Cross-chain source or split changed")
        root = (ROOT / "results/trajectory_hyperlora" /
                (stem + "_episodes") / DOMAIN)
        target_path = root / "base" / f"episode_{target_index+1:03}" / "trajectory.json"
        source_path = root / "online" / f"episode_{source_index+1:03}" / "trajectory.json"
        if (file_hash(target_path) != label["target_trajectory_sha256"] or
                file_hash(source_path) != label["source_trajectory_sha256"]):
            raise ValueError("Trajectory content changed")
        target = json.loads(target_path.read_text())["steps"][0]
        source = json.loads(source_path.read_text())["steps"][0]
        report = json.loads((ROOT / "results/trajectory_hyperlora" /
            (stem + ".json")).read_text())
        source_reward = next(x["online"]["reward"] for x in report["rows"]
            if x["domain"] == DOMAIN and x["index"] == source_index)
        evidence = extract(source["action"], source_index=source_index,
            trajectory_sha256=file_hash(source_path),
            feedback=source["public_feedback"],
            reward=float(source_reward), query=source["query"])
        selected = [x for x in evidence if x.path == label["schema_path"] and
                    x.action_sha256 == label["source_action_sha256"]]
        if len(selected) != 1:
            raise ValueError("Reviewed source action or schema path changed")
        candidates = [x for x in proposals(target["action"], selected[0])
                      if x.operator == "array_union"]
        if not candidates and float(label["utility_delta"]) == 0:
            continue
        if len(candidates) != 1:
            raise ValueError("Reviewed candidate operator changed")
        rows.append((split, features(candidates[0], target["action"],
            target_index=target_index, target_query=target["query"]),
            float(label["utility_delta"])))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=ROOT /
        "data/annotations/clbench_evidence_gate_v16_candidates_20261008.json")
    parser.add_argument("--review", type=Path, default=ROOT /
        "data/annotations/clbench_evidence_gate_v16_reviewed_20261008.json")
    parser.add_argument("--output", type=Path, default=ROOT /
        "results/trajectory_hyperlora/clbench_typed_gate_v17_20261008.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = training_rows(args.candidates, args.review)
    train = [("array_union", x, y) for split, x, y in rows if split == "train"]
    dev = [(x, y) for split, x, y in rows if split == "dev"]
    model = fit(train, penalty=1., min_examples=8)
    model.update({"candidates_sha256": file_hash(args.candidates),
        "review_sha256": file_hash(args.review),
        "train_rows": len(train), "dev_rows": len(dev),
        "trained_operators": ["array_union"],
        "untrained_operators": ["numeric_mean", "scalar_copy", "text_hint"]})
    fitted = model["operators"]["array_union"]
    estimates = []
    for x, y in dev:
        scaled = (x - np.array(fitted["mean"])) / np.array(fitted["std"])
        prediction = float(np.dot(np.r_[1., scaled], fitted["weight"]))
        estimates.append((prediction, y))
    model["dev_mse"] = float(np.mean([(prediction-y)**2 for prediction,y in estimates]))
    model["dev_positive_truth"] = sum(y > 0 for _,y in estimates)
    model["dev_positive_lower_bound"] = sum(
        prediction > fitted["residual_margin"] for prediction,_ in estimates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"model_sha256": file_hash(args.output),
        "train": len(train), "dev": len(dev), "dev_mse": model["dev_mse"],
        "dev_positive_lower_bound": model["dev_positive_lower_bound"]}))


if __name__ == "__main__":
    main()
