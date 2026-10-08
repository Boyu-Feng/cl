"""Fit one task-neutral gate over reviewed array and numeric operator deltas."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v2 import BENCH, ROOT
from ttcl.trajectory_hyperlora.train_typed_numeric_v19 import prepare_rows
from ttcl.trajectory_hyperlora.train_typed_gate_v17 import training_rows
from ttcl.trajectory_hyperlora.typed_evidence_v17 import fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--array-candidates", type=Path, default=ROOT /
        "data/annotations/clbench_evidence_gate_v16_candidates_20261008.json")
    parser.add_argument("--array-review", type=Path, default=ROOT /
        "data/annotations/clbench_evidence_gate_v16_reviewed_20261008.json")
    parser.add_argument("--numeric-candidates", type=Path, default=ROOT /
        "data/annotations/clbench_typed_numeric_v19_candidates_20261008.json")
    parser.add_argument("--numeric-review", type=Path, default=ROOT /
        "data/annotations/clbench_typed_numeric_v19_review_20261008.json")
    parser.add_argument("--output", type=Path, default=ROOT /
        "results/trajectory_hyperlora/clbench_typed_gate_v19_20261008.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    numeric = json.loads(args.numeric_candidates.read_text())
    review = json.loads(args.numeric_review.read_text())
    labels = numeric["labels"]
    os.chdir(BENCH)
    if (review["candidates_sha256"] != file_hash(args.numeric_candidates) or
            numeric["input_content_sha256"] != digest(labels) or
            numeric != prepare_rows() or
            len(review["annotations"]) != len(labels)):
        raise ValueError("Numeric label file/review changed")
    for label, annotation in zip(labels, review["annotations"], strict=True):
        content = {key: value for key, value in label.items()
                   if key != "input_content_sha256"}
        if (label["input_content_sha256"] != digest(content) or
                annotation["input_content_sha256"] != label["input_content_sha256"] or
                not annotation["approved"] or
                annotation["review_basis"] == "pending" or
                label["source_index"] >= label["target_index"] or
                label["target_index"] >= 8):
            raise ValueError("Unreviewed or out-of-split numeric utility")
    array = training_rows(args.array_candidates, args.array_review)
    train = [("array_union", x, y) for split,x,y in array if split == "train"]
    train += [(x["operator"], np.array(x["features"]), x["utility_delta"])
        for x in labels if x["split"] == "train"]
    model = fit(train, penalty=1., min_examples=8)
    model.update({"array_candidates_sha256": file_hash(args.array_candidates),
        "array_review_sha256": file_hash(args.array_review),
        "numeric_candidates_sha256": file_hash(args.numeric_candidates),
        "numeric_review_sha256": file_hash(args.numeric_review),
        "train_rows": len(train),
        "untrained_operators": ["text_hint"]})
    evaluation = {}
    for operator in model["operators"]:
        dev = [(np.array(x["features"]), x["utility_delta"])
            for x in labels if x["split"] == "dev" and x["operator"] == operator]
        if operator == "array_union":
            dev = [(x,y) for split,x,y in array if split == "dev"]
        fitted = model["operators"][operator]
        estimates = []
        for value, truth in dev:
            scaled = (value - np.array(fitted["mean"])) / np.array(fitted["std"])
            estimate = float(np.dot(np.r_[1.,scaled], fitted["weight"]))
            estimates.append((estimate, truth))
        picked = [truth for estimate, truth in estimates
                  if estimate > fitted["residual_margin"]]
        evaluation[operator] = {"dev_rows": len(dev),
            "dev_mse": float(np.mean([(a-b)**2 for a,b in estimates])) if dev else None,
            "positive_truth": sum(y>0 for _,y in estimates),
            "negative_truth": sum(y<0 for _,y in estimates),
            "selected": len(picked),
            "selected_positive": sum(y>0 for y in picked),
            "selected_mean_utility": float(np.mean(picked)) if picked else None}
    model["dev_evaluation"] = evaluation
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"model_sha256": file_hash(args.output),
        "train_rows": len(train), "dev_evaluation": evaluation}), flush=True)


if __name__ == "__main__":
    main()
