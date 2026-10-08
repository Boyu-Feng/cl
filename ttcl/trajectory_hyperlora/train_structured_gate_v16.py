"""Fit a domain-agnostic evidence gate on paired next-action reward deltas.

Previously exposed BSM task chains 20-55 train the gate; 56-79 select one
ridge penalty. New indices 80+ stay sealed for subsequent online evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v2 import ROOT
from ttcl.trajectory_hyperlora.structured_evidence_gate_v16 import (
    candidate, extract, features, fit_ridge, predict,
)
from ttcl.trajectory_hyperlora.train_clbench_cumulative_hyperlora_v4 import official_score


CHAINS = (
    ("clbench_hybrid_v7_bsm_eventonly20_31_20261008", 20, 32, "train"),
    ("clbench_safe_v8_bsm_eventonly32_43_20261008", 32, 44, "train"),
    ("clbench_mixed_v10_bsm_poker44_55_20261008", 44, 56, "train"),
    ("clbench_mixed_v11_bsm56_79_20261008", 56, 80, "dev"),
)
DOMAIN = "blind_spectrum_monitoring"


def build_candidates():
    labels = []
    provenance = []
    for stem, start, stop, split in CHAINS:
        report_path = ROOT / "results/trajectory_hyperlora" / (stem + ".json")
        report = json.loads(report_path.read_text())
        root = report_path.parent / (stem + "_episodes") / DOMAIN
        rows = {x["index"]: x for x in report["rows"] if x["domain"] == DOMAIN}
        if list(sorted(rows)) != list(range(start, stop)):
            raise ValueError("Changed training chain targets")
        provenance.append({"stem": stem, "report_sha256": file_hash(report_path),
                           "start": start, "stop": stop, "split": split})
        history = []
        for index in range(start, stop):
            row = rows[index]
            if row["base"]["status"] != "complete" or row["online"]["status"] != "complete":
                raise ValueError("Training chain has missing reward")
            base_path = root / "base" / f"episode_{index+1:03}" / "trajectory.json"
            online_path = root / "online" / f"episode_{index+1:03}" / "trajectory.json"
            base = json.loads(base_path.read_text())
            online = json.loads(online_path.read_text())
            if len(base["steps"]) != 1 or len(online["steps"]) != 1:
                raise ValueError("Expected one official action in evidence task")
            action = base["steps"][0]["action"]
            target_query = base["steps"][0]["query"]
            base_score = official_score(index, action)
            if abs(base_score - float(row["base"]["reward"])) > 1e-6:
                raise ValueError("Official base score changed")
            for source_index, evidence in history[-8:]:
                for item in evidence:
                    vector = features(action, item, target_index=index,
                                      target_query=target_query)
                    if vector is None:
                        continue
                    score = official_score(index, candidate(action, item))
                    content = {"split": split, "target_index": index,
                        "source_index": source_index,
                        "source_trajectory_sha256": item.source_trajectory_sha256,
                        "source_action_sha256": item.source_action_sha256,
                        "target_trajectory_sha256": file_hash(base_path),
                        "schema_path": item.schema_path,
                        "features": vector.tolist(),
                        "base_reward": base_score, "candidate_reward": score,
                        "utility_delta": score - base_score}
                    labels.append({**content, "input_content_sha256": digest(content)})
            step = online["steps"][0]
            history.append((index, extract(step["action"], source_index=index,
                trajectory_sha256=file_hash(online_path),
                feedback=step["public_feedback"],
                reward=float(row["online"]["reward"]), query=step["query"])))
    return {"protocol": "Public evidence read utility; prior exposed 20-79, new test 80+ sealed",
            "provenance": provenance, "labels": labels,
            "input_content_sha256": digest({"provenance": provenance, "labels": labels})}


def prepare(args):
    if args.candidates.exists() or args.review.exists():
        raise FileExistsError("Fresh candidate/review paths required")
    data = build_candidates()
    args.candidates.parent.mkdir(parents=True, exist_ok=True)
    args.candidates.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    review = {"candidates_sha256": file_hash(args.candidates),
              "annotations": [{"input_content_sha256": x["input_content_sha256"],
                               "approved": False, "review_basis": "pending"}
                              for x in data["labels"]]}
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"train": sum(x["split"] == "train" for x in data["labels"]),
                      "dev": sum(x["split"] == "dev" for x in data["labels"])}), flush=True)


def checked(args):
    data = json.loads(args.candidates.read_text())
    review = json.loads(args.review.read_text())
    if (data != build_candidates() or
            review["candidates_sha256"] != file_hash(args.candidates) or
            [x["input_content_sha256"] for x in data["labels"]] !=
            [x["input_content_sha256"] for x in review["annotations"]] or
            not all(x["approved"] and x["review_basis"] != "pending"
                    for x in review["annotations"])):
        raise ValueError("Unreviewed evidence utility labels")
    return data


def train(args):
    if args.model_out.exists():
        raise FileExistsError(args.model_out)
    data = checked(args)
    train_rows = [(np.array(x["features"]), float(x["utility_delta"]))
                  for x in data["labels"] if x["split"] == "train"]
    dev_rows = [(np.array(x["features"]), float(x["utility_delta"]))
                for x in data["labels"] if x["split"] == "dev"]
    candidates = []
    for penalty in (0.01, 0.1, 1.0, 10.0, 100.0):
        model = fit_ridge(train_rows, penalty)
        prediction = np.array([predict(model, value) for value, _ in dev_rows])
        truth = np.array([target for _, target in dev_rows])
        candidates.append({"penalty": penalty, "dev_mse": float(np.mean((prediction - truth)**2)),
            "dev_positive_prediction": int((prediction > 0).sum()),
            "dev_positive_truth": int((truth > 0).sum()),
            "dev_negative_truth": int((truth < 0).sum())})
    best = min(candidates, key=lambda x: x["dev_mse"])
    model = fit_ridge(train_rows, best["penalty"])
    model.update({"training_protocol": data["protocol"],
                  "candidates_sha256": file_hash(args.candidates),
                  "review_sha256": file_hash(args.review),
                  "dev_selection": candidates,
                  "train_chains": data["provenance"]})
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"model_sha256": file_hash(args.model_out),
                      "best_penalty": best["penalty"],
                      "train": len(train_rows), "dev": len(dev_rows),
                      "dev_mse": best["dev_mse"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "train"))
    parser.add_argument("--candidates", type=Path,
        default=ROOT / "data/annotations/clbench_evidence_gate_v16_candidates_20261008.json")
    parser.add_argument("--review", type=Path,
        default=ROOT / "data/annotations/clbench_evidence_gate_v16_reviewed_20261008.json")
    parser.add_argument("--model-out", type=Path,
        default=ROOT / "results/trajectory_hyperlora/clbench_evidence_gate_v16_20261008.json")
    args = parser.parse_args()
    (prepare if args.command == "prepare" else train)(args)


if __name__ == "__main__":
    main()
