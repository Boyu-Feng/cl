"""Audited paired analysis for the own-trajectory train-stream experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from math import comb
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save


def audited(report_path: Path, audit_path: Path):
    report = json.loads(report_path.read_text())
    audit = json.loads(audit_path.read_text())
    if (audit.get("audit") != "passed" or
            audit.get("report_sha256") != file_hash(report_path)):
        raise ValueError(f"Changed or unaudited report: {report_path}")
    return report


def paired(first, second):
    first_only = sum(bool(a) and not bool(b) for a, b in
                     zip(first, second, strict=True))
    second_only = sum(bool(b) and not bool(a) for a, b in
                      zip(first, second, strict=True))
    discordant = first_only + second_only
    p = min(1., 2 * sum(comb(discordant, i) for i in
                       range(min(first_only, second_only) + 1)) /
            2**discordant) if discordant else 1.
    return {"first_only": first_only, "second_only": second_only,
            "exact_two_sided_p": p}


def analyze(args):
    bootstrap = audited(args.bootstrap, args.bootstrap_audit)
    contrast = audited(args.contrast, args.contrast_audit)
    residual = audited(args.residual, args.residual_audit)
    tagged = audited(args.tagged, args.tagged_audit)
    retrieval = audited(args.retrieval, args.retrieval_audit)
    retrieval_review = json.loads(args.retrieval_review.read_text())
    review = json.loads(args.bootstrap_review.read_text())
    train_review = json.loads(args.train_review.read_text())
    if (len(bootstrap.get("rows", [])) != 120 or
            len(contrast.get("rows", [])) != 120 or
            len(residual.get("rows", [])) != 120 or
            len(tagged.get("rows", [])) != 120 or
            len(retrieval.get("rows", [])) != 120 or
            len(retrieval_review.get("targets", [])) != 120 or
            len(review.get("targets", [])) != 120 or
            bootstrap.get("bootstrap_review_sha256") != file_hash(args.train_review) or
            contrast.get("contrast_review_sha256") is None or
            bootstrap.get("review_sha256") != file_hash(args.bootstrap_review)):
        raise ValueError("Changed suffix alignment")
    arms = {name: [] for name in ("base", "hyper", "rotated", "static_B",
                                 "static_AB", "contrast_success", "contrast_failure",
                                 "prefix_failure_residual", "tagged_success",
                                 "tagged_rotated", "tagged_failure",
                                 "goal_retrieval")}
    changed_paths = {"hyper_vs_rotated": 0,
                     "contrast_success_vs_failure": 0,
                     "residual_vs_contrast_success": 0,
                     "tagged_success_vs_rotated": 0,
                     "tagged_success_vs_failure": 0,
                     "retrieval_vs_original": 0}
    same_family_rotation = []
    families = defaultdict(list)
    for i, (row, other, shifted, tagged_row, retrieved_row, target) in enumerate(zip(
            bootstrap["rows"], contrast["rows"], residual["rows"],
            tagged["rows"], retrieval["rows"], review["targets"],
            strict=True)):
        if (row["index"] != i or other["index"] != i or
                shifted["index"] != i or row["game"] != shifted["game"] or
                tagged_row["index"] != i or row["game"] != tagged_row["game"] or
                retrieved_row["index"] != i or row["game"] != retrieved_row["game"] or
                row["game"] != other["game"] or
                row["game"] != target["game"] or
                row["source_index"] != other["good_source_index"]):
            raise ValueError(f"Changed task/source alignment at {i}")
        values = {"base": row["base"]["reward"],
            "hyper": row["hyper"]["reward"],
            "rotated": row["rotated"]["reward"],
            "static_B": row["static"]["reward"],
            "static_AB": row["full_static"]["reward"],
            "contrast_success": other["good"]["reward"],
            "contrast_failure": other["bad"]["reward"],
            "prefix_failure_residual": shifted["episode"]["reward"],
            "tagged_success": tagged_row["good"]["reward"],
            "tagged_rotated": tagged_row["rotated"]["reward"],
            "tagged_failure": tagged_row["bad"]["reward"],
            "goal_retrieval": retrieved_row["episode"]["reward"]}
        for name, value in values.items():
            arms[name].append(value)
        commands = lambda episode: [step["command"] for step in episode["trajectory"]]
        changed_paths["hyper_vs_rotated"] += (
            commands(row["hyper"]) != commands(row["rotated"]))
        changed_paths["contrast_success_vs_failure"] += (
            commands(other["good"]) != commands(other["bad"]))
        changed_paths["residual_vs_contrast_success"] += (
            commands(shifted["episode"]) != commands(other["good"]))
        changed_paths["tagged_success_vs_rotated"] += (
            commands(tagged_row["good"]) != commands(tagged_row["rotated"]))
        changed_paths["tagged_success_vs_failure"] += (
            commands(tagged_row["good"]) != commands(tagged_row["bad"]))
        changed_paths["retrieval_vs_original"] += (
            commands(retrieved_row["episode"]) != commands(row["hyper"]))
        families[row["game"].split("/")[2].split("-")[0]].append(i)
        sources = train_review["sources"]
        if (sources[target["rotated_source_index"]]["family"] ==
                sources[target["own_success_source_index"]]["family"]):
            same_family_rotation.append(i)
    if any(sum(values) != bootstrap["summary"].get({
            "base": "base", "hyper": "fresh_hyper",
            "rotated": "rotated_source", "static_B": "fresh_static",
            "static_AB": "full_static"}[name])
           for name, values in arms.items() if name not in
           ("contrast_success", "contrast_failure", "prefix_failure_residual",
            "tagged_success", "tagged_rotated", "tagged_failure",
            "goal_retrieval")):
        raise ValueError("Changed bootstrap scores")
    if (sum(arms["contrast_success"]) != contrast["summary"]["success_source"] or
            sum(arms["contrast_failure"]) != contrast["summary"]["failure_source"] or
            sum(arms["prefix_failure_residual"]) != residual["summary"]["successes"] or
            any(sum(arms["tagged_" + arm]) != tagged["summary"][key]
                for arm, key in (("success", "good"),
                                 ("rotated", "rotated"),
                                 ("failure", "bad"))) or
            sum(arms["goal_retrieval"]) != retrieval["summary"]["successes"]):
        raise ValueError("Changed contrast scores")
    subset = lambda name: [arms[name][i] for i in same_family_rotation]
    return {"protocol": "Official train-stream suffix120, prefix120-only own-trajectory training, equal one-attempt budgets; descriptive because stream has historical exposure",
        "report_sha256": {"bootstrap": file_hash(args.bootstrap),
                          "contrast": file_hash(args.contrast),
                          "residual": file_hash(args.residual),
                          "tagged": file_hash(args.tagged),
                          "retrieval": file_hash(args.retrieval)},
        "n": 120, "scores": {name: sum(x) for name, x in arms.items()},
        "changed_action_sequences": changed_paths,
        "paired": {
            "fresh_hyper_vs_static_B": paired(arms["hyper"], arms["static_B"]),
            "fresh_hyper_vs_static_AB": paired(arms["hyper"], arms["static_AB"]),
            "fresh_hyper_vs_base": paired(arms["hyper"], arms["base"]),
            "fresh_hyper_vs_rotated": paired(arms["hyper"], arms["rotated"]),
            "contrast_success_vs_original_hyper": paired(
                arms["contrast_success"], arms["hyper"]),
            "contrast_success_vs_failure": paired(
                arms["contrast_success"], arms["contrast_failure"]),
            "contrast_success_vs_static_AB": paired(
                arms["contrast_success"], arms["static_AB"]),
            "prefix_failure_residual_vs_fresh_hyper": paired(
                arms["prefix_failure_residual"], arms["hyper"]),
            "prefix_failure_residual_vs_contrast_success": paired(
                arms["prefix_failure_residual"], arms["contrast_success"]),
            "prefix_failure_residual_vs_static_AB": paired(
                arms["prefix_failure_residual"], arms["static_AB"]),
            "tagged_success_vs_failure": paired(
                arms["tagged_success"], arms["tagged_failure"]),
            "tagged_success_vs_rotated": paired(
                arms["tagged_success"], arms["tagged_rotated"]),
            "tagged_success_vs_static_AB": paired(
                arms["tagged_success"], arms["static_AB"]),
            "goal_retrieval_vs_original_hyper": paired(
                arms["goal_retrieval"], arms["hyper"]),
            "goal_retrieval_vs_static_AB": paired(
                arms["goal_retrieval"], arms["static_AB"]),
        },
        "retrieval_source_changes": sum(
            x["retrieved_source_index"] != x["original_source_index"]
            for x in retrieval_review["targets"]),
        "same_family_rotation": {
            "n": len(same_family_rotation),
            "hyper": sum(subset("hyper")),
            "rotated": sum(subset("rotated")),
            "paired": paired(subset("hyper"), subset("rotated")),
            "tagged_success": sum(subset("tagged_success")),
            "tagged_rotated": sum(subset("tagged_rotated")),
            "tagged_paired": paired(subset("tagged_success"),
                                    subset("tagged_rotated"))},
        "family_scores": {name: {arm: sum(values[i] for i in indices)
            for arm, values in arms.items()} for name, indices in
            sorted(families.items())}}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    names = {"bootstrap": "alfworld_bootstrap_own_success_v43c_suffix120_20261010.json",
        "bootstrap_audit": "alfworld_bootstrap_own_success_v43c_suffix120_audit_20261010.json",
        "contrast": "alfworld_success_failure_contrast_v44_suffix120_20261010.json",
        "contrast_audit": "alfworld_success_failure_contrast_v44_suffix120_audit_20261010.json",
        "residual": "alfworld_prefix_failure_residual_v48_suffix120_20261010.json",
        "residual_audit": "alfworld_prefix_failure_residual_v48_suffix120_audit_20261010.json",
        "tagged": "alfworld_outcome_tagged_contrast_v49_suffix120_20261010.json",
        "tagged_audit": "alfworld_outcome_tagged_contrast_v49_suffix120_audit_20261010.json",
        "retrieval": "alfworld_goal_retrieval_v50_suffix120_20261010.json",
        "retrieval_audit": "alfworld_goal_retrieval_v50_suffix120_audit_20261010.json"}
    for name, filename in names.items():
        p.add_argument("--" + name.replace("_", "-"), type=Path,
                       default=Path("results/trajectory_hyperlora") / filename)
    p.add_argument("--bootstrap-review", type=Path, default=Path(
        "data/annotations/alfworld_bootstrap_own_success_v43c_suffix120_20261010.json"))
    p.add_argument("--train-review", type=Path, default=Path(
        "data/annotations/alfworld_bootstrap_own_success_v43b_train240_20261010.json"))
    p.add_argument("--retrieval-review", type=Path, default=Path(
        "data/annotations/alfworld_goal_retrieval_v50_suffix120_20261010.json"))
    p.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_bootstrap_v46_analysis_20261010.json"))
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = analyze(args)
    save(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "family_scores"}),
          flush=True)
