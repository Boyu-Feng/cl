"""Audit causal source and factor-state chains in online LoRA updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import (
    checked_targets, factors_hash,
)
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)


def checked_report(args, mode, path, review_path):
    args.mode = mode
    args.review = review_path
    targets = checked_targets(args)
    result = json.loads(path.read_text())
    if (result["review_sha256"] != file_hash(review_path) or
            result["checkpoint_sha256"] != file_hash(args.checkpoint) or
            result["max_steps"] != 30 or result["max_new_tokens"] != 64 or
            result["source_field_token_limit"] != 40 or
            result.get("update_mode", "sum") != mode or result["failures"] or
            len(result["games"]) != len(targets)):
        raise ValueError(f"Changed or incomplete {mode} run")
    previous_factors = factors_hash(None)
    prior_episodes = []
    increments = 0
    for target, row in zip(targets, result["games"], strict=True):
        ep = row["episode"]
        if (row["game"] != target["game"] or
                row["input_content_sha256"] != target["input_content_sha256"] or
                row["prior_episode_count"] != len(prior_episodes) or
                row["prior_episodes_sha256"] != digest(prior_episodes) or
                row["factor_sha256_before"] != previous_factors or
                ep["status"] != "complete" or ep["steps"] > 30 or
                ep["steps"] != len(ep["trajectory"]) or
                ep["invalid_commands"] != 0 or
                ep["reward"] != float(ep["termination"] == "success")):
            raise ValueError(f"Changed input, factor chain, or reward: {target['game']}")
        records = records_from_episode(ep)
        if (row["new_records_sha256"] != digest(records) or
                row["increment_applied"] != (len(records) >= 2)):
            raise ValueError("Own feedback or update flag changed")
        increments += int(row["increment_applied"])
        if row.get("increment_count_after", increments) != increments:
            raise ValueError("Increment count changed")
        prior_episodes.append({"game": row["game"],
                               "reward": ep["reward"], "records": records})
        previous_factors = row["factor_sha256_after"]
    return result


def audit(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    control = json.loads(args.control.read_text())
    if (control["checkpoint_sha256"] != file_hash(args.checkpoint) or
            control["max_steps"] != 30 or
            control["max_new_tokens"] != 64 or
            control["source_field_token_limit"] != 40):
        raise ValueError("Historical all-history control changed")
    reports = {"mean": checked_report(args, "mean", args.mean_report,
                                       args.mean_review)}
    if not args.skip_sum:
        reports["sum"] = checked_report(args, "sum", args.sum_report,
                                         args.sum_review)
    old = control["games"]
    if any([r["game"] for r in reports[mode]["games"]] !=
           [r["game"] for r in old] for mode in reports):
        raise ValueError("Control and cumulative task sequences differ")
    for mode in reports:
        if reports[mode]["games"][0]["episode"]["trajectory"] != \
                old[0]["online"]["trajectory"]:
            raise ValueError("Empty-memory first episode differs from control")
    rows = []
    for index, item in enumerate(old):
        rows.append({"index": index, "game": item["game"],
            "base": item["base"]["reward"],
            "all_history": item["online"]["reward"],
            "factor_mean": reports["mean"]["games"][index]["episode"]["reward"]})
        if "sum" in reports:
            rows[-1]["factor_sum"] = reports["sum"]["games"][index]["episode"]["reward"]
    summary = {"n": len(rows), **{key: sum(row[key] for row in rows)
                for key in ("base", "all_history", "factor_mean")},
        "mean_only_vs_all_history": sum(row["factor_mean"] > row["all_history"]
                                        for row in rows),
        "all_history_only_vs_mean": sum(row["factor_mean"] < row["all_history"]
                                        for row in rows),
        "mean_final_factor_l2": reports["mean"]["games"][-1]["factor_l2_after"]}
    if "sum" in reports:
        summary["factor_sum"] = sum(row["factor_sum"] for row in rows)
        summary["sum_final_factor_l2"] = reports["sum"]["games"][-1]["factor_l2_after"]
    output = {"protocol": "Content-bound audit of frozen, from-empty ALFWorld online memory updates on the identical 18 training games; scores use official won, each arm follows its own prior actions",
        "control_sha256": file_hash(args.control),
        "mean_sha256": file_hash(args.mean_report),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "summary": summary, "games": rows}
    if "sum" in reports:
        output["sum_sha256"] = file_hash(args.sum_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_from_empty_seq0_6_statusfirst_20261005.json"))
    parser.add_argument("--sum-report", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_lora_additive_train_seq0_6_20261007.json"))
    parser.add_argument("--mean-report", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_lora_mean_train_seq0_6_20261007.json"))
    parser.add_argument("--sum-review", type=Path, default=Path(
        "data/annotations/alfworld_online_lora_additive_train_seq0_6_reviewed_20261007.json"))
    parser.add_argument("--mean-review", type=Path, default=Path(
        "data/annotations/alfworld_online_lora_mean_train_seq0_6_reviewed_20261007.json"))
    parser.add_argument("--parent-review", type=Path, default=Path(
        "data/annotations/alfworld_online_from_empty_train_seq0_6_reviewed_20261005.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_warmstart_20261005/seed42_v2.pt"))
    parser.add_argument("--plan", type=Path, default=Path(
        "ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_lora_accumulate_audited_20261007.json"))
    parser.add_argument("--skip-sum", action="store_true")
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
