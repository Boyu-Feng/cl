"""Replay the 13 random-direction development retry controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay
from ttcl.trajectory_hyperlora.evaluate_alfworld_failure_retry_placebo_v31c import inputs


def audit(args):
    _, _, comparison, fixed, targets = inputs(args)
    review = json.loads(args.review.read_text())
    report = json.loads(args.report.read_text())
    if (review["targets"] != targets or
            review["targets_sha256"] != digest(targets) or
            any(review[k] != v for k, v in fixed.items()) or
            report["review_sha256"] != file_hash(args.review) or
            any(report[k] != v for k, v in fixed.items()) or
            len(report["games"]) != 13 or report["failures"]):
        raise ValueError("Changed or incomplete placebo result")
    random_wins = true_wins = 0.
    for row, target in zip(report["games"], targets, strict=True):
        i = target["index"]
        if (row["index"] != i or row["game"] != target["game"] or
                row["input_content_sha256"] !=
                    target["input_content_sha256"] or
                target["game_sha256"] != file_hash(args.data_root /
                    target["game"]) or
                not target["reviewed_target"] or
                row["retry"]["status"] != "complete" or
                row["retry"]["steps"] > 50):
            raise ValueError(f"Changed placebo target at {i}")
        replay(args.data_root / row["game"], row["retry"])
        random_wins += row["retry"]["reward"]
        true_wins += comparison["games"][i]["retries"][
            "away_from_failure"]["reward"]
    summary = {"failed_targets": 13, "random_rescues": random_wins,
               "true_failure_rescues": true_wins, "failures": 0}
    if report["summary"] != summary:
        raise ValueError("Changed placebo summary")
    return {"audit": "passed", "summary": summary,
        "report_sha256": file_hash(args.report),
        "review_sha256": file_hash(args.review)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed-pairs", type=Path, default=Path(
        "data/annotations/alf_reward_router_train72_reviewed_20261007.json"))
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train600_reviewed_20261006.json"))
    parser.add_argument("--retry-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_same_game_retry_candidates_20261006.json"))
    parser.add_argument("--retry-review", type=Path, default=Path(
        "data/annotations/alf_same_game_retry_reviewed_20261006.json"))
    parser.add_argument("--all-source-review", type=Path, default=Path(
        "data/annotations/alf_expert_first_attempts_reviewed_20261006.json"))
    parser.add_argument("--first-report", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_parent_loop2_dev24_20261007.json"))
    parser.add_argument("--comparison", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_failure_retry_v31b_dev24_20261009.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/alfworld_failure_retry_placebo_v31c_dev24_20261009.json"))
    parser.add_argument("--report", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_failure_retry_placebo_v31c_dev24_20261009.json"))
    parser.add_argument("--max-relative", type=float, default=.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result), flush=True)
