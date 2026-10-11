"""Independently replay v31 retries and check frozen first attempts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay
from ttcl.trajectory_hyperlora.evaluate_alfworld_failure_retry_v31b import (
    ARMS, checked_input,
)


def audit(args):
    pairs, first_report, fixed, targets = checked_input(args)
    review = json.loads(args.review.read_text())
    report = json.loads(args.report.read_text())
    if (review["targets"] != targets or
            review["targets_sha256"] != digest(targets) or
            any(review[key] != value for key, value in fixed.items()) or
            report["review_sha256"] != file_hash(args.review) or
            any(report[key] != value for key, value in fixed.items()) or
            tuple(report["arms"]) != ARMS or
            len(report["games"]) != 24 or report["failures"]):
        raise ValueError("Changed or incomplete frozen v31b run")
    first_wins = 0.
    first_failures = 0
    totals = {arm: 0. for arm in ARMS}
    static_matches = 0
    for i, (pair, original, binding, row) in enumerate(zip(
            pairs, first_report["games"], targets, report["games"],
            strict=True)):
        first = original["new"]
        if (row["index"] != i or row["game"] != binding["game"] or
                row["input_content_sha256"] !=
                    binding["input_content_sha256"] or
                row["first_episode_sha256"] != digest(first) or
                row["first_reward"] != first["reward"] or
                binding["game_sha256"] != file_hash(args.data_root /
                    pair["target_game"])):
            raise ValueError(f"Changed first attempt or target at {i}")
        replay(args.data_root / row["game"], first)
        first_wins += first["reward"]
        if bool(first["reward"]):
            if row["retries"]:
                raise ValueError("Retried a successful first attempt")
            for arm in ARMS:
                totals[arm] += first["reward"]
            continue
        first_failures += 1
        if (tuple(row["retries"]) != ARMS or
                row["failed_source_tokens_retained"] > 2048 or
                row["residual_info"]["relative_cap"] != args.max_relative or
                row["residual_info"]["residual_effective_norm"] >
                    args.max_relative * row["residual_info"][
                        "expert_effective_norm"] + 1e-6):
            raise ValueError(f"Changed failure correction at {i}")
        for arm in ARMS:
            episode = row["retries"][arm]
            if (episode["status"] != "complete" or
                    episode["initial_observation"] !=
                    first["initial_observation"] or
                    episode["steps"] > 50):
                raise ValueError(f"Invalid retry at {i}:{arm}")
            replay(args.data_root / row["game"], episode)
            totals[arm] += episode["reward"]
        if row["retries"]["static_retry"] == first:
            static_matches += 1
    summary = {"n": 24, "first_wins": first_wins,
        "first_failures": first_failures,
        "two_attempt_wins": totals, "failures": 0}
    if summary != report["summary"]:
        raise ValueError("Changed official score summary")
    return {"audit": "passed", "summary": summary,
        "static_retries_identical_to_first": static_matches,
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
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/alfworld_failure_retry_v31b_dev24_20261009.json"))
    parser.add_argument("--report", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_failure_retry_v31b_dev24_20261009.json"))
    parser.add_argument("--max-relative", type=float, default=.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result), flush=True)
