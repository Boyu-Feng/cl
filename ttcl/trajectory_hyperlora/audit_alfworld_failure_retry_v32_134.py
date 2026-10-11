"""Audit old 68/134 first attempts and independently replay new retries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay
from ttcl.trajectory_hyperlora.evaluate_alfworld_failure_retry_v32_134 import (
    ARMS, checked_input,
)


def audit(args):
    _, first, fixed, bindings = checked_input(args)
    review = json.loads(args.review.read_text())
    report = json.loads(args.report.read_text())
    if (review["targets"] != bindings or
            review["targets_sha256"] != digest(bindings) or
            any(review[k] != v for k, v in fixed.items()) or
            report["review_sha256"] != file_hash(args.review) or
            any(report[k] != v for k, v in fixed.items()) or
            tuple(report["arms"]) != ARMS or
            len(report["games"]) != 134 or report["failures"]):
        raise ValueError("Changed or incomplete official retry run")
    first_wins = 0.
    first_failures = 0
    totals = {arm: 0. for arm in ARMS}
    static_matches = 0
    for i, (old, binding, row) in enumerate(zip(
            first, bindings, report["games"], strict=True)):
        episode = old["lora"]
        if (row["index"] != i or row["game"] != binding["game"] or
                row["input_content_sha256"] !=
                    binding["input_content_sha256"] or
                row["first_episode_sha256"] != digest(episode) or
                row["first_reward"] != episode["reward"] or
                binding["game_sha256"] != file_hash(args.data_root /
                    binding["game"])):
            raise ValueError(f"Changed bound first attempt at {i}")
        first_wins += episode["reward"]
        if bool(episode["reward"]):
            if row["retries"]:
                raise ValueError("Retried a first-attempt success")
            for arm in ARMS:
                totals[arm] += 1.
            continue
        first_failures += 1
        info = row["residual_info"]
        if (tuple(row["retries"]) != ARMS or
                row["failed_source_tokens_retained"] > 2048 or
                info["relative_cap"] != .25 or
                info["residual_effective_norm"] >
                    .25 * info["expert_effective_norm"] + 1e-6):
            raise ValueError(f"Changed correction bound at {i}")
        if row["retries"]["static_retry"] != episode:
            raise ValueError(f"Static retry changed old deterministic actor at {i}")
        static_matches += 1
        for arm in ("away_from_failure", "matched_random"):
            retry_episode = row["retries"][arm]
            if (retry_episode["status"] != "complete" or
                    retry_episode["initial_observation"] !=
                    episode["initial_observation"] or
                    retry_episode["steps"] > 50):
                raise ValueError(f"Incomplete retry at {i}:{arm}")
            replay(args.data_root / row["game"], retry_episode)
        for arm in ARMS:
            totals[arm] += row["retries"][arm]["reward"]
    summary = {"n": 134, "first_wins": first_wins,
        "first_failures": first_failures,
        "two_attempt_wins": totals, "failures": 0}
    if summary != report["summary"]:
        raise ValueError("Changed official score summary")
    return {"audit": "passed", "summary": summary,
        "static_retries_identical_to_audited_first": static_matches,
        "new_retries_replayed": first_failures * 2,
        "report_sha256": file_hash(args.report),
        "review_sha256": file_hash(args.review)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--memrl-plan", type=Path, default=Path(
        "results/memrl_comparison/20260928_budgeted/plan.json"))
    parser.add_argument("--source-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train600_reviewed_20261006.json"))
    parser.add_argument("--retry-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_same_game_retry_candidates_20261006.json"))
    parser.add_argument("--retry-review", type=Path, default=Path(
        "data/annotations/alf_same_game_retry_reviewed_20261006.json"))
    parser.add_argument("--all-source-review", type=Path, default=Path(
        "data/annotations/alf_expert_first_attempts_reviewed_20261006.json"))
    parser.add_argument("--target-review", type=Path, default=Path(
        "data/annotations/alf_memrl134_hyperlora_train_sources_reviewed_20261006.json"))
    parser.add_argument("--first-parts", type=Path, nargs=4, default=[Path(
        "results/trajectory_hyperlora/alf_parent_loop2_official_0_34_20261007.json"), Path(
        "results/trajectory_hyperlora/alf_parent_loop2_official_34_34_20261007.json"), Path(
        "results/trajectory_hyperlora/alf_parent_loop2_official_68_33_20261007.json"), Path(
        "results/trajectory_hyperlora/alf_parent_loop2_official_101_33_20261007.json")])
    parser.add_argument("--old-audit", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_parent_loop2_official_audited_20261007.json"))
    parser.add_argument("--dev-report", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_failure_retry_v31b_dev24_20261009.json"))
    parser.add_argument("--dev-audit", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_failure_retry_v31b_dev24_audit_20261009.json"))
    parser.add_argument("--placebo-report", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_failure_retry_placebo_v31c_dev24_20261009.json"))
    parser.add_argument("--placebo-audit", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_failure_retry_placebo_v31c_dev24_audit_20261009.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/alfworld_failure_retry_v32_134_20261009.json"))
    parser.add_argument("--report", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_failure_retry_v32_134_20261009.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result), flush=True)
