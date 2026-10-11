"""Replay official episodes and reconstruct both persistent failure states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_failure_retry_residual_v31b import bounded_residual
from ttcl.trajectory_hyperlora.alfworld_success_gated_factors_v27 import episode_factor
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_source_fields
from ttcl.trajectory_hyperlora.evaluate_alfworld_failure_retry_v31b import source_factors
from ttcl.trajectory_hyperlora.alfworld_retry_delta_next_v35 import read_inputs
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import (
    ARMS, bound_targets, bounded_readout, update_state,
)
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay


def audit(args):
    pairs, first, prior = read_inputs(args)
    fixed, targets = bound_targets(args, pairs, first, prior)
    review = json.loads(args.review.read_text())
    report = json.loads(args.report.read_text())
    if (review["targets"] != targets or
            review["targets_sha256"] != digest(targets) or
            any(review.get(k) != v or report.get(k) != v
                for k, v in fixed.items()) or
            report["review_sha256"] != file_hash(args.review) or
            len(report["rows"]) != 24 or report["failures"]):
        raise ValueError("Changed or incomplete v36 run")
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    state = {arm: None for arm in ARMS}
    count = {arm: 0 for arm in ARMS}
    scores = {arm: 0. for arm in ARMS}
    baseline = 0.
    diverged_states = 0
    for i, (pair, old, binding, row) in enumerate(zip(
            pairs, first["games"], targets, report["rows"], strict=True)):
        if (row["index"] != i or row["game"] != binding["game"] or
                row["input_content_sha256"] != binding["input_content_sha256"] or
                row["baseline_episode_sha256"] != digest(old["new"]) or
                row["baseline_reward"] != old["new"]["reward"]):
            raise ValueError(f"Changed baseline or target at {i}")
        baseline += row["baseline_reward"]
        fields = contextual_source_fields(agent, tokenizer,
            pair["source_records"], args.device, 2048, pooling="both")
        initial = old["new"]["initial_observation"]
        expert = source_factors(agent, tokenizer, fields, initial, args.device)
        if row["expert_factor_sha256"] != factors_hash(expert):
            raise ValueError(f"Changed expert source at {i}")
        if set(row["arms"]) != set(ARMS):
            raise ValueError("Changed arm set")
        for arm in ARMS:
            entry = row["arms"][arm]
            if (entry["state_sha256_before"] != factors_hash(state[arm]) or
                    entry["failure_updates_before"] != count[arm]):
                raise ValueError(f"Broken state chain before {i}:{arm}")
            applied, info = bounded_readout(agent, state[arm], expert)
            if (entry["applied_sha256"] != factors_hash(applied) or
                    entry["read_info"] != info or
                    info["applied_effective_norm"] >
                        .25 * info["reference_effective_norm"] + 1e-6):
                raise ValueError(f"Readout mismatch or norm overflow at {i}:{arm}")
            episode = entry["episode"]
            if (episode["status"] != "complete" or episode["steps"] > 50 or
                    episode["initial_observation"] != initial):
                raise ValueError(f"Invalid official episode at {i}:{arm}")
            replay(args.data_root / binding["game"], episode)
            scores[arm] += episode["reward"]
            if bool(episode["reward"]):
                if entry["write_kind"] != "none":
                    raise ValueError("Wrote successful episode")
            else:
                failed, _, original, retained = episode_factor(
                    agent, tokenizer, episode,
                    argparse.Namespace(device=args.device, context_tokens=2048))
                toward, delta_info = bounded_residual(
                    agent, expert, failed, max_relative=.25)
                delta = [-x for x in toward]
                if (entry["write_kind"] != "own_failure" or
                        entry["failed_factor_sha256"] != factors_hash(failed) or
                        entry["delta_sha256"] != factors_hash(delta) or
                        entry["delta_info"] != delta_info or
                        entry["failed_source_tokens_original"] != original or
                        entry["failed_source_tokens_retained"] != retained):
                    raise ValueError(f"Failure delta changed at {i}:{arm}")
                state[arm] = update_state(state[arm], delta, count[arm], arm)
                count[arm] += 1
            if (entry["state_sha256_after"] != factors_hash(state[arm]) or
                    entry["failure_updates_after"] != count[arm]):
                raise ValueError(f"Broken state chain after {i}:{arm}")
        if (row["arms"][ARMS[0]]["state_sha256_after"] !=
                row["arms"][ARMS[1]]["state_sha256_after"]):
            diverged_states += 1
    summary = {"n": 24, "baseline": baseline, **scores,
        "failure_updates": count, "failures": 0}
    if report["summary"] != summary:
        raise ValueError("Changed official summary")
    return {"audit": "passed", "summary": summary,
        "replayed_episodes": 48, "positions_with_different_state": diverged_states,
        "report_sha256": file_hash(args.report),
        "review_sha256": file_hash(args.review)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=Path("data/annotations/alf_reward_router_train72_reviewed_20261007.json"))
    parser.add_argument("--first-report", type=Path, default=Path("results/trajectory_hyperlora/alf_parent_loop2_dev24_20261007.json"))
    parser.add_argument("--retry-report", type=Path, default=Path("results/trajectory_hyperlora/alfworld_failure_retry_v31b_dev24_20261009.json"))
    parser.add_argument("--retry-audit", type=Path, default=Path("results/trajectory_hyperlora/alfworld_failure_retry_v31b_dev24_audit_20261009.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--model", type=Path, default=Path("current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--data-root", type=Path, default=Path("ttcl/data/alfworld_delta"))
    parser.add_argument("--review", type=Path, default=Path("data/annotations/alfworld_persistent_failure_delta_v36_dev24_20261009.json"))
    parser.add_argument("--report", type=Path, default=Path("results/trajectory_hyperlora/alfworld_persistent_failure_delta_v36_dev24_20261009.json"))
    parser.add_argument("--output", type=Path, default=Path("results/trajectory_hyperlora/alfworld_persistent_failure_delta_v36_dev24_audit_20261009.json"))
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--gpu-fraction", type=float, default=.65)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
