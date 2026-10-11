"""Replay v38 fixed-first-failure episodes and their held residual state."""

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
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import bounded_readout
from ttcl.trajectory_hyperlora.alfworld_first_failure_delta_v38 import (
    bindings, checked_source,
)
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay


def audit(args):
    pairs, old = checked_source(args)
    fixed, targets = bindings(args, pairs, old)
    review = json.loads(args.review.read_text())
    report = json.loads(args.report.read_text())
    if (review["targets"] != targets or
            review["targets_sha256"] != digest(targets) or
            any(review.get(k) != v or report.get(k) != v
                for k, v in fixed.items()) or
            report["review_sha256"] != file_hash(args.review) or
            len(report["rows"]) != 48 or report["failures"]):
        raise ValueError("Changed or incomplete v38 run")
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    state = None
    scores = {"baseline": 0., "failure_mean": 0.,
              "fixed_first_failure": 0.}
    writes = 0
    for i, (pair, binding, source, row) in enumerate(zip(
            pairs, targets, old["rows"], report["rows"], strict=True)):
        if (row["index"] != i or row["game"] != binding["game"] or
                row["input_content_sha256"] != binding["input_content_sha256"] or
                row["baseline_reward"] != source["baseline"]["reward"] or
                row["mean_reward"] != source["arms"]["failure_mean"]["episode"]["reward"] or
                row["state_sha256_before"] != factors_hash(state)):
            raise ValueError(f"Changed target or state at {i}")
        scores["baseline"] += row["baseline_reward"]
        scores["failure_mean"] += row["mean_reward"]
        fields = contextual_source_fields(agent, tokenizer,
            pair["source_records"], args.device, 2048, pooling="both")
        initial = source["baseline"]["initial_observation"]
        expert = source_factors(agent, tokenizer, fields, initial, args.device)
        if row["expert_factor_sha256"] != factors_hash(expert):
            raise ValueError(f"Changed expert source at {i}")
        applied, info = bounded_readout(agent, state, expert)
        if (row["applied_sha256"] != factors_hash(applied) or
                row["read_info"] != info or
                info["applied_effective_norm"] >
                    .25 * info["reference_effective_norm"] + 1e-6):
            raise ValueError(f"Readout changed at {i}")
        episode = row["episode"]
        if (episode["status"] != "complete" or episode["steps"] > 50 or
                episode["initial_observation"] != initial):
            raise ValueError(f"Invalid official episode at {i}")
        replay(args.data_root / row["game"], episode)
        scores["fixed_first_failure"] += episode["reward"]
        if state is None and not bool(episode["reward"]):
            failed, _, original, retained = episode_factor(
                agent, tokenizer, episode,
                argparse.Namespace(device=args.device, context_tokens=2048))
            toward, delta_info = bounded_residual(agent, expert, failed,
                                                   max_relative=.25)
            state = [-x for x in toward]
            if (row["write_kind"] != "first_own_failure" or
                    row["failed_factor_sha256"] != factors_hash(failed) or
                    row["delta_sha256"] != factors_hash(state) or
                    row["delta_info"] != delta_info or
                    row["failed_source_tokens_original"] != original or
                    row["failed_source_tokens_retained"] != retained):
                raise ValueError(f"First failure delta changed at {i}")
            writes += 1
        elif row["write_kind"] != "none":
            raise ValueError("Unexpected later write")
        if row["state_sha256_after"] != factors_hash(state):
            raise ValueError(f"Broken held state after {i}")
    summary = {"n": 48, **scores, "first_failure_writes": writes,
               "failures": 0}
    if summary != report["summary"]:
        raise ValueError("Changed official score summary")
    return {"audit": "passed", "summary": summary,
        "replayed_episodes": 48,
        "report_sha256": file_hash(args.report),
        "review_sha256": file_hash(args.review)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", required=True, choices=("interleave", "reverse"))
    parser.add_argument("--pairs", type=Path, default=Path("data/annotations/alf_reward_router_train72_reviewed_20261007.json"))
    parser.add_argument("--v37-report", required=True, type=Path)
    parser.add_argument("--v37-audit", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--model", type=Path, default=Path("current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--data-root", type=Path, default=Path("ttcl/data/alfworld_delta"))
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
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
