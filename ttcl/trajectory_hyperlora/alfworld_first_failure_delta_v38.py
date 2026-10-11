"""Frozen-first-failure control for the reviewed v37 online cohorts."""

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
from ttcl.trajectory_hyperlora.evaluate_alfworld_failure_retry_v31b import retry, source_factors
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import bounded_readout, save
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v37 import ordered_pairs


def checked_source(args):
    pairs = ordered_pairs(args)
    old = json.loads(args.v37_report.read_text())
    audit = json.loads(args.v37_audit.read_text())
    if (old.get("order") != args.order or len(old["rows"]) != 48 or
            old["failures"] or audit.get("audit") != "passed" or
            audit.get("report_sha256") != file_hash(args.v37_report) or
            old["checkpoint_sha256"] != file_hash(args.checkpoint) or
            old["actor_sha256"] != file_hash(Path(__file__).with_name("alfworld_zero_shot_probe.py"))):
        raise ValueError("Changed or unaudited v37 cohort")
    return pairs, old


def bindings(args, pairs, old):
    fixed = {"method": "v38 first own failure residual fixed forever",
        "order": args.order, "pairs_sha256": file_hash(args.pairs),
        "v37_report_sha256": file_hash(args.v37_report),
        "v37_audit_sha256": file_hash(args.v37_audit),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "actor_sha256": old["actor_sha256"],
        "runner_sha256": file_hash(Path(__file__)),
        "selected_direction": "away_from_failure",
        "max_relative": .25, "max_steps": 50, "max_new_tokens": 64,
        "actor_history_turns": 2, "loop_guard_max": 2}
    targets = []
    for i, (pair, prior) in enumerate(zip(pairs, old["rows"], strict=True)):
        if (prior["game"] != pair["target_game"] or
                prior["baseline"]["status"] != "complete" or
                prior["arms"]["failure_mean"]["episode"]["status"] != "complete"):
            raise ValueError("Changed v37 task or episode")
        content = {"index": i, "game": pair["target_game"],
            "game_sha256": file_hash(args.data_root / pair["target_game"]),
            "expert_source_sha256": pair["source_records_sha256"],
            "baseline_episode_sha256": digest(prior["baseline"]),
            "mean_episode_sha256": digest(prior["arms"]["failure_mean"]["episode"]),
            "v37_report_sha256": fixed["v37_report_sha256"],
            "checkpoint_sha256": fixed["checkpoint_sha256"],
            "runner_sha256": fixed["runner_sha256"]}
        if content["game_sha256"] != pair["target_game_sha256"]:
            raise ValueError("Changed task content")
        targets.append({**content, "input_content_sha256": digest(content),
                        "reviewed_target": True})
    return fixed, targets


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    pairs, old = checked_source(args)
    fixed, targets = bindings(args, pairs, old)
    save(args.review, {"protocol": "From empty residual, write first own official failure correction once and hold it across remaining tasks; same reviewed 48 train tasks/order/actor/budget as audited v37; no further failure writes",
        **fixed, "targets": targets, "targets_sha256": digest(targets)})
    print(json.dumps({"order": args.order, "reviewed_targets": len(targets)}))


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    pairs, old = checked_source(args)
    fixed, targets = bindings(args, pairs, old)
    review = json.loads(args.review.read_text())
    if (review.get("targets") != targets or
            review.get("targets_sha256") != digest(targets) or
            any(review.get(k) != v for k, v in fixed.items())):
        raise ValueError("Changed v38 review")
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    state = None
    report = {"protocol": review["protocol"], "review_sha256": file_hash(args.review),
        **fixed, "rows": [], "failures": []}
    for i, (pair, binding, source) in enumerate(zip(pairs, targets,
                                                     old["rows"], strict=True)):
        fields = contextual_source_fields(agent, tokenizer,
            pair["source_records"], args.device, 2048, pooling="both")
        initial = source["baseline"]["initial_observation"]
        expert = source_factors(agent, tokenizer, fields, initial, args.device)
        applied, read_info = bounded_readout(agent, state, expert)
        episode = retry(agent, tokenizer, args.data_root / binding["game"],
                        fields, args.device, applied)
        row = {"index": i, "game": binding["game"],
            "input_content_sha256": binding["input_content_sha256"],
            "baseline_reward": source["baseline"]["reward"],
            "mean_reward": source["arms"]["failure_mean"]["episode"]["reward"],
            "expert_factor_sha256": factors_hash(expert),
            "state_sha256_before": factors_hash(state),
            "applied_sha256": factors_hash(applied),
            "read_info": read_info, "episode": episode,
            "write_kind": "none"}
        if (episode["status"] != "complete" or
                episode["initial_observation"] != initial):
            report["failures"].append({"index": i, "status": episode["status"]})
        elif state is None and not bool(episode["reward"]):
            failed, _, original, retained = episode_factor(
                agent, tokenizer, episode,
                argparse.Namespace(device=args.device, context_tokens=2048))
            toward, info = bounded_residual(agent, expert, failed, max_relative=.25)
            state = [-x for x in toward]
            row.update(write_kind="first_own_failure",
                failed_factor_sha256=factors_hash(failed),
                delta_sha256=factors_hash(state), delta_info=info,
                failed_source_tokens_original=original,
                failed_source_tokens_retained=retained)
        row["state_sha256_after"] = factors_hash(state)
        report["rows"].append(row)
        report["summary"] = {"n": len(report["rows"]),
            "baseline": sum(x["baseline_reward"] for x in report["rows"]),
            "failure_mean": sum(x["mean_reward"] for x in report["rows"]),
            "fixed_first_failure": sum(x["episode"]["reward"] for x in report["rows"]),
            "first_failure_writes": sum(x["write_kind"] == "first_own_failure"
                                        for x in report["rows"]),
            "failures": len(report["failures"])}
        save(args.output, report)
        print(json.dumps({"order": args.order, **report["summary"]}), flush=True)
        if report["failures"]:
            raise RuntimeError("Incomplete first-failure control episode")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--order", required=True, choices=("interleave", "reverse"))
    parser.add_argument("--pairs", type=Path, default=Path("data/annotations/alf_reward_router_train72_reviewed_20261007.json"))
    parser.add_argument("--v37-report", required=True, type=Path)
    parser.add_argument("--v37-audit", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--model", type=Path, default=Path("current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--data-root", type=Path, default=Path("ttcl/data/alfworld_delta"))
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.65)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else evaluate)(args)


if __name__ == "__main__":
    main()
