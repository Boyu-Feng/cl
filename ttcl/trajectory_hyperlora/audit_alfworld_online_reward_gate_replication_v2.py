"""Audit a second frozen ALFWorld sequence for reward-gated online memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.audit_alfworld_online_context_memory_v1 import checked_run


def audit(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    factor = checked_run(args, "factor", args.factor_report, args.factor_review)
    gate = checked_run(args, "gate", args.gate_report, args.gate_review)
    paired = []
    for left, right in zip(factor["games"], gate["games"], strict=True):
        if left["game"] != right["game"]:
            raise ValueError("Second-sequence target order changed")
        paired.append({"game": left["game"],
            "base": left["base"]["reward"],
            "persistent_factor": left["online"]["reward"],
            "reward_gated_vector": right["episode"]["reward"],
            "gate_updated": right["memory_updated"]})
    summary = {"n": len(paired),
        "base": sum(x["base"] for x in paired),
        "persistent_factor": sum(x["persistent_factor"] for x in paired),
        "reward_gated_vector": sum(x["reward_gated_vector"] for x in paired),
        "gate_updates": sum(x["gate_updated"] for x in paired),
        "gate_only_vs_base": sum(x["reward_gated_vector"] > x["base"] for x in paired),
        "base_only_vs_gate": sum(x["base"] > x["reward_gated_vector"] for x in paired)}
    result = {"protocol": "Content-bound independent 18-game train-domain replication of fixed reward-gated online source-vector memory; each arm starts empty and follows own trajectories",
        "factor_report_sha256": file_hash(args.factor_report),
        "gate_report_sha256": file_hash(args.gate_report),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "summary": summary, "games": paired}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor-report", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_context_persistent_train_seq0_6_20261007.json"))
    parser.add_argument("--gate-report", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq0_6_20261007.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--plan", type=Path, default=Path(
        "ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--parent-review", type=Path, default=Path(
        "data/annotations/alfworld_online_from_empty_train_seq0_6_reviewed_20261005.json"))
    parser.add_argument("--factor-review", type=Path, default=Path(
        "data/annotations/alfworld_online_context_persistent_train_seq0_6_reviewed_20261007.json"))
    parser.add_argument("--parent-persistent-review", type=Path, default=Path(
        "data/annotations/alfworld_online_context_persistent_train_seq0_6_reviewed_20261007.json"))
    parser.add_argument("--gate-review", type=Path, default=Path(
        "data/annotations/alfworld_online_context_vector_reward_gate_train_seq0_6_reviewed_20261007.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_reward_gate_replication_audited_20261007.json"))
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
