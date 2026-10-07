"""Audit both from-empty contextual online memories on one reviewed sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import (
    checked_targets as checked_factor_targets,
)
from ttcl.trajectory_hyperlora.alfworld_online_context_vector_mean_v1 import (
    checked_targets as checked_vector_targets,
)
from ttcl.trajectory_hyperlora.alfworld_online_context_vector_reward_gate_v2 import (
    checked_targets as checked_gate_targets,
)


def checked_run(args, kind, path, review_path):
    if kind == "factor":
        args.review = review_path
        targets = checked_factor_targets(args)
        review_key = "review_sha256"
    else:
        args.vector_review = review_path
        targets = (checked_gate_targets(args) if kind == "gate" else
                   checked_vector_targets(args))
        review_key = "vector_review_sha256"
    result = json.loads(path.read_text())
    if (result[review_key] != file_hash(review_path) or
            result["checkpoint_sha256"] != file_hash(args.checkpoint) or
            result["max_steps"] != 50 or result["max_new_tokens"] != 64 or
            result["actor_history_turns"] != 2 or
            result["loop_guard_max"] != 2 or
            result["context_tokens"] != 2048 or
            result["failures"] or len(result["games"]) != len(targets)):
        raise ValueError(f"Changed or incomplete {kind} online result")
    prior = []
    state_hash = None
    updates = 0
    for target, row in zip(targets, result["games"], strict=True):
        episode = row["online"] if kind == "factor" else row["episode"]
        before_key = ("factor_sha256_before" if kind == "factor" else
                      "source_vector_sha256_before")
        after_key = ("factor_sha256_after" if kind == "factor" else
                     "source_vector_sha256_after")
        if (row["game"] != target["game"] or
                row["input_content_sha256"] != target["input_content_sha256"] or
                row["prior_episode_count"] != len(prior) or
                row["prior_episodes_sha256"] != digest(prior) or
                (state_hash is not None and row[before_key] != state_hash) or
                episode["status"] != "complete" or episode["steps"] > 50 or
                episode["steps"] != len(episode["trajectory"]) or
                episode["invalid_commands"] != 0 or
                episode["reward"] != float(episode["termination"] == "success") or
                row.get("source_tokens_retained", 0) > 2048 or
                row["update_count_after"] != updates +
                    (int(bool(episode["reward"])) if kind == "gate" else 1) or
                (kind == "gate" and (
                    row["memory_updated"] != bool(episode["reward"]) or
                    (not episode["reward"] and row[after_key] != row[before_key])))):
            raise ValueError(f"Changed target, source, state, or reward: {row['game']}")
        records = records_from_episode(episode)
        if row["new_records_sha256"] != digest(records):
            raise ValueError("Own trajectory records changed")
        prior.append({"game": row["game"],
                      "reward": episode["reward"], "records": records})
        state_hash = row[after_key]
        updates = row["update_count_after"]
    return result


def audit(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    factor = checked_run(args, "factor", args.factor_report,
                         args.parent_persistent_review)
    vector = checked_run(args, "vector", args.vector_report,
                         args.vector_review)
    gate = checked_run(args, "gate", args.gate_report,
                       args.gate_review)
    if [x["game"] for x in factor["games"]] != \
            [x["game"] for x in vector["games"]] or \
            [x["game"] for x in factor["games"]] != \
            [x["game"] for x in gate["games"]]:
        raise ValueError("Online task sequence changed between methods")
    paired = []
    for left, right, gated in zip(factor["games"], vector["games"],
                                  gate["games"], strict=True):
        def first_changed(a, b):
            for index, (turn_a, turn_b) in enumerate(zip(a, b)):
                if turn_a != turn_b:
                    return index
            return min(len(a), len(b)) if len(a) != len(b) else None

        base_steps = left["base"]["trajectory"]
        factor_steps = left["online"]["trajectory"]
        vector_steps = right["episode"]["trajectory"]
        gate_steps = gated["episode"]["trajectory"]
        paired.append({"game": left["game"],
            "base": left["base"]["reward"],
            "persistent_factor": left["online"]["reward"],
            "source_vector": right["episode"]["reward"],
            "reward_gated_vector": gated["episode"]["reward"],
            "factor_vs_base_first_changed_turn": first_changed(factor_steps, base_steps),
            "factor_vs_vector_first_changed_turn": first_changed(factor_steps, vector_steps),
            "vector_vs_gate_first_changed_turn": first_changed(vector_steps, gate_steps),
            "same_trajectory": factor_steps == vector_steps})
    keys = ("base", "persistent_factor", "source_vector", "reward_gated_vector")
    summary = {"n": len(paired),
        **{key: sum(row[key] for row in paired) for key in keys},
        "same_trajectory": sum(row["same_trajectory"] for row in paired),
        "factor_changes_base_trajectory": sum(
            row["factor_vs_base_first_changed_turn"] is not None for row in paired),
        "factor_only": sum(row["persistent_factor"] > row["source_vector"]
                           for row in paired),
        "vector_only": sum(row["source_vector"] > row["persistent_factor"]
                           for row in paired)}
    result = {"protocol": "Content-bound audit of frozen contextual hyper-LoRA online persistent-B, persistent-source-vector and reward-gated source-vector memory on 18 train games; each arm follows its own prior trajectory",
        "factor_report_sha256": file_hash(args.factor_report),
        "vector_report_sha256": file_hash(args.vector_report),
        "gate_report_sha256": file_hash(args.gate_report),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "summary": summary, "games": paired}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor-report", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_context_persistent_train_seq6_6_20261007.json"))
    parser.add_argument("--vector-report", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_context_vector_mean_train_seq6_6_20261007.json"))
    parser.add_argument("--gate-report", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq6_6_20261007.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--plan", type=Path, default=Path(
        "ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--parent-review", type=Path, default=Path(
        "data/annotations/alfworld_online_from_empty_train_seq6_6_reviewed_20261007.json"))
    parser.add_argument("--parent-persistent-review", type=Path, default=Path(
        "data/annotations/alfworld_online_context_persistent_train_seq6_6_reviewed_20261007.json"))
    parser.add_argument("--vector-review", type=Path, default=Path(
        "data/annotations/alfworld_online_context_vector_mean_train_seq6_6_reviewed_20261007.json"))
    parser.add_argument("--gate-review", type=Path, default=Path(
        "data/annotations/alfworld_online_context_vector_reward_gate_train_seq6_6_reviewed_20261007.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_context_memory_audited_20261007.json"))
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
