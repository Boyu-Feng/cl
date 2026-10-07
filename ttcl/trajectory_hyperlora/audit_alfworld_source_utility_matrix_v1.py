"""Audit the cross-source ALFWorld utility matrix and its selection ceiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_source_utility_matrix_v1 import (
    checked_pairs, SOURCE_INDICES, TARGET_INDICES,
)


def audit(args):
    if args.audit.exists():
        raise FileExistsError(args.audit)
    expected = checked_pairs(args)
    result = json.loads(args.output.read_text())
    if (result["review_sha256"] != file_hash(args.review) or
            result["checkpoint_sha256"] != file_hash(args.checkpoint) or
            result["max_steps"] != 50 or result["max_new_tokens"] != 64 or
            result["actor_history_turns"] != 2 or
            result["loop_guard_max"] != 2 or
            result["context_tokens"] != 2048 or
            result["failures"] or len(result["pairs"]) != len(expected)):
        raise ValueError("Changed or incomplete utility matrix")
    rows = []
    for prior, row in zip(expected, result["pairs"], strict=True):
        episode = row["episode"]
        if (row["source_index"] != prior["source_index"] or
                row["target_index"] != prior["target_index"] or
                row["source_game"] != prior["source_game"] or
                row["target_game"] != prior["target_game"] or
                row["source_family"] != prior["source_family"] or
                row["target_family"] != prior["target_family"] or
                row["input_content_sha256"] != prior["input_content_sha256"] or
                row["target_base_reward"] != prior["target_base_reward"] or
                row["source_tokens_retained"] > 2048 or
                episode["status"] != "complete" or episode["steps"] > 50 or
                episode["steps"] != len(episode["trajectory"]) or
                episode["invalid_commands"] != 0 or
                episode["reward"] != float(episode["termination"] == "success")):
            raise ValueError("Changed source, target, budget, or reward")
        rows.append({"source_index": row["source_index"],
            "target_index": row["target_index"],
            "source_family": row["source_family"],
            "target_family": row["target_family"],
            "base": row["target_base_reward"],
            "lora": episode["reward"]})
    source_summary = {str(index): {
        "family": next(x["source_family"] for x in rows
                       if x["source_index"] == index),
        "base": sum(x["base"] for x in rows if x["source_index"] == index),
        "lora": sum(x["lora"] for x in rows if x["source_index"] == index)}
        for index in SOURCE_INDICES}
    target_rows = [[x for x in rows if x["target_index"] == index]
                   for index in TARGET_INDICES]
    static_best = max(item["lora"] for item in source_summary.values())
    oracle_best = sum(max(x["lora"] for x in group)
                      for group in target_rows)
    summary = {"pairs": len(rows), "source_summary": source_summary,
        "targets": len(target_rows),
        "targets_with_source_dependent_reward": sum(len({x["lora"] for x in group}) > 1
                                                  for group in target_rows),
        "best_source_oracle_success": oracle_best,
        "best_static_source_success": static_best,
        "source_selection_headroom": oracle_best - static_best,
        "base_success": sum(group[0]["base"] for group in target_rows),
        "matched_family_pairs": sum(x["source_family"] == x["target_family"]
                                    for x in rows),
        "matched_family_success": sum(x["lora"] for x in rows
                                      if x["source_family"] == x["target_family"]),
        "other_family_success": sum(x["lora"] for x in rows
                                    if x["source_family"] != x["target_family"])}
    if result["summary"] != {"n": len(rows),
                              "success": sum(x["lora"] for x in rows),
                              "failures": 0}:
        raise ValueError("Utility matrix aggregate changed")
    audited = {"protocol": "Content-bound audit of own-success trajectory LoRA transfer matrix on disjoint training tasks; family labels for posthoc analysis only",
        "raw_report_sha256": file_hash(args.output),
        "review_sha256": file_hash(args.review),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "summary": summary, "pairs": rows}
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audited, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--source-report", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq0_6_20261007.json"))
    parser.add_argument("--target-report", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_context_persistent_train_seq6_6_20261007.json"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/alf_own_success_source_utility_matrix48_reviewed_20261007.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_own_success_source_utility_matrix48_20261007.json"))
    parser.add_argument("--audit", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_own_success_source_utility_matrix48_audited_20261007.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
