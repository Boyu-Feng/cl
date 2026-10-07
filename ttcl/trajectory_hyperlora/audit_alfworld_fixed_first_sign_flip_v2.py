"""Audit the first-success LoRA sign-flip control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_online_context_vector_mean_v1 import vector_hash
from ttcl.trajectory_hyperlora.alfworld_online_fixed_first_sign_flip_v2 import (
    source_binding,
)


def audit(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    reference, targets, _, content = source_binding(args)
    review = json.loads(args.review.read_text())
    if (not review["reviewed_source"] or
            review["target_count"] != len(targets) or
            review["input_content_sha256"] != digest(content)):
        raise ValueError("Changed control review")
    result = json.loads(args.report.read_text())
    if (result["review_sha256"] != file_hash(args.review) or
            result["adapter_sign"] != -1 or
            result["reference_sha256"] != file_hash(args.reference) or
            result["target_review_sha256"] != file_hash(args.target_review) or
            result["checkpoint_sha256"] != file_hash(args.checkpoint) or
            result["source_index"] != args.source_index or
            result["max_steps"] != 50 or result["max_new_tokens"] != 64 or
            result["actor_history_turns"] != 2 or result["loop_guard_max"] != 2 or
            result["context_tokens"] != 2048 or result["failures"] or
            len(result["games"]) != len(targets)):
        raise ValueError("Changed or incomplete fixed-first-source result")
    empty_hash = vector_hash(None)
    fixed_hash = reference["games"][args.source_index][
        "source_vector_sha256_after"]
    rows = []
    for index, (target, current, original) in enumerate(zip(
            targets, result["games"], reference["games"], strict=True)):
        episode = current["episode"]
        if (current["game"] != target["game"] or
                current["family"] != target["family"] or
                current["input_content_sha256"] != target["input_content_sha256"] or
                current["fixed_source_vector_sha256"] != (
                    empty_hash if index <= args.source_index else fixed_hash) or
                episode["status"] != "complete" or episode["steps"] > 50 or
                episode["steps"] != len(episode["trajectory"]) or
                episode["invalid_commands"] != 0 or
                episode["reward"] != float(episode["termination"] == "success") or
                (index <= args.source_index and
                 episode["trajectory"] != original["online"]["trajectory"])):
            raise ValueError(f"Changed fixed-source game: {target['game']}")
        rows.append({"game": target["game"], "family": target["family"],
            "base": original["base"]["reward"],
            "online": original["online"]["reward"],
            "fixed_first": episode["reward"],
            "same_online_trajectory": episode["trajectory"] ==
                                      original["online"]["trajectory"]})
    summary = {"n": len(rows),
        "base": sum(x["base"] for x in rows),
        "online": sum(x["online"] for x in rows),
        "fixed_first": sum(x["fixed_first"] for x in rows),
        "same_online_trajectory": sum(x["same_online_trajectory"] for x in rows),
        "online_only_vs_fixed": sum(x["online"] > x["fixed_first"] for x in rows),
        "fixed_only_vs_online": sum(x["fixed_first"] > x["online"] for x in rows)}
    if result["summary"] != {"n": len(rows),
                              "success": summary["fixed_first"],
                              "failures": 0}:
        raise ValueError("Fixed-source aggregate changed")
    audited = {"protocol": "Content-bound audit of sign-flipped first-success LoRA versus online accumulation, on same 36 official valid_unseen games",
        "report_sha256": file_hash(args.report),
        "reference_sha256": file_hash(args.reference),
        "summary": summary, "games": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audited, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--parent-review", type=Path, default=Path(
        "data/annotations/alf_memrl134_hyperlora_train_sources_reviewed_20261006.json"))
    parser.add_argument("--target-review", type=Path, default=Path(
        "data/annotations/alf_online_reward_gate_valid_unseen_36_reviewed_20261007.json"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/alf_online_fixed_first_sign_flip_reviewed_20261007.json"))
    parser.add_argument("--reference", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_reward_gate_valid_unseen_36_20261007.json"))
    parser.add_argument("--report", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_fixed_first_sign_flip_valid_unseen36_20261007.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_fixed_first_sign_flip_audited_20261007.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--source-index", type=int, default=1)
    parser.add_argument("--per-family", type=int, default=6)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
