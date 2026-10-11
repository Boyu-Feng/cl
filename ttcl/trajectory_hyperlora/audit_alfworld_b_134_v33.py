"""Audit bound online B histories and replay every official ALFWorld episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash, records_from_episode
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay


def audit(args):
    report = json.loads(args.report.read_text())
    review = json.loads(args.review.read_text())
    source = json.loads(args.target_review.read_text())
    if (len(review["targets"]) != 134 or len(source["targets"]) != 134 or
            len(report["games"]) != 134 or report["failures"] or
            review["targets_sha256"] != digest(review["targets"]) or
            report["review_sha256"] != file_hash(args.review) or
            report["target_review_sha256"] != file_hash(args.target_review) or
            report["checkpoint_sha256"] != file_hash(args.checkpoint) or
            report["runner_sha256"] != file_hash(args.runner)):
        raise ValueError("Incomplete or changed 134-game report")
    if args.mode == "joint":
        if (report["update_model_sha256"] != file_hash(args.update_model) or
                report["training_report_sha256"] != file_hash(args.training_report)):
            raise ValueError("Changed trained updater")
    else:
        if report["mode"] != args.mode or review["mode"] != args.mode:
            raise ValueError("Changed baseline update mode")
    history = []
    state = factors_hash(None)
    wins = 0.
    updates = 0
    for index, (row, target, original) in enumerate(zip(
            report["games"], review["targets"], source["targets"], strict=True)):
        episode = row["episode"]
        if (row["index"] != index or target["index"] != index or
                row["game"] != target["game"] or
                row["game"] != original["game"] or
                target["game_sha256"] != original["game_sha256"] or
                target["game_sha256"] != file_hash(args.data_root / row["game"]) or
                row["input_content_sha256"] != target["input_content_sha256"] or
                row["prior_episodes_sha256"] != digest(history) or
                row["factor_sha256_before"] != state or
                episode["status"] != "complete" or episode["steps"] > 50 or
                episode["invalid_commands"] != 0 or
                episode["reward"] != float(episode["termination"] == "success")):
            raise ValueError(f"Changed task, chronology, or score at {index}")
        replay(args.data_root / row["game"], episode)
        records = records_from_episode(episode)
        if (row["new_records_sha256"] != digest(records) or
                row.get("source_tokens_retained", 0) > 2048):
            raise ValueError(f"Changed bound trajectory at {index}")
        should_update = (args.mode in ("joint", "all_mean") or
                         bool(episode["reward"]))
        if row["memory_updated"] != should_update:
            raise ValueError(f"Changed write rule at {index}")
        if should_update:
            updates += 1
            if (row["factor_sha256_after"] == state or
                    row["update_count_after"] != updates):
                raise ValueError(f"Missing parameter update at {index}")
            if args.mode == "joint" and (len(row["gate"]) != 2 or
                    any(not 0 <= value <= 1 for value in row["gate"])):
                raise ValueError(f"Invalid learned gate at {index}")
        elif (row["factor_sha256_after"] != state or
              row["update_count_after"] != updates):
            raise ValueError(f"Rejected episode changed B at {index}")
        state = row["factor_sha256_after"]
        history.append({"game": row["game"], "reward": episode["reward"],
                        "records_sha256": digest(records)})
        wins += episode["reward"]
    if report["summary"] != {"n": 134, "wins": wins,
                             "updates": updates, "failures": 0}:
        raise ValueError("Changed official summary")
    return {"audit": "passed", "mode": args.mode,
        "n": 134, "wins": wins, "updates": updates,
        "replayed": 134, "report_sha256": file_hash(args.report),
        "review_sha256": file_hash(args.review)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("all_mean", "success_mean", "joint"),
                        required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-review", type=Path, default=Path(
        "data/annotations/alf_memrl134_hyperlora_train_sources_reviewed_20261006.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--update-model", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_joint_b_update_v33_train_20261009.pt"))
    parser.add_argument("--training-report", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_joint_b_update_v33_train_20261009.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result))
