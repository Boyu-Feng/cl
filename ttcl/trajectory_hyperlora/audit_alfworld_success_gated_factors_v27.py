"""Audit official ALFWorld wins, target bindings, and persistent-B chronology."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)


def audit(args):
    report = json.loads(args.report.read_text())
    review = json.loads(args.review.read_text())
    source = json.loads(args.source_review.read_text())
    if (report["review_sha256"] != file_hash(args.review) or
            report["source_review_sha256"] != file_hash(args.source_review) or
            report["runner_sha256"] != file_hash(args.runner) or
            review["runner_sha256"] != file_hash(args.runner) or
            report["actor_sha256"] != file_hash(args.runner.with_name(
                "alfworld_zero_shot_probe.py")) or
            review["actor_sha256"] != report["actor_sha256"] or
            report["factor_update_sha256"] != file_hash(args.runner.with_name(
                "alfworld_online_accumulate_lora_v1.py")) or
            review["factor_update_sha256"] != report["factor_update_sha256"] or
            report["checkpoint_sha256"] != review["checkpoint_sha256"] or
            report["model_config_sha256"] != file_hash(args.model / "config.json") or
            report["sequence"] != review["sequence"] or
            review["input_content_sha256"] != digest(review["targets"]) or
            len(review["targets"]) != len(source["targets"])):
        raise ValueError("Changed frozen model, review, source, or runner")
    if (report["max_steps"], report["max_new_tokens"],
            report["actor_history_turns"], report["loop_guard_max"]) != \
            (50, 64, 2, 2):
        raise ValueError("Changed action budget")
    if len(report["games"]) != len(review["targets"]) and not args.partial:
        raise ValueError("Incomplete full result")
    if len(report["games"]) > len(review["targets"]):
        raise ValueError("Extra result")
    prior = []
    state_hash = factors_hash(None)
    updates = 0
    wins = 0
    base_wins = 0
    for i, (row, target, parent) in enumerate(zip(
            report["games"], review["targets"], source["targets"])):
        if (target["index"] != i or target["game"] != parent["game"] or
                target["game_sha256"] != parent["game_sha256"] or
                target["game_sha256"] != file_hash(args.data_root / target["game"]) or
                row["game"] != target["game"] or
                row["input_content_sha256"] != target["input_content_sha256"] or
                row["prior_episode_count"] != len(prior) or
                row["prior_episodes_sha256"] != digest(prior) or
                row["factor_sha256_before"] != state_hash):
            raise ValueError(f"Changed target or parameter chain at index {i}")
        base, online = row["base"], row["online"]
        if (base["status"] != "complete" or
                online["status"] != "complete" or
                base["initial_observation"] != online["initial_observation"]):
            raise ValueError(f"Unpaired official episode at index {i}")
        for episode in (base, online):
            if (episode["steps"] > 50 or
                    episode["steps"] != len(episode["trajectory"]) or
                    episode["invalid_commands"] != 0 or
                    episode["reward"] != float(episode["termination"] == "success")):
                raise ValueError(f"Changed official episode at index {i}")
        records = records_from_episode(online)
        if (row["new_records_sha256"] != digest(records) or
                row["memory_updated"] != bool(online["reward"]) or
                row["update_count_after"] != updates + int(bool(online["reward"])) or
                row.get("source_tokens_retained", 0) > report["context_tokens"]):
            raise ValueError(f"Changed update source or gate at index {i}")
        if row["memory_updated"]:
            if row["factor_sha256_after"] == state_hash:
                raise ValueError(f"Won trajectory did not update B at index {i}")
            updates += 1
        elif row["factor_sha256_after"] != state_hash:
            raise ValueError(f"Lost trajectory updated B at index {i}")
        state_hash = row["factor_sha256_after"]
        prior.append({"game": row["game"], "reward": online["reward"],
                      "records": records})
        wins += int(online["reward"])
        base_wins += int(base["reward"])
    if (report["failures"] or report["summary"] != {
            "n": len(report["games"]), "base": float(base_wins),
            "online": float(wins), "updates": updates, "failures": 0}):
        raise ValueError("Changed summary or failure record")
    result = {"audit": "passed", "n": len(report["games"]),
        "base_wins": base_wins, "factor_wins": wins, "updates": updates,
        "report_sha256": file_hash(args.report),
        "review_sha256": file_hash(args.review)}
    if args.vector_report:
        vector = json.loads(args.vector_report.read_text())
        if len(vector["games"]) != len(report["games"]):
            raise ValueError("Vector comparator target count changed")
        vector_rewards = []
        for i, (factor_row, vector_row) in enumerate(zip(
                report["games"], vector["games"])):
            if factor_row["game"] != vector_row["game"]:
                raise ValueError(f"Vector comparator order changed at {i}")
            episode = vector_row.get("episode", vector_row.get("online"))
            if episode["status"] != "complete":
                raise ValueError(f"Incomplete vector comparator at {i}")
            vector_rewards.append(int(episode["reward"]))
        result.update(vector_wins=sum(vector_rewards),
            factor_only=sum(bool(r["online"]["reward"]) and not v for r,v in zip(
                report["games"], vector_rewards)),
            vector_only=sum(not bool(r["online"]["reward"]) and v for r,v in zip(
                report["games"], vector_rewards)),
            vector_report_sha256=file_hash(args.vector_report))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--source-review", type=Path, required=True)
    parser.add_argument("--vector-report", type=Path)
    parser.add_argument("--runner", type=Path, default=Path(
        "ttcl/trajectory_hyperlora/alfworld_success_gated_factors_v27.py"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
