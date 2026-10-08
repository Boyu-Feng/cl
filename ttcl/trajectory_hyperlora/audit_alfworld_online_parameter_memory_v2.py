"""Read-only content and official-environment audit of online parameter v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)
from ttcl.trajectory_hyperlora.alfworld_online_parameter_memory_v2 import checked_targets
from ttcl.trajectory_hyperlora.audit_alfworld_source_utility_dataset_v3 import replay
from ttcl.trajectory_hyperlora.online_parameter_memory_v2 import ParameterMemory


def audit(args):
    if args.audit_output.exists():
        raise FileExistsError(args.audit_output)
    targets = checked_targets(args)
    report = json.loads(args.output.read_text())
    if (report["review_sha256"] != file_hash(args.review) or
            report["checkpoint_sha256"] != file_hash(args.checkpoint) or
            report["model_config_sha256"] != file_hash(args.model / "config.json") or
            report["capacity"] != args.capacity or
            report["temperature"] != args.temperature or
            report["attention_mix"] != args.attention_mix or
            report["context_tokens"] != args.context_tokens or
            report["max_steps"] != 50 or report["max_new_tokens"] != 64 or
            report["actor_history_turns"] != 2 or
            report["loop_guard_max"] != 2 or
            report["failures"] or len(report["games"]) != len(targets)):
        raise ValueError("Changed or incomplete parameter-memory run")
    prior = []
    previous_hash = ParameterMemory().digest()
    entries = 0
    base_rewards = []
    online_rewards = []
    for target, row in zip(targets, report["games"], strict=True):
        if (row["game"] != target["game"] or
                row["input_content_sha256"] != target["input_content_sha256"] or
                row["prior_episode_count"] != len(prior) or
                row["prior_episodes_sha256"] != digest(prior) or
                row["memory_entries_before"] != entries or
                row["retrieval"]["entries"] != entries or
                row["memory_sha256_before"] != previous_hash):
            raise ValueError("Broken target or online memory lineage")
        weights = row["retrieval"]["weights"]
        if (len(weights) != entries or
                len(row["retrieval"]["cosine"]) != entries or
                (entries and (abs(sum(weights) - 1) > 1e-5 or
                    not 0 <= row["retrieval"]["selected"] < entries))):
            raise ValueError("Changed readout weights")
        game = args.data_root / row["game"]
        base_rewards.append(replay(game, row["base"]))
        online_rewards.append(replay(game, row["online"]))
        if row["base"]["initial_observation"] != row["online"]["initial_observation"]:
            raise ValueError("Paired resets changed")
        records = records_from_episode(row["online"])
        if row["new_records_sha256"] != digest(records):
            raise ValueError("Trajectory content changed")
        updated = bool(row["online"]["reward"])
        if (row["memory_updated"] is not updated or
                row["memory_entries_after"] !=
                    min(args.capacity, entries + int(updated)) or
                (not updated and row["memory_sha256_after"] !=
                    row["memory_sha256_before"]) or
                (updated and (row["memory_sha256_after"] ==
                    row["memory_sha256_before"] or
                    row["write_slot"] >= args.capacity)) or
                row.get("source_tokens_retained", 0) > args.context_tokens):
            raise ValueError("Invalid reward-controlled write")
        prior.append({"game": row["game"], "reward": row["online"]["reward"],
                      "records": records})
        previous_hash = row["memory_sha256_after"]
        entries = row["memory_entries_after"]
    summary = {"n": len(targets), "base": sum(base_rewards),
               "online": sum(online_rewards),
               "writes": sum(row["memory_updated"] for row in report["games"]),
               "online_only": sum(b < o for b, o in zip(base_rewards, online_rewards)),
               "base_only": sum(b > o for b, o in zip(base_rewards, online_rewards)),
               "replay_failures": 0}
    if (any(summary[key] != report["summary"][key] for key in
            ("n", "base", "online", "writes")) or
            report["summary"]["failures"] != 0):
        raise ValueError("Reported summary changed")
    value = {"protocol": "Content-bound online state chain and original-environment replay of all paired official train episodes",
             "raw_report_sha256": file_hash(args.output),
             "review_sha256": file_hash(args.review),
             "checkpoint_sha256": file_hash(args.checkpoint),
             "summary": summary}
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--plan", type=Path, default=Path(
        "ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json"))
    parser.add_argument("--data-root", type=Path, default=Path("ttcl/data/alfworld_delta"))
    parser.add_argument("--parent-review", type=Path, default=Path(
        "data/annotations/alfworld_online_from_empty_train_seq6_6_reviewed_20261007.json"))
    parser.add_argument("--parent-persistent-review", type=Path, default=Path(
        "data/annotations/alfworld_online_context_persistent_train_seq6_6_reviewed_20261007.json"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/alfworld_online_parameter_memory_v2_train_seq6_6_reviewed_20261008.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_parameter_memory_v2_train_seq6_6_20261008.json"))
    parser.add_argument("--audit-output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_parameter_memory_v2_train_seq6_6_audited_20261008.json"))
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=.05)
    parser.add_argument("--attention-mix", type=float, default=.75)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
