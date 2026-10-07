"""Verify target, source, reward, and state lineage of online valid_unseen probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)
from ttcl.trajectory_hyperlora.alfworld_online_reward_gate_valid_unseen_v1 import (
    checked_targets,
)


def audit(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    targets = checked_targets(args)
    raw = json.loads(args.report.read_text())
    if (raw["review_sha256"] != file_hash(args.review) or
            raw["checkpoint_sha256"] != file_hash(args.checkpoint) or
            raw["per_family"] != args.per_family or
            raw["max_steps"] != 50 or raw["max_new_tokens"] != 64 or
            raw["actor_history_turns"] != 2 or raw["loop_guard_max"] != 2 or
            raw["context_tokens"] != 2048 or raw["failures"] or
            len(raw["games"]) != len(targets)):
        raise ValueError("Changed or incomplete official online result")
    prior = []
    state_hash = None
    updates = 0
    paired = []
    for target, row in zip(targets, raw["games"], strict=True):
        base, online = row["base"], row["online"]
        if (row["game"] != target["game"] or
                row["family"] != target["family"] or
                row["input_content_sha256"] != target["input_content_sha256"] or
                row["prior_episode_count"] != len(prior) or
                row["prior_episodes_sha256"] != digest(prior) or
                (state_hash is not None and
                 row["source_vector_sha256_before"] != state_hash) or
                base["status"] != "complete" or online["status"] != "complete" or
                base["initial_observation"] != online["initial_observation"] or
                any(ep["steps"] > 50 or ep["steps"] != len(ep["trajectory"]) or
                    ep["invalid_commands"] != 0 or
                    ep["reward"] != float(ep["termination"] == "success")
                    for ep in (base, online)) or
                row["memory_updated"] != bool(online["reward"]) or
                row["update_count_after"] != updates + int(bool(online["reward"])) or
                row.get("source_tokens_retained", 0) > 2048 or
                (not online["reward"] and
                 row["source_vector_sha256_before"] !=
                    row["source_vector_sha256_after"])):
            raise ValueError(f"Changed target, episode, or update: {row['game']}")
        records = records_from_episode(online)
        if row["new_records_sha256"] != digest(records):
            raise ValueError("Own trajectory content changed")
        prior.append({"game": row["game"],
                      "reward": online["reward"], "records": records})
        state_hash = row["source_vector_sha256_after"]
        updates = row["update_count_after"]
        paired.append({"game": row["game"], "family": row["family"],
                       "base": base["reward"], "online": online["reward"],
                       "memory_updated": row["memory_updated"]})
    families = sorted({row["family"] for row in paired})
    summary = {"n": len(paired), "base": sum(x["base"] for x in paired),
        "online": sum(x["online"] for x in paired), "updates": updates,
        "gains": sum(x["online"] > x["base"] for x in paired),
        "losses": sum(x["online"] < x["base"] for x in paired),
        "by_family": {family: {
            "n": sum(x["family"] == family for x in paired),
            "base": sum(x["base"] for x in paired if x["family"] == family),
            "online": sum(x["online"] for x in paired if x["family"] == family)}
            for family in families}}
    if raw["summary"] != {key: summary[key] for key in
            ("n", "base", "online", "updates")} | {"failures": 0}:
        raise ValueError("Aggregate summary changed")
    result = {"protocol": "Content-bound audit of official valid_unseen exploratory online reward-gated source-vector memory; frozen target order and per-episode state chain",
        "raw_report_sha256": file_hash(args.report),
        "review_sha256": file_hash(args.review),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "summary": summary, "games": paired}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--parent-review", type=Path, default=Path(
        "data/annotations/alf_memrl134_hyperlora_train_sources_reviewed_20261006.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/alf_online_reward_gate_valid_unseen_36_reviewed_20261007.json"))
    parser.add_argument("--report", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_reward_gate_valid_unseen_36_20261007.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_reward_gate_valid_unseen_36_audited_20261007.json"))
    parser.add_argument("--per-family", type=int, default=6)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
