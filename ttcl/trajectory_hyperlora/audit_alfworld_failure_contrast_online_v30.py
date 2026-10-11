"""Audit v30 online chronology and independently replay official actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)
from ttcl.trajectory_hyperlora.alfworld_failure_contrast_online_v30 import ARMS
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay


def audit(args):
    report = json.loads(args.report.read_text())
    review = json.loads(args.review.read_text())
    source = json.loads(args.source_review.read_text())
    if (report["review_sha256"] != file_hash(args.review) or
            review["targets_sha256"] != digest(review["targets"]) or
            tuple(report["arms"]) != ARMS or
            len(report["games"]) != 18 or
            len(review["targets"]) != 18 or
            len(source["targets"]) != 18 or report["failures"]):
        raise ValueError("Changed or incomplete failure-contrast run")
    paths = {"runner_sha256": args.runner,
             "actor_sha256": args.actor,
             "factor_helper_sha256": args.factor_helper,
             "update_sha256": args.update,
             "checkpoint_sha256": args.checkpoint,
             "source_review_sha256": args.source_review}
    for key, path in paths.items():
        if report[key] != file_hash(path) or review[key] != report[key]:
            raise ValueError(f"Changed {key}")
    if report["model_config_sha256"] != file_hash(args.model / "config.json"):
        raise ValueError("Changed actor model")
    if (report["max_steps"], report["max_new_tokens"],
            report["history_turns"], report["repeat_limit"],
            report["sequence"]) != (50, 64, 2, 2, "train_seq6_6"):
        raise ValueError("Changed protocol budget")
    history = {arm: [] for arm in ARMS}
    hashes = {arm: factors_hash(None) for arm in ARMS}
    successes = {arm: 0 for arm in ARMS}
    states = {arm: 0 for arm in ARMS}
    paired = {arm: 0 for arm in ARMS}
    contrasts = {arm: 0 for arm in ARMS}
    totals = {"base": 0., **{arm: 0. for arm in ARMS}}
    for i, (row, target, prior) in enumerate(zip(
            report["games"], review["targets"], source["targets"],
            strict=True)):
        content = {key: value for key, value in target.items()
                   if key not in ("input_content_sha256", "reviewed_target")}
        if (row["index"] != i or target["index"] != i or
                row["game"] != target["game"] != prior["game"] or
                target["game_sha256"] != file_hash(args.data_root /
                    target["game"]) or
                target["game_sha256"] != prior["game_sha256"] or
                not target["reviewed_target"] or
                target["input_content_sha256"] != digest(content) or
                row["input_content_sha256"] != target["input_content_sha256"] or
                tuple(row["arms"]) != ARMS):
            raise ValueError(f"Changed target/order at {i}")
        base = row["base"]
        replay(args.data_root / row["game"], base)
        totals["base"] += base["reward"]
        for arm in ARMS:
            entry = row["arms"][arm]
            episode = entry["episode"]
            replay(args.data_root / row["game"], episode)
            if (episode["initial_observation"] !=
                    base["initial_observation"] or
                    entry["prior_episode_count"] != len(history[arm]) or
                    entry["prior_episodes_sha256"] != digest(history[arm]) or
                    entry["factor_sha256_before"] != hashes[arm] or
                    entry["records_sha256"] != digest(
                        records_from_episode(episode)) or
                    entry.get("source_tokens_retained", 0) >
                    report["context_tokens"]):
                raise ValueError(f"Changed episode/history at {i}:{arm}")
            kind = entry["update_kind"]
            if bool(episode["reward"]):
                if kind != "own_success":
                    raise ValueError("Successful episode did not update")
                successes[arm] += 1
                states[arm] += 1
            elif bool(base["reward"]) and arm == "base_success":
                if kind != "paired_base_success":
                    raise ValueError("Missing positive-only comparator update")
                paired[arm] += 1
                states[arm] += 1
            elif bool(base["reward"]) and arm == "failure_contrast":
                if kind == "paired_failure_contrast":
                    contrasts[arm] += 1
                    states[arm] += 1
                    if entry["contrast"]["applied_norm"] <= 0:
                        raise ValueError("Empty recorded contrast")
                elif kind != "zero_contrast_abstain":
                    raise ValueError("Missing failure-contrast decision")
            elif kind != "none":
                raise ValueError("Unsupported failure update")
            if (entry["success_count_after"] != successes[arm] or
                    entry["state_count_after"] != states[arm] or
                    entry["base_positive_count_after"] != paired[arm] or
                    entry["contrast_count_after"] != contrasts[arm] or
                    (kind in ("none", "zero_contrast_abstain") and
                     entry["factor_sha256_after"] != hashes[arm])):
                raise ValueError(f"Changed state chronology at {i}:{arm}")
            hashes[arm] = entry["factor_sha256_after"]
            history[arm].append({"game": row["game"],
                "reward": episode["reward"],
                "records": records_from_episode(episode)})
            totals[arm] += episode["reward"]
    summary = {"n": 18, **totals,
        "own_success_updates": successes,
        "state_updates": states,
        "paired_base_success_updates": paired,
        "paired_failure_contrast_updates": contrasts,
        "failures": 0}
    if report["summary"] != summary:
        raise ValueError("Changed result summary")
    if args.old_report:
        old = json.loads(args.old_report.read_text())
        same_base = sum(row["base"] == old_row["base"] for row, old_row in
                        zip(report["games"], old["games"], strict=True))
        same_success_only = sum(row["arms"]["success_only"]["episode"] ==
            old_row["online"] for row, old_row in
            zip(report["games"], old["games"], strict=True))
    else:
        same_base = same_success_only = None
    return {"audit": "passed", "summary": summary,
        "same_base_as_v27": same_base,
        "same_success_only_as_v27": same_success_only,
        "report_sha256": file_hash(args.report),
        "review_sha256": file_hash(args.review)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_failure_contrast_online_v30_train_seq6_6_20261009.json"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/alfworld_failure_contrast_online_v30_train_seq6_6_20261009.json"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alfworld_online_context_persistent_train_seq6_6_reviewed_20261007.json"))
    parser.add_argument("--old-report", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_success_gated_factors_v27_train_seq6_6_20261009.json"))
    parser.add_argument("--runner", type=Path, default=Path(
        "ttcl/trajectory_hyperlora/alfworld_failure_contrast_online_v30.py"))
    parser.add_argument("--actor", type=Path, default=Path(
        "ttcl/trajectory_hyperlora/alfworld_zero_shot_probe.py"))
    parser.add_argument("--factor-helper", type=Path, default=Path(
        "ttcl/trajectory_hyperlora/alfworld_success_gated_factors_v27.py"))
    parser.add_argument("--update", type=Path, default=Path(
        "ttcl/trajectory_hyperlora/contrastive_failure_factors_v30.py"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result), flush=True)
