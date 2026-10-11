"""Replay all six development arms and verify their frozen target bindings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay


ARMS = ("no_lora", "fixed_first", "random_same_reward",
        "source_permuted", "selected_text", "joint_lora")


def audit(args):
    review = json.loads(args.review.read_text())
    report = json.loads(args.report.read_text())
    if (report["review_sha256"] != file_hash(args.review) or
            review["targets_sha256"] != digest(review["targets"]) or
            review["runner_sha256"] != file_hash(args.runner) or
            review["actor_sha256"] != file_hash(args.actor) or
            review["action_guard_sha256"] != file_hash(args.guard) or
            review["trained_model_sha256"] != file_hash(args.trained_model) or
            review["checkpoint_sha256"] != file_hash(args.checkpoint) or
            report["trained_model_sha256"] != review["trained_model_sha256"] or
            report["checkpoint_sha256"] != review["checkpoint_sha256"] or
            report["actor_sha256"] != review["actor_sha256"] or
            report["action_guard_sha256"] != review["action_guard_sha256"] or
            len(report["games"]) > len(review["targets"]) or
            report["failures"]):
        raise ValueError("Changed development lineage or failure record")
    if len(report["games"]) != len(review["targets"]) and not args.partial:
        raise ValueError("Incomplete development result")
    totals = {arm: 0 for arm in ARMS}
    for row, target in zip(report["games"], review["targets"]):
        content = {key: val for key, val in target.items()
                   if key not in ("input_content_sha256", "reviewed_target")}
        if (not target["reviewed_target"] or
                target["input_content_sha256"] != digest(content) or
                row["index"] != target["index"] or
                row["game"] != target["game"] or
                row["input_content_sha256"] != target["input_content_sha256"] or
                target["game_sha256"] != file_hash(args.data_root /
                    target["game"]) or
                tuple(row["arms"]) != ARMS):
            raise ValueError("Changed target or arm order")
        for arm in ARMS:
            episode = row["arms"][arm]
            if episode["status"] != "complete":
                raise ValueError("Incomplete arm")
            replay(args.data_root / row["game"], episode)
            totals[arm] += episode["reward"]
    if not args.partial and totals != report["summary"]:
        raise ValueError("Changed development summary")
    return {"audit": "passed", "games": len(report["games"]),
        "totals": totals, "report_sha256": file_hash(args.report),
        "review_sha256": file_hash(args.review)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/my_online_update_v28_dev12_20261009.json"))
    parser.add_argument("--report", type=Path, default=Path(
        "results/trajectory_hyperlora/my_online_update_v28_dev12_20261009.json"))
    parser.add_argument("--runner", type=Path, default=Path(
        "ttcl/trajectory_hyperlora/evaluate_my_online_update_v28_dev.py"))
    parser.add_argument("--actor", type=Path, default=Path(
        "ttcl/trajectory_hyperlora/alfworld_actor_generic_v28.py"))
    parser.add_argument("--guard", type=Path, default=Path(
        "ttcl/trajectory_hyperlora/action_feedback_guard_v28.py"))
    parser.add_argument("--trained-model", type=Path, default=Path(
        "results/trajectory_hyperlora/my_online_update_v28_reward_train_20261009.pt"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result), flush=True)
