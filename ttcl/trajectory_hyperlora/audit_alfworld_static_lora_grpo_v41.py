"""Audit GRPO target bindings, sampled policy records and official rewards."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.train_alfworld_static_lora_grpo_v41 import bindings, groups, parser
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay


def audit(args):
    tasks = groups(args)
    fixed, targets = bindings(args, tasks)
    review = json.loads(args.review.read_text())
    report = json.loads(args.output.read_text())
    if (review.get("targets") != targets or review.get("targets_sha256") != digest(targets)
            or any(review.get(k) != v or report.get(k) != v for k, v in fixed.items())
            or report.get("review_sha256") != file_hash(args.review)
            or len(report.get("groups", [])) != 33 or report.get("failures")
            or report.get("model_sha256") != file_hash(args.model_output)):
        raise ValueError("Changed or incomplete GRPO run")
    model = torch.load(args.model_output, map_location="cpu", weights_only=False)
    if (model["review_sha256"] != file_hash(args.review)
            or model["checkpoint_sha256"] != fixed["checkpoint_sha256"]
            or model["trainer_sha256"] != fixed["trainer_sha256"]
            or model["rank"] != 4 or model["layers"] != 2
            or factors_hash(model["factors"]) != report["factor_sha256"]):
        raise ValueError("Changed final GRPO factor checkpoint")
    rewards = 0.
    updated = 0
    for i, (task, binding, row) in enumerate(zip(tasks, targets,
                                                   report["groups"], strict=True)):
        if (row["group_index"] != i or row["game"] != binding["game"] or
                row["input_content_sha256"] != binding["input_content_sha256"] or
                row["sampling_seeds"] != binding["sampling_seeds"] or
                len(row["episodes"]) != args.group_size or
                len(row["events"]) != args.group_size or
                (i and row["factor_sha256_before"] !=
                 report["groups"][i-1]["factor_sha256_after"])):
            raise ValueError(f"Changed GRPO group at {i}")
        group_rewards = []
        for episode, events in zip(row["episodes"], row["events"], strict=True):
            if (episode["status"] != "complete" or episode["steps"] > 50 or
                    len(events) != episode["steps"]):
                raise ValueError(f"Invalid sampled episode at {i}")
            replay(args.data_root / task["target_game"], episode)
            for step, event in zip(episode["trajectory"], events, strict=True):
                prompt = event["prompt_ids"]
                response = event["response_ids"]
                allowed = event["allowed_token_ids"]
                logps = event["old_log_probs"]
                if (event["raw_response"] != step["response"] or
                        event["token_binding"] != digest([prompt, response, allowed])
                        or not prompt or not response or
                        len(response) != len(allowed) or len(response) != len(logps)
                        or any(token not in options or not math.isfinite(logp)
                               for token, options, logp in zip(
                                   response, allowed, logps, strict=True))):
                    raise ValueError(f"Changed constrained policy event at {i}")
            group_rewards.append(episode["reward"])
            rewards += episode["reward"]
        variance = max(group_rewards) != min(group_rewards)
        if row["update"]["updated"] != variance:
            raise ValueError(f"Group advantage update changed at {i}")
        updated += variance
    summary = {"groups": 33, "rollouts": 33 * args.group_size,
        "wins": rewards, "updated_groups": updated, "failures": 0}
    if report["summary"] != summary or report["groups"][-1]["factor_sha256_after"] != report["factor_sha256"]:
        raise ValueError("Changed GRPO summary or final state")
    return {"audit": "passed", "summary": summary,
        "replayed_episodes": 33 * args.group_size,
        "report_sha256": file_hash(args.output),
        "review_sha256": file_hash(args.review),
        "model_sha256": file_hash(args.model_output)}


if __name__ == "__main__":
    p = parser()
    p.add_argument("--audit-output", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_static_lora_grpo_v41_train33_audit_20261010.json"))
    args = p.parse_args()
    if args.command != "train" or args.audit_output.exists():
        raise ValueError("Audit requires completed train and fresh output")
    result = audit(args)
    save(args.audit_output, result)
    print(json.dumps(result), flush=True)
