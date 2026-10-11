"""Frozen hypernetwork counterfactual: rotate reviewed train sources within family.

This isolates the contribution of the particular successful source trajectory
from the frozen actor, hypernetwork weights, target games, and action budget.
The already exposed valid_unseen set makes the result descriptive.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v39_134 import (
    bound as old_bound, parser,
)
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_source_fields
from ttcl.trajectory_hyperlora.evaluate_alfworld_failure_retry_v31b import retry
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay


def rotated_sources(targets):
    by_family = defaultdict(list)
    for i, row in enumerate(targets):
        by_family[row["family"]].append(i)
    assignment = {}
    for indices in by_family.values():
        if len(indices) < 2:
            raise ValueError("Cannot rotate a singleton task family")
        for position, index in enumerate(indices):
            assignment[index] = indices[(position + 1) % len(indices)]
    if sorted(assignment) != list(range(len(targets))) or any(
            targets[i]["family"] != targets[j]["family"] or i == j
            for i, j in assignment.items()):
        raise ValueError("Invalid within-family source permutation")
    return assignment


def bound(args):
    targets, first, _, _ = old_bound(args)
    assignment = rotated_sources(targets)
    fixed = {
        "method": "v42 same-family source-trajectory rotation, frozen hypernetwork",
        "target_review_sha256": file_hash(args.target_review),
        "old_audit_sha256": file_hash(args.old_audit),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "runner_sha256": file_hash(Path(__file__)),
        "actor_sha256": file_hash(Path(__file__).with_name("alfworld_zero_shot_probe.py")),
        "assignment": "cyclic-next-target-within-family-in-original-order",
        "max_steps": 50, "max_new_tokens": 64,
        "actor_history_turns": 2, "loop_guard_max": 2,
    }
    bindings = []
    for i, (target, old) in enumerate(zip(targets, first, strict=True)):
        j = assignment[i]
        donor = targets[j]
        if "/train/" not in "/" + donor["source_game"]:
            raise ValueError("Source left official train split")
        content = {
            "index": i, "game": target["game"],
            "game_sha256": target["game_sha256"],
            "family": target["family"],
            "original_source_records_sha256": target["source_records_sha256"],
            "donor_index": j, "donor_source_game": donor["source_game"],
            "donor_source_records_sha256": donor["source_records_sha256"],
            "old_target_input_content_sha256": target["input_content_sha256"],
            "old_episode_sha256": digest(old["lora"]),
            "runner_sha256": fixed["runner_sha256"],
            "checkpoint_sha256": fixed["checkpoint_sha256"],
        }
        bindings.append({**content, "input_content_sha256": digest(content),
                         "reviewed_target": True})
    return targets, first, assignment, fixed, bindings


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    _, _, _, fixed, bindings = bound(args)
    save(args.review, {
        "protocol": "Exposed official valid_unseen 134, one attempt per target; same frozen actor and hypernetwork, same-family cyclically rotated reviewed train success trajectory; fixed mapping independent of target rewards; descriptive source-content ablation",
        **fixed, "targets": bindings, "targets_sha256": digest(bindings),
    })
    print(json.dumps({"reviewed_targets": len(bindings)}), flush=True)


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    targets, first, assignment, fixed, bindings = bound(args)
    review = json.loads(args.review.read_text())
    if (review.get("targets") != bindings or
            review.get("targets_sha256") != digest(bindings) or
            any(review.get(k) != v for k, v in fixed.items())):
        raise ValueError("Changed reviewed permutation")
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    report = {"protocol": review["protocol"], "review_sha256": file_hash(args.review),
              **fixed, "rows": [], "failures": []}
    for i, (target, old, binding) in enumerate(zip(targets, first, bindings,
                                                     strict=True)):
        donor = targets[assignment[i]]
        fields = contextual_source_fields(agent, tokenizer, donor["source_records"],
                                           args.device, 2048, pooling="both")
        ep = retry(agent, tokenizer, args.data_root / target["game"],
                   fields, args.device)
        row = {"index": i, "game": target["game"],
               "input_content_sha256": binding["input_content_sha256"],
               "donor_index": assignment[i],
               "old_hyper_reward": old["lora"]["reward"],
               "episode": ep}
        if (ep["status"] != "complete" or
                ep["initial_observation"] != old["lora"]["initial_observation"]):
            report["failures"].append({"index": i, "status": ep["status"]})
        report["rows"].append(row)
        report["summary"] = {
            "n": len(report["rows"]),
            "old_hyper": sum(x["old_hyper_reward"] for x in report["rows"]),
            "rotated_source": sum(x["episode"]["reward"] for x in report["rows"]),
            "failures": len(report["failures"]),
        }
        save(args.output, report)
        print(json.dumps(report["summary"]), flush=True)
        if report["failures"]:
            raise RuntimeError("Incomplete rotated-source episode")


def audit(args):
    targets, first, assignment, fixed, bindings = bound(args)
    review = json.loads(args.review.read_text())
    report = json.loads(args.output.read_text())
    if (review.get("targets") != bindings or
            review.get("targets_sha256") != digest(bindings) or
            any(review.get(k) != v or report.get(k) != v for k, v in fixed.items()) or
            report.get("review_sha256") != file_hash(args.review) or
            len(report.get("rows", [])) != 134 or report.get("failures")):
        raise ValueError("Changed or incomplete source permutation")
    old = new = 0.
    for i, (target, prior, binding, row) in enumerate(zip(
            targets, first, bindings, report["rows"], strict=True)):
        ep = row["episode"]
        if (row["index"] != i or row["game"] != target["game"] or
                row["input_content_sha256"] != binding["input_content_sha256"] or
                row["donor_index"] != assignment[i] or
                row["old_hyper_reward"] != prior["lora"]["reward"] or
                ep["status"] != "complete" or ep["steps"] > 50 or
                ep["initial_observation"] != prior["lora"]["initial_observation"]):
            raise ValueError(f"Changed episode {i}")
        replay(args.data_root / target["game"], ep)
        old += row["old_hyper_reward"]
        new += ep["reward"]
    summary = {"n": 134, "old_hyper": old,
               "rotated_source": new, "failures": 0}
    if report["summary"] != summary:
        raise ValueError("Changed score summary")
    return {"audit": "passed", "summary": summary,
            "replayed_episodes": 134,
            "report_sha256": file_hash(args.output),
            "review_sha256": file_hash(args.review)}


if __name__ == "__main__":
    p = parser()
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--audit-output", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_source_permutation_v42_134_audit_20261010.json"))
    args = p.parse_args()
    if args.audit_only:
        if args.audit_output.exists():
            raise FileExistsError(args.audit_output)
        result = audit(args)
        save(args.audit_output, result)
        print(json.dumps(result), flush=True)
    else:
        (prepare if args.command == "prepare" else evaluate)(args)
