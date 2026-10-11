"""Matched official unseen 134 evaluation for static rank-4 direct LoRA arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v39_134 import bound as unseen_bound, parser as old_parser
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay


def inputs(args):
    targets, first, _, _ = unseen_bound(args)
    audited = json.loads(args.old_audit.read_text())["games"]
    if (len(targets) != 134 or len(audited) != 134 or
            any(x["game"] != y["game"] or x["lora"]["reward"] != y["old_lora_reward"]
                for x, y in zip(first, audited, strict=True)) or
            sum(x["base_reward"] for x in audited) != 41 or
            sum(x["lora"]["reward"] for x in first) != 68):
        raise ValueError("Changed audited old official baselines")
    return targets, [{**x, "base_reward": y["base_reward"]}
                     for x, y in zip(first, audited, strict=True)]


def student(args):
    report = json.loads(args.training_report.read_text())
    model = torch.load(args.student, map_location="cpu", weights_only=False)
    if (report.get("model_sha256") != file_hash(args.student)
            or report.get("review_sha256") != file_hash(args.training_review)
            or model.get("review_sha256") != file_hash(args.training_review)
            or model.get("checkpoint_sha256") != file_hash(args.checkpoint)
            or model.get("rank") != 4 or model.get("layers") != 2
            or len(model.get("factors", [])) != 2
            or factors_hash(model["factors"]) != report.get("factor_sha256")):
        raise ValueError("Changed or incomplete static LoRA arm")
    return model["factors"]


def bound(args, targets, first):
    factors = student(args)
    fixed = {"method": "v41 direct static LoRA, same A/rank/layers as old hypernetwork",
        "arm": args.arm,
        "training_review_sha256": file_hash(args.training_review),
        "training_report_sha256": file_hash(args.training_report),
        "student_sha256": file_hash(args.student),
        "factor_sha256": factors_hash(factors),
        "target_review_sha256": file_hash(args.target_review),
        "old_audit_sha256": file_hash(args.old_audit),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "actor_sha256": file_hash(Path(__file__).with_name("alfworld_zero_shot_probe.py")),
        "runner_sha256": file_hash(Path(__file__)),
        "rank": 4, "layers": 2, "max_steps": 50,
        "max_new_tokens": 64, "actor_history_turns": 2,
        "loop_guard_max": 2, "constrain_actions": True}
    bindings = []
    for i, (target, old) in enumerate(zip(targets, first, strict=True)):
        content = {"index": i, "game": target["game"],
            "game_sha256": target["game_sha256"],
            "old_target_input_content_sha256": target["input_content_sha256"],
            "old_base_reward": old["base_reward"],
            "old_hyper_episode_sha256": digest(old["lora"]),
            "runner_sha256": fixed["runner_sha256"],
            "student_sha256": fixed["student_sha256"]}
        bindings.append({**content, "input_content_sha256": digest(content),
                         "reviewed_target": True})
    return fixed, bindings


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    targets, first = inputs(args)
    fixed, bindings = bound(args, targets, first)
    save(args.review, {"protocol": "Exposed official valid_unseen 134, exact old order and one-attempt actor budget; static direct B-LoRA rank4 last two down projections; no expert source at inference; descriptive comparison with audited base and hypernetwork",
        **fixed, "targets": bindings, "targets_sha256": digest(bindings)})
    print(json.dumps({"arm": args.arm, "reviewed_targets": len(bindings)}), flush=True)


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    targets, first = inputs(args)
    fixed, bindings = bound(args, targets, first)
    review = json.loads(args.review.read_text())
    if (review.get("targets") != bindings or review.get("targets_sha256") != digest(bindings)
            or any(review.get(k) != v for k, v in fixed.items())):
        raise ValueError("Changed reviewed static LoRA targets")
    factors = student(args)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if [list(x.shape) for x in factors] != [
            [1, layer.base.out_features, layer.rank] for layer in agent.adapters]:
        raise ValueError("Static factor geometry differs from hypernetwork")
    report = {"protocol": review["protocol"], "review_sha256": file_hash(args.review),
              **fixed, "rows": [], "failures": []}
    for i, (target, old, binding) in enumerate(zip(targets, first, bindings, strict=True)):
        ep = run_episode(agent, tokenizer, args.data_root / target["game"], {},
            adapter=False, fixed_adapter=factors, device=args.device,
            max_steps=50, max_new_tokens=64, constrain_actions=True,
            actor_history_turns=2, loop_guard_max=2)
        row = {"index": i, "game": target["game"],
            "input_content_sha256": binding["input_content_sha256"],
            "base_reward": old["base_reward"],
            "old_hyper_reward": old["lora"]["reward"],
            "episode": ep}
        if (ep["status"] != "complete" or
                ep["initial_observation"] != old["lora"]["initial_observation"]):
            report["failures"].append({"index": i, "status": ep["status"]})
        report["rows"].append(row)
        report["summary"] = {"n": len(report["rows"]),
            "base": sum(x["base_reward"] for x in report["rows"]),
            "old_hyper": sum(x["old_hyper_reward"] for x in report["rows"]),
            "static_lora": sum(x["episode"]["reward"] for x in report["rows"]),
            "failures": len(report["failures"])}
        save(args.output, report)
        print(json.dumps({"arm": args.arm, **report["summary"]}), flush=True)
        if report["failures"]:
            raise RuntimeError("Incomplete official static LoRA episode")


def audit(args):
    targets, first = inputs(args)
    fixed, bindings = bound(args, targets, first)
    review = json.loads(args.review.read_text())
    report = json.loads(args.output.read_text())
    if (review.get("targets") != bindings or review.get("targets_sha256") != digest(bindings)
            or any(review.get(k) != v or report.get(k) != v for k, v in fixed.items())
            or report.get("review_sha256") != file_hash(args.review)
            or len(report.get("rows", [])) != 134 or report.get("failures")):
        raise ValueError("Changed or incomplete static LoRA run")
    scores = {"base": 0., "old_hyper": 0., "static_lora": 0.}
    for i, (target, old, binding, row) in enumerate(zip(targets, first, bindings,
                                                          report["rows"], strict=True)):
        if (row["index"] != i or row["game"] != target["game"] or
                row["input_content_sha256"] != binding["input_content_sha256"] or
                row["base_reward"] != old["base_reward"] or
                row["old_hyper_reward"] != old["lora"]["reward"]):
            raise ValueError(f"Changed target at {i}")
        ep = row["episode"]
        if ep["status"] != "complete" or ep["steps"] > 50 or ep["initial_observation"] != old["lora"]["initial_observation"]:
            raise ValueError(f"Invalid static episode at {i}")
        replay(args.data_root / target["game"], ep)
        scores["base"] += row["base_reward"]
        scores["old_hyper"] += row["old_hyper_reward"]
        scores["static_lora"] += ep["reward"]
    if report["summary"] != {"n": 134, **scores, "failures": 0}:
        raise ValueError("Official summary changed")
    return {"audit": "passed", "arm": args.arm,
        "summary": report["summary"], "replayed_episodes": 134,
        "report_sha256": file_hash(args.output),
        "review_sha256": file_hash(args.review)}


def parser():
    p = old_parser()
    p.add_argument("--arm", required=True, choices=("sft", "grpo"))
    p.add_argument("--training-review", type=Path, required=True)
    p.add_argument("--training-report", type=Path, required=True)
    p.add_argument("--student", type=Path, required=True)
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--audit-output", type=Path)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.audit_only:
        if args.audit_output is None or args.audit_output.exists():
            raise ValueError("Fresh audit output required")
        result = audit(args)
        save(args.audit_output, result)
        print(json.dumps(result), flush=True)
    else:
        (prepare if args.command == "prepare" else evaluate)(args)
