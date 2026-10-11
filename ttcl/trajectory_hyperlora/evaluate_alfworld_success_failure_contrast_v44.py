"""Future-task test of success-failure contrast and failure-source ablation."""

from __future__ import annotations

import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import (
    contextual_source_fields, contextual_text_fields,
)
from ttcl.trajectory_hyperlora.alfworld_success_gated_factors_v27 import (
    bounded_source_text,
)
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay
from ttcl.trajectory_hyperlora.train_alfworld_success_failure_contrast_v44 import (
    bad_source_for, checked, parser,
)


def bound(args):
    successes, failures, suffix, train_fixed, _ = checked(args)
    train = json.loads(args.contrast_report.read_text())
    eval_review = json.loads(args.parent_eval_review.read_text())
    if (train.get("review_sha256") != file_hash(args.contrast_review) or
            train.get("model_sha256") != file_hash(args.contrast_model) or
            any(train.get(k) != v for k, v in train_fixed.items()) or
            len(eval_review.get("targets", [])) != 120):
        raise ValueError("Changed contrast training or parent suffix binding")
    fixed = {"method": "v44 success-failure contrast on own-prefix trajectories",
        "contrast_review_sha256": file_hash(args.contrast_review),
        "contrast_report_sha256": file_hash(args.contrast_report),
        "contrast_model_sha256": file_hash(args.contrast_model),
        "parent_eval_review_sha256": file_hash(args.parent_eval_review),
        "runner_sha256": file_hash(Path(__file__)),
        "actor_sha256": file_hash(Path(__file__).with_name(
            "alfworld_zero_shot_probe.py")),
        "max_steps": 30, "max_new_tokens": 64,
        "constrain_actions": True, "actor_history_turns": 2,
        "loop_guard_max": 2}
    bindings = []
    for i, (row, parent) in enumerate(zip(suffix, eval_review["targets"],
                                           strict=True)):
        good = parent["own_success_source_index"]
        bad = bad_source_for(row, failures)
        if (parent["game"] != row["game"] or
                parent["own_success_source_records_sha256"] !=
                    digest(successes[good]["source_records"]) or
                file_hash(args.data_root / row["game"]) != parent["game_sha256"]):
            raise ValueError("Changed future task or good source")
        content = {"index": i, "game": row["game"],
            "game_sha256": parent["game_sha256"],
            "parent_target_input_content_sha256": parent["input_content_sha256"],
            "good_source_index": good,
            "good_source_records_sha256": digest(successes[good]["source_records"]),
            "bad_source_index": bad,
            "bad_source_records_sha256": digest(failures[bad]["source_records"]),
            "runner_sha256": fixed["runner_sha256"],
            "contrast_model_sha256": fixed["contrast_model_sha256"]}
        bindings.append({**content, "input_content_sha256": digest(content),
                         "reviewed_target": True})
    return successes, failures, suffix, fixed, bindings


def prepare(args):
    if args.eval_review.exists():
        raise FileExistsError(args.eval_review)
    _, _, _, fixed, bindings = bound(args)
    save(args.eval_review, {"protocol": "Official train suffix120 after own-success/failure prefix training; one same-budget rollout with own-success source and one with own-failure source; no suffix outcome used in model training or source selection",
        **fixed, "targets": bindings, "targets_sha256": digest(bindings)})
    print(json.dumps({"reviewed_suffix_targets": len(bindings)}), flush=True)


def evaluate(args):
    if args.eval_output.exists():
        raise FileExistsError(args.eval_output)
    successes, failures, suffix, fixed, bindings = bound(args)
    review = json.loads(args.eval_review.read_text())
    if (review.get("targets") != bindings or
            review.get("targets_sha256") != digest(bindings) or
            any(review.get(k) != v for k, v in fixed.items())):
        raise ValueError("Changed reviewed v44 suffix targets")
    agent, tokenizer = load_agent(args.model, args.contrast_model,
                                  args.device, args.gpu_fraction)
    report = {"protocol": review["protocol"],
        "review_sha256": file_hash(args.eval_review),
        **fixed, "rows": [], "failures": []}
    for i, (row, binding) in enumerate(zip(suffix, bindings, strict=True)):
        game = args.data_root / row["game"]
        good = successes[binding["good_source_index"]]
        bad = failures[binding["bad_source_index"]]
        good_fields = contextual_source_fields(agent, tokenizer,
            good["source_records"], args.device, 2048, pooling="both")
        good_ep = run_episode(agent, tokenizer, game, good_fields,
            adapter=True, device=args.device, max_steps=30,
            max_new_tokens=64, constrain_actions=True,
            actor_history_turns=2, loop_guard_max=2)
        bad_text, _, _ = bounded_source_text(tokenizer,
                                              bad["source_records"], 2048)
        bad_fields = contextual_text_fields(agent, tokenizer, bad_text,
            args.device, 2048, pooling="both")
        bad_ep = run_episode(agent, tokenizer, game, bad_fields,
            adapter=True, device=args.device, max_steps=30,
            max_new_tokens=64, constrain_actions=True,
            actor_history_turns=2, loop_guard_max=2)
        entry = {"index": i, "game": row["game"],
            "input_content_sha256": binding["input_content_sha256"],
            "good_source_index": binding["good_source_index"],
            "bad_source_index": binding["bad_source_index"],
            "good": good_ep, "bad": bad_ep}
        initial = row["arms"]["base"]["initial_observation"]
        if (good_ep["status"] != "complete" or bad_ep["status"] != "complete" or
                good_ep["initial_observation"] != initial or
                bad_ep["initial_observation"] != initial):
            report["failures"].append({"index": i})
        report["rows"].append(entry)
        report["summary"] = {"n": len(report["rows"]),
            "success_source": sum(x["good"]["reward"] for x in report["rows"]),
            "failure_source": sum(x["bad"]["reward"] for x in report["rows"]),
            "failures": len(report["failures"])}
        save(args.eval_output, report)
        print(json.dumps(report["summary"]), flush=True)
        if report["failures"]:
            raise RuntimeError("Incomplete contrast suffix episode")


def audit(args):
    _, _, suffix, fixed, bindings = bound(args)
    review = json.loads(args.eval_review.read_text())
    report = json.loads(args.eval_output.read_text())
    if (review.get("targets") != bindings or
            review.get("targets_sha256") != digest(bindings) or
            any(review.get(k) != v or report.get(k) != v for k, v in fixed.items()) or
            report.get("review_sha256") != file_hash(args.eval_review) or
            len(report.get("rows", [])) != 120 or report.get("failures")):
        raise ValueError("Changed or incomplete contrast suffix run")
    good = bad = 0.
    for i, (source, binding, row) in enumerate(zip(
            suffix, bindings, report["rows"], strict=True)):
        if (row["index"] != i or row["game"] != source["game"] or
                row["input_content_sha256"] != binding["input_content_sha256"] or
                row["good_source_index"] != binding["good_source_index"] or
                row["bad_source_index"] != binding["bad_source_index"]):
            raise ValueError(f"Changed target {i}")
        for arm in ("good", "bad"):
            ep = row[arm]
            if ep["status"] != "complete" or ep["steps"] > 30:
                raise ValueError(f"Invalid {arm} episode at {i}")
            replay(args.data_root / row["game"], ep)
        good += row["good"]["reward"]
        bad += row["bad"]["reward"]
    if report["summary"] != {"n": 120,
            "success_source": good, "failure_source": bad, "failures": 0}:
        raise ValueError("Changed score summary")
    return {"audit": "passed", "summary": report["summary"],
        "replayed_episodes": 240,
        "report_sha256": file_hash(args.eval_output),
        "review_sha256": file_hash(args.eval_review)}


if __name__ == "__main__":
    p = parser()
    p.add_argument("--parent-eval-review", type=Path, default=Path(
        "data/annotations/alfworld_bootstrap_own_success_v43c_suffix120_20261010.json"))
    p.add_argument("--eval-review", type=Path, default=Path(
        "data/annotations/alfworld_success_failure_contrast_v44_suffix120_20261010.json"))
    p.add_argument("--eval-output", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_success_failure_contrast_v44_suffix120_20261010.json"))
    p.add_argument("--audit-output", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_success_failure_contrast_v44_suffix120_audit_20261010.json"))
    p.add_argument("--audit-only", action="store_true")
    args = p.parse_args()
    if args.audit_only:
        if args.audit_output.exists():
            raise FileExistsError(args.audit_output)
        result = audit(args)
        save(args.audit_output, result)
        print(json.dumps(result), flush=True)
    else:
        (prepare if args.command == "prepare" else evaluate)(args)
