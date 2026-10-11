"""Suffix test of outcome-tagged success, rotated-success, and failure sources."""

from __future__ import annotations

import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay
from ttcl.trajectory_hyperlora.evaluate_alfworld_bootstrap_own_success_v43 import rotated_source_for
from ttcl.trajectory_hyperlora.train_alfworld_outcome_tagged_contrast_v49 import (
    checked, outcome_text, parser,
)
from ttcl.trajectory_hyperlora.train_alfworld_success_failure_contrast_v44 import bad_source_for


def bound(args):
    successes, failures, suffix, train_fixed, _ = checked(args)
    train = json.loads(args.tag_report.read_text())
    parent = json.loads(args.parent_eval_review.read_text())
    if (train.get("review_sha256") != file_hash(args.tag_review) or
            train.get("model_sha256") != file_hash(args.tag_model) or
            any(train.get(k) != v for k, v in train_fixed.items()) or
            len(parent.get("targets", [])) != 120):
        raise ValueError("Changed tagged training or suffix review")
    fixed = {"method": "v49 tagged-source success/failure and success-rotation",
        "tag_review_sha256": file_hash(args.tag_review),
        "tag_report_sha256": file_hash(args.tag_report),
        "tag_model_sha256": file_hash(args.tag_model),
        "parent_eval_review_sha256": file_hash(args.parent_eval_review),
        "runner_sha256": file_hash(Path(__file__)),
        "actor_sha256": file_hash(Path(__file__).with_name("alfworld_zero_shot_probe.py")),
        "max_steps": 30, "max_new_tokens": 64,
        "constrain_actions": True, "actor_history_turns": 2,
        "loop_guard_max": 2}
    bindings = []
    for i, (row, parent_row) in enumerate(zip(suffix, parent["targets"],
                                               strict=True)):
        good = parent_row["own_success_source_index"]
        rotated = rotated_source_for(row, successes, good)
        bad = bad_source_for(row, failures)
        if (row["game"] != parent_row["game"] or
                file_hash(args.data_root / row["game"]) != parent_row["game_sha256"] or
                digest(successes[good]["source_records"]) !=
                parent_row["own_success_source_records_sha256"]):
            raise ValueError("Changed suffix source or game")
        content = {"index": i, "game": row["game"],
            "game_sha256": parent_row["game_sha256"],
            "parent_input_content_sha256": parent_row["input_content_sha256"],
            "good_source_index": good,
            "rotated_source_index": rotated,
            "bad_source_index": bad,
            "good_source_records_sha256": digest(successes[good]["source_records"]),
            "rotated_source_records_sha256": digest(successes[rotated]["source_records"]),
            "bad_source_records_sha256": digest(failures[bad]["source_records"]),
            "runner_sha256": fixed["runner_sha256"],
            "tag_model_sha256": fixed["tag_model_sha256"]}
        bindings.append({**content, "input_content_sha256": digest(content),
                         "reviewed_target": True})
    return successes, failures, suffix, fixed, bindings


def prepare(args):
    if args.eval_review.exists():
        raise FileExistsError(args.eval_review)
    _, _, _, fixed, bindings = bound(args)
    save(args.eval_review, {"protocol": "Official train-stream suffix120 after prefix-only outcome-tagged contrast. One 30-step attempt with original own-success, rotated own-success, and own-failure same-family sources; no suffix trajectory used in training or source selection.",
        **fixed, "targets": bindings, "targets_sha256": digest(bindings)})
    print(json.dumps({"reviewed_targets": len(bindings)}), flush=True)


def evaluate(args):
    if args.eval_output.exists():
        raise FileExistsError(args.eval_output)
    successes, failures, suffix, fixed, bindings = bound(args)
    review = json.loads(args.eval_review.read_text())
    if (review.get("targets") != bindings or
            review.get("targets_sha256") != digest(bindings) or
            any(review.get(k) != v for k, v in fixed.items())):
        raise ValueError("Changed reviewed tagged suffix targets")
    agent, tokenizer = load_agent(args.model, args.tag_model,
                                  args.device, args.gpu_fraction)
    report = {"protocol": review["protocol"],
        "review_sha256": file_hash(args.eval_review),
        **fixed, "rows": [], "failures": []}
    for i, (row, binding) in enumerate(zip(suffix, bindings, strict=True)):
        game = args.data_root / row["game"]
        fields = {}
        for name, sources, source_index, succeeded in (
                ("good", successes, binding["good_source_index"], True),
                ("rotated", successes, binding["rotated_source_index"], True),
                ("bad", failures, binding["bad_source_index"], False)):
            text = outcome_text(tokenizer,
                sources[source_index]["source_records"], succeeded)
            fields[name] = contextual_text_fields(agent, tokenizer, text,
                args.device, 2048, pooling="both")
        episodes = {name: run_episode(agent, tokenizer, game, fields[name],
            adapter=True, device=args.device, max_steps=30,
            max_new_tokens=64, constrain_actions=True,
            actor_history_turns=2, loop_guard_max=2)
            for name in ("good", "rotated", "bad")}
        initial = row["arms"]["base"]["initial_observation"]
        if any(ep["status"] != "complete" or
               ep["initial_observation"] != initial
               for ep in episodes.values()):
            report["failures"].append({"index": i})
        report["rows"].append({"index": i, "game": row["game"],
            "input_content_sha256": binding["input_content_sha256"],
            **episodes})
        report["summary"] = {"n": len(report["rows"]),
            **{name: sum(x[name]["reward"] for x in report["rows"])
               for name in ("good", "rotated", "bad")},
            "failures": len(report["failures"])}
        save(args.eval_output, report)
        print(json.dumps(report["summary"]), flush=True)
        if report["failures"]:
            raise RuntimeError("Incomplete tagged suffix episode")


def audit(args):
    _, _, suffix, fixed, bindings = bound(args)
    review = json.loads(args.eval_review.read_text())
    report = json.loads(args.eval_output.read_text())
    if (review.get("targets") != bindings or
            review.get("targets_sha256") != digest(bindings) or
            any(review.get(k) != v or report.get(k) != v for k, v in fixed.items()) or
            report.get("review_sha256") != file_hash(args.eval_review) or
            len(report.get("rows", [])) != 120 or report.get("failures")):
        raise ValueError("Changed or incomplete v49 suffix evaluation")
    scores = {name: 0. for name in ("good", "rotated", "bad")}
    for i, (source, binding, row) in enumerate(zip(
            suffix, bindings, report["rows"], strict=True)):
        if (row["index"] != i or row["game"] != source["game"] or
                row["input_content_sha256"] != binding["input_content_sha256"]):
            raise ValueError(f"Changed target {i}")
        for name in scores:
            ep = row[name]
            if ep["status"] != "complete" or ep["steps"] > 30:
                raise ValueError(f"Invalid episode {i}:{name}")
            replay(args.data_root / row["game"], ep)
            scores[name] += ep["reward"]
    if report["summary"] != {"n": 120, **scores, "failures": 0}:
        raise ValueError("Changed score summary")
    return {"audit": "passed", "summary": report["summary"],
        "replayed_episodes": 360,
        "report_sha256": file_hash(args.eval_output),
        "review_sha256": file_hash(args.eval_review)}


if __name__ == "__main__":
    p = parser()
    paths = {"parent_audit": "results/trajectory_hyperlora/alfworld_success_failure_contrast_v44_train_audit_20261010.json",
        "tag_review": "data/annotations/alfworld_outcome_tagged_contrast_v49_train120_20261010.json",
        "tag_model": "results/trajectory_hyperlora/alfworld_outcome_tagged_contrast_v49_20261010.pt",
        "tag_report": "results/trajectory_hyperlora/alfworld_outcome_tagged_contrast_v49_20261010.json",
        "parent_eval_review": "data/annotations/alfworld_bootstrap_own_success_v43c_suffix120_20261010.json",
        "eval_review": "data/annotations/alfworld_outcome_tagged_contrast_v49_suffix120_20261010.json",
        "eval_output": "results/trajectory_hyperlora/alfworld_outcome_tagged_contrast_v49_suffix120_20261010.json",
        "audit_output": "results/trajectory_hyperlora/alfworld_outcome_tagged_contrast_v49_suffix120_audit_20261010.json"}
    for name, value in paths.items():
        p.add_argument("--" + name.replace("_", "-"), type=Path,
                       default=Path(value))
    p.add_argument("--tag-steps", type=int, default=300)
    p.add_argument("--tag-lr", type=float, default=1e-5)
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
