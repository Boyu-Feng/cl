"""Diagnostic single-attempt all-on v40 student on exposed unseen 134.

The development-selected gate is OFF. This separate, explicitly exploratory
run measures the predicted correction head when applied once to every task;
it does not replace the gated method or provide independent held-out evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.evaluate_alfworld_failure_retry_v31b import retry
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import bounded_readout, save
from ttcl.trajectory_hyperlora.evaluate_alfworld_failure_distill_v40 import (
    checked_model, inputs, parser,
)
from ttcl.trajectory_hyperlora.train_alfworld_failure_distill_v40 import preattempt, predict
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay


def bound(args):
    samples = inputs(args)
    checked_model(args)
    dev = json.loads(args.dev_output.read_text())
    audit = json.loads(args.dev_audit_output.read_text())
    if (dev.get("selected_threshold") != 1. or audit.get("audit") != "passed"
            or audit.get("report_sha256") != file_hash(args.dev_output)):
        raise ValueError("Changed development result")
    fixed = {"method": "v40 trained correction head, diagnostic always on",
        "split": "exposed official valid_unseen", "train_only_student_sha256": file_hash(args.student),
        "train_review_sha256": file_hash(args.train_review),
        "train_report_sha256": file_hash(args.train_report),
        "dev_report_sha256": file_hash(args.dev_output),
        "dev_audit_sha256": file_hash(args.dev_audit_output),
        "old_audit_sha256": file_hash(args.old_audit),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "actor_sha256": file_hash(Path(__file__).with_name("alfworld_zero_shot_probe.py")),
        "runner_sha256": file_hash(Path(__file__)),
        "max_relative": .25, "max_steps": 50, "max_new_tokens": 64,
        "actor_history_turns": 2, "loop_guard_max": 2,
        "diagnostic_activation": "always_on_despite_dev_gate_off"}
    targets = []
    for i, (game, game_hash, _, source_hash, source_input_hash, old) in enumerate(samples):
        if file_hash(args.data_root / game) != game_hash:
            raise ValueError("Changed game content")
        content = {"index": i, "game": game, "game_sha256": game_hash,
            "expert_source_records_sha256": source_hash,
            "old_target_input_content_sha256": source_input_hash,
            "old_first_episode_sha256": digest(old),
            "runner_sha256": fixed["runner_sha256"],
            "student_sha256": fixed["train_only_student_sha256"]}
        targets.append({**content, "input_content_sha256": digest(content),
                        "reviewed_target": True})
    return samples, fixed, targets


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    samples, fixed, targets = bound(args)
    save(args.review, {"protocol": "Diagnostic all-on trained correction on previously exposed unseen 134, one attempt each; gate was selected OFF on development, so this is exploratory and not the selected method",
        **fixed, "targets": targets, "targets_sha256": digest(targets)})
    print(json.dumps({"reviewed_targets": len(samples)}), flush=True)


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    samples, fixed, targets = bound(args)
    review = json.loads(args.review.read_text())
    if (review.get("targets") != targets or review.get("targets_sha256") != digest(targets)
            or any(review.get(k) != v for k, v in fixed.items())):
        raise ValueError("Changed reviewed diagnostic targets")
    student = checked_model(args)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    report = {"protocol": review["protocol"], "review_sha256": file_hash(args.review),
              **fixed, "rows": [], "failures": []}
    for i, (game, _, records, _, _, old) in enumerate(samples):
        latent, expert, fields = preattempt(agent, tokenizer, records,
                                            old["initial_observation"], args.device)
        probability, predicted = predict(student, latent)
        applied, info = bounded_readout(agent, predicted, expert)
        episode = retry(agent, tokenizer, args.data_root / game, fields,
                        args.device, applied)
        row = {"index": i, "game": game,
            "input_content_sha256": targets[i]["input_content_sha256"],
            "old_first_episode_sha256": targets[i]["old_first_episode_sha256"],
            "old_first_reward": old["reward"],
            "prelatent_sha256": digest(latent.tolist()),
            "expert_factor_sha256": factors_hash(expert),
            "predicted_factor_sha256": factors_hash(predicted),
            "applied_factor_sha256": factors_hash(applied),
            "read_info": info, "gate_probability": probability,
            "episode": episode}
        if (episode["status"] != "complete" or
                episode["initial_observation"] != old["initial_observation"]):
            report["failures"].append({"index": i, "status": episode["status"]})
        report["rows"].append(row)
        report["summary"] = {"n": len(report["rows"]),
            "old_first_wins": sum(x["old_first_reward"] for x in report["rows"]),
            "diagnostic_all_on_wins": sum(x["episode"]["reward"] for x in report["rows"]),
            "failures": len(report["failures"])}
        save(args.output, report)
        print(json.dumps(report["summary"]), flush=True)
        if report["failures"]:
            raise RuntimeError("Incomplete diagnostic episode")


def audit(args):
    samples, fixed, targets = bound(args)
    review = json.loads(args.review.read_text())
    report = json.loads(args.output.read_text())
    if (review.get("targets") != targets or review.get("targets_sha256") != digest(targets)
            or any(review.get(k) != v or report.get(k) != v for k, v in fixed.items())
            or report.get("review_sha256") != file_hash(args.review)
            or len(report.get("rows", [])) != 134 or report.get("failures")):
        raise ValueError("Changed or incomplete diagnostic run")
    student = checked_model(args)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    old_wins = new_wins = 0
    for i, ((game, _, records, _, _, old), row, binding) in enumerate(zip(
            samples, report["rows"], targets, strict=True)):
        if (row["index"] != i or row["game"] != game or
                row["input_content_sha256"] != binding["input_content_sha256"] or
                row["old_first_episode_sha256"] != digest(old) or
                row["old_first_reward"] != old["reward"]):
            raise ValueError(f"Changed target at {i}")
        latent, expert, _ = preattempt(agent, tokenizer, records,
                                      old["initial_observation"], args.device)
        probability, predicted = predict(student, latent)
        applied, info = bounded_readout(agent, predicted, expert)
        if (row["prelatent_sha256"] != digest(latent.tolist()) or
                row["expert_factor_sha256"] != factors_hash(expert) or
                row["predicted_factor_sha256"] != factors_hash(predicted) or
                row["applied_factor_sha256"] != factors_hash(applied) or
                row["read_info"] != info or
                abs(row["gate_probability"] - probability) > 1e-7 or
                info["applied_effective_norm"] > .25 * info["reference_effective_norm"] + 1e-6):
            raise ValueError(f"Correction changed at {i}")
        ep = row["episode"]
        if ep["status"] != "complete" or ep["steps"] > 50 or ep["initial_observation"] != old["initial_observation"]:
            raise ValueError(f"Invalid episode at {i}")
        replay(args.data_root / game, ep)
        old_wins += old["reward"]
        new_wins += ep["reward"]
    summary = {"n": 134, "old_first_wins": old_wins,
        "diagnostic_all_on_wins": new_wins, "failures": 0}
    if report["summary"] != summary:
        raise ValueError("Official score summary changed")
    return {"audit": "passed", "summary": summary,
        "replayed_episodes": 134, "report_sha256": file_hash(args.output),
        "review_sha256": file_hash(args.review)}


if __name__ == "__main__":
    p = parser()
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--audit-output", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_failure_distill_v40_all_on_134_audit_20261010.json"))
    args = p.parse_args()
    if args.split != "unseen":
        raise ValueError("Diagnostic split must be unseen")
    if args.audit_only:
        if args.audit_output.exists():
            raise FileExistsError(args.audit_output)
        result = audit(args)
        save(args.audit_output, result)
        print(json.dumps(result), flush=True)
    else:
        (prepare if args.command == "prepare" else evaluate)(args)
