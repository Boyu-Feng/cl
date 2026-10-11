"""One-attempt evaluation of the train-only failure correction student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.evaluate_alfworld_failure_retry_v31b import retry
from ttcl.trajectory_hyperlora.alfworld_retry_delta_next_v35 import read_inputs
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import bounded_readout, save
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v39_134 import bound as unseen_bound, parser as old_parser
from ttcl.trajectory_hyperlora.train_alfworld_failure_distill_v40 import preattempt, predict


THRESHOLDS = (0., .4, .5, .6, .7, .8, 1.)


def inputs(args):
    if args.split == "dev":
        pairs, first, _ = read_inputs(args)
        return [(p["target_game"], p["target_game_sha256"],
                 p["source_records"], p["source_records_sha256"],
                 p["input_content_sha256"], row["new"])
                for p, row in zip(pairs, first["games"], strict=True)]
    targets, first, _, _ = unseen_bound(args)
    return [(p["game"], p["game_sha256"], p["source_records"],
             p["source_records_sha256"], p["input_content_sha256"], row["lora"])
            for p, row in zip(targets, first, strict=True)]


def checked_model(args):
    train = json.loads(args.train_report.read_text())
    model = torch.load(args.student, map_location="cpu", weights_only=False)
    if (train["model_sha256"] != file_hash(args.student)
            or train["train_review_sha256"] != file_hash(args.train_review)
            or model["train_review_sha256"] != file_hash(args.train_review)
            or model["checkpoint_sha256"] != file_hash(args.checkpoint)
            or model["trainer_sha256"] != train["trainer_sha256"]
            or train["n"] != 48 or train["failure_teachers"] != 21):
        raise ValueError("Changed or incomplete train-only student")
    return model


def binding(args, samples):
    checked_model(args)
    fixed = {"method": "v40 frozen train-only predicted failure B correction",
        "split": args.split, "pairs_sha256": file_hash(args.pairs),
        "train_review_sha256": file_hash(args.train_review),
        "train_report_sha256": file_hash(args.train_report),
        "student_sha256": file_hash(args.student),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "runner_sha256": file_hash(Path(__file__)),
        "actor_sha256": file_hash(Path(__file__).with_name("alfworld_zero_shot_probe.py")),
        "max_relative": .25, "max_steps": 50, "max_new_tokens": 64,
        "actor_history_turns": 2, "loop_guard_max": 2,
        "threshold_grid": list(THRESHOLDS)}
    if args.split == "dev":
        fixed.update(first_report_sha256=file_hash(args.first_report),
                     retry_audit_sha256=file_hash(args.retry_audit))
    else:
        old_audit = json.loads(args.old_audit.read_text())
        if old_audit["summary"]["overall"]["old_lora"] != 68:
            raise ValueError("Changed old unseen baseline")
        fixed.update(old_audit_sha256=file_hash(args.old_audit),
                     v32_audit_sha256=file_hash(args.v32_audit))
        if args.dev_output.exists():
            dev = json.loads(args.dev_output.read_text())
            audit = json.loads(args.dev_audit_output.read_text())
            if (audit.get("audit") != "passed"
                    or audit.get("report_sha256") != file_hash(args.dev_output)
                    or dev.get("selected_threshold") not in THRESHOLDS):
                raise ValueError("Changed or unaudited development selection")
            fixed.update(dev_report_sha256=file_hash(args.dev_output),
                         dev_audit_sha256=file_hash(args.dev_audit_output),
                         selected_threshold=dev["selected_threshold"])
        else:
            raise ValueError("Missing development selection")
    targets = []
    for i, (game, game_hash, _, source_hash, input_hash, old) in enumerate(samples):
        if file_hash(args.data_root / game) != game_hash:
            raise ValueError("Changed official task content")
        content = {"index": i, "game": game, "game_sha256": game_hash,
            "expert_source_records_sha256": source_hash,
            "old_target_input_content_sha256": input_hash,
            "old_first_episode_sha256": digest(old),
            "runner_sha256": fixed["runner_sha256"],
            "student_sha256": fixed["student_sha256"]}
        targets.append({**content, "input_content_sha256": digest(content),
                        "reviewed_target": True})
    return fixed, targets


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    samples = inputs(args)
    fixed, targets = binding(args, samples)
    save(args.review, {"protocol": "Frozen train-only student; pre-attempt latent predicts B correction and failure gate; one official attempt per task; dev compares predeclared thresholds, unseen uses selected threshold; no target failure is read before action",
        **fixed, "targets": targets, "targets_sha256": digest(targets)})
    print(json.dumps({"split": args.split, "targets": len(targets)}), flush=True)


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    samples = inputs(args)
    fixed, targets = binding(args, samples)
    review = json.loads(args.review.read_text())
    if (review.get("targets") != targets or review.get("targets_sha256") != digest(targets)
            or any(review.get(k) != v for k, v in fixed.items())):
        raise ValueError("Changed reviewed evaluation targets")
    model = checked_model(args)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    report = {"protocol": review["protocol"], "review_sha256": file_hash(args.review),
              **fixed, "rows": [], "failures": []}
    for i, (game, _, records, _, _, old) in enumerate(samples):
        latent, expert, fields = preattempt(agent, tokenizer, records,
                                            old["initial_observation"], args.device)
        gate_probability, predicted = predict(model, latent)
        applied, read_info = bounded_readout(agent, predicted, expert)
        selected = (args.split == "dev" or
                    gate_probability >= fixed["selected_threshold"])
        episode = retry(agent, tokenizer, args.data_root / game,
                        fields, args.device, applied) if selected else None
        row = {"index": i, "game": game,
            "input_content_sha256": targets[i]["input_content_sha256"],
            "old_first_episode_sha256": targets[i]["old_first_episode_sha256"],
            "old_first_reward": old["reward"],
            "prelatent_sha256": digest(latent.tolist()),
            "expert_factor_sha256": factors_hash(expert),
            "predicted_factor_sha256": factors_hash(predicted),
            "applied_factor_sha256": factors_hash(applied),
            "read_info": read_info, "gate_probability": gate_probability,
            "activated": selected, "episode": episode}
        if episode is not None and (episode["status"] != "complete" or
                episode["initial_observation"] != old["initial_observation"]):
            report["failures"].append({"index": i, "status": episode["status"]})
        report["rows"].append(row)
        report["summary"] = {"n": len(report["rows"]),
            "old_first_wins": sum(x["old_first_reward"] for x in report["rows"]),
            "activated": sum(x["activated"] for x in report["rows"]),
            "modified_attempt_wins": sum(x["episode"]["reward"] for x in report["rows"]
                                         if x["episode"] is not None),
            "failures": len(report["failures"])}
        save(args.output, report)
        print(json.dumps({"split": args.split, **report["summary"]}), flush=True)
        if report["failures"]:
            raise RuntimeError("Incomplete official v40 episode")
    if args.split == "dev":
        scores = {}
        for threshold in THRESHOLDS:
            scores[str(threshold)] = sum(
                x["episode"]["reward"] if x["gate_probability"] >= threshold
                else x["old_first_reward"] for x in report["rows"])
        selected = max(THRESHOLDS, key=lambda x: (scores[str(x)], x))
        report.update(threshold_scores=scores, selected_threshold=selected)
    else:
        report["summary"]["gated_one_attempt_wins"] = sum(
            x["episode"]["reward"] if x["activated"] else x["old_first_reward"]
            for x in report["rows"])
    save(args.output, report)
    print(json.dumps({"final_summary": report["summary"],
        "selected_threshold": report.get("selected_threshold")}), flush=True)


def parser():
    p = old_parser()
    p.add_argument("--split", required=True, choices=("dev", "unseen"))
    p.add_argument("--pairs", type=Path, default=Path(
        "data/annotations/alf_reward_router_train72_reviewed_20261007.json"))
    p.add_argument("--first-report", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_parent_loop2_dev24_20261007.json"))
    p.add_argument("--retry-report", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_failure_retry_v31b_dev24_20261009.json"))
    p.add_argument("--retry-audit", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_failure_retry_v31b_dev24_audit_20261009.json"))
    p.add_argument("--train-review", type=Path, default=Path(
        "data/annotations/alfworld_failure_distill_v40_train48_20261010.json"))
    p.add_argument("--train-report", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_failure_distill_v40_train48_20261010.json"))
    p.add_argument("--student", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_failure_distill_v40_train48_20261010.pt"))
    p.add_argument("--dev-output", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_failure_distill_v40_dev24_20261010.json"))
    p.add_argument("--dev-audit-output", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_failure_distill_v40_dev24_audit_20261010.json"))
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    (prepare if args.command == "prepare" else evaluate)(args)
