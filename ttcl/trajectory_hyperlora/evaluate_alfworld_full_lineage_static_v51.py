"""Audited one-attempt comparison of old hypernetwork and matched-lineage LoRA."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay


def bound(args):
    train = json.loads(args.train_report.read_text())
    train_review = json.loads(args.train_review.read_text())
    old = json.loads(args.old_report.read_text())
    old_review = json.loads(args.old_review.read_text())
    old_audit = json.loads(args.old_audit.read_text())
    if (train.get("review_sha256") != file_hash(args.train_review) or
            train.get("model_sha256") != file_hash(args.static_model) or
            train_review.get("targets_sha256") != digest(train_review.get("targets")) or
            old_audit.get("audit") != "passed" or
            old_audit.get("report_sha256") != file_hash(args.old_report) or
            old.get("review_sha256") != file_hash(args.old_review) or
            old.get("checkpoint_sha256") != file_hash(args.old_checkpoint) or
            old.get("model_config_sha256") != file_hash(args.model / "config.json") or
            len(old.get("rows", [])) != 134 or
            len(old_review.get("targets", [])) != 134):
        raise ValueError("Changed v51 train or old-hyper evidence")
    fixed = {"method": "v51 full-lineage static A+B vs old hypernetwork",
             "train_review_sha256": file_hash(args.train_review),
             "train_report_sha256": file_hash(args.train_report),
             "static_model_sha256": file_hash(args.static_model),
             "old_report_sha256": file_hash(args.old_report),
             "old_review_sha256": file_hash(args.old_review),
             "old_audit_sha256": file_hash(args.old_audit),
             "runner_sha256": file_hash(Path(__file__)),
             "actor_sha256": file_hash(Path(__file__).with_name("alfworld_zero_shot_probe.py")),
             "max_steps": 50, "max_new_tokens": 64,
             "constrain_actions": True, "actor_history_turns": 2,
             "loop_guard_max": 2}
    targets = []
    for i, (row, reviewed) in enumerate(zip(old["rows"], old_review["targets"], strict=True)):
        game = row["game"]
        if (i != row["index"] or i != reviewed["index"] or
                game != reviewed["game"] or
                row["input_content_sha256"] != reviewed["input_content_sha256"] or
                file_hash(args.data_root / game) != reviewed["game_sha256"]):
            raise ValueError(f"Changed official game {i}")
        content = {"index": i, "game": game,
                   "game_sha256": reviewed["game_sha256"],
                   "old_row_input_content_sha256": row["input_content_sha256"],
                   "static_model_sha256": fixed["static_model_sha256"],
                   "runner_sha256": fixed["runner_sha256"]}
        targets.append({**content, "input_content_sha256": digest(content),
                        "reviewed_target": True})
    return old, fixed, targets


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    _, fixed, targets = bound(args)
    save(args.review, {"protocol": "Previously exposed official valid_unseen134; same one-attempt 50-step actor constraints as old H; no validation data in LoRA training",
                       **fixed, "targets": targets, "targets_sha256": digest(targets)})
    print(json.dumps({"targets": len(targets)}), flush=True)


def checked(args):
    old, fixed, targets = bound(args)
    review = json.loads(args.review.read_text())
    if (review.get("targets") != targets or
            review.get("targets_sha256") != digest(targets) or
            any(review.get(k) != v for k, v in fixed.items())):
        raise ValueError("Changed v51 evaluation review")
    return old, fixed, targets


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    old, fixed, targets = checked(args)
    static = torch.load(args.static_model, map_location="cpu", weights_only=False)
    if static.get("review_sha256") != file_hash(args.train_review):
        raise ValueError("Changed v51 static factors")
    agent, tokenizer = load_agent(args.model, args.old_checkpoint,
                                  args.device, args.gpu_fraction)
    if len(agent.adapters) != 2 or len(static["A"]) != 2 or len(static["factors"]) != 2:
        raise ValueError("Changed adapter geometry")
    with torch.no_grad():
        for layer, a in zip(agent.adapters, static["A"], strict=True):
            if layer.a.shape != a.shape:
                raise ValueError("Changed static A geometry")
            layer.a.copy_(a.to(layer.a.device, layer.a.dtype))
    report = {"protocol": "v51 full-lineage static vs frozen old H on exposed official valid_unseen134",
              "review_sha256": file_hash(args.review), **fixed, "rows": [], "failures": []}
    for i, (target, old_row) in enumerate(zip(targets, old["rows"], strict=True)):
        episode = run_episode(agent, tokenizer, args.data_root / target["game"], {},
            adapter=False, fixed_adapter=static["factors"], device=args.device,
            max_steps=50, max_new_tokens=64, constrain_actions=True,
            actor_history_turns=2, loop_guard_max=2)
        entry = {"index": i, "game": target["game"],
                 "input_content_sha256": target["input_content_sha256"],
                 "old_hyper_reward": old_row["old_hyper_reward"],
                 "static": episode}
        if (episode["status"] != "complete" or
                episode["initial_observation"] != old_row["episode"]["initial_observation"]):
            report["failures"].append({"index": i, "status": episode["status"]})
        report["rows"].append(entry)
        report["summary"] = {"n": len(report["rows"]),
             "old_hyper": sum(r["old_hyper_reward"] for r in report["rows"]),
             "static": sum(r["static"]["reward"] for r in report["rows"]),
             "failures": len(report["failures"])}
        save(args.output, report)
        print(json.dumps(report["summary"]), flush=True)
        if report["failures"]:
            raise RuntimeError("Incomplete v51 episode")


def paired(a, b):
    wins = sum(x > y for x, y in zip(a, b, strict=True))
    losses = sum(x < y for x, y in zip(a, b, strict=True))
    n = wins + losses
    p = min(1., 2 * sum(math.comb(n, j) for j in range(min(wins, losses) + 1)) / (2 ** n)) if n else 1.
    return {"old_only": wins, "static_only": losses, "exact_two_sided_p": p}


def audit(args):
    old, fixed, targets = checked(args)
    report = json.loads(args.output.read_text())
    if (len(report.get("rows", [])) != 134 or report.get("failures") or
            report.get("review_sha256") != file_hash(args.review) or
            any(report.get(k) != v for k, v in fixed.items())):
        raise ValueError("Changed or incomplete v51 report")
    a, b = [], []
    for i, (target, row, old_row) in enumerate(zip(
            targets, report["rows"], old["rows"], strict=True)):
        if (row["index"] != i or row["game"] != target["game"] or
                row["input_content_sha256"] != target["input_content_sha256"] or
                row["old_hyper_reward"] != old_row["old_hyper_reward"] or
                row["static"]["status"] != "complete" or
                row["static"]["initial_observation"] != old_row["episode"]["initial_observation"] or
                row["static"]["steps"] > 50):
            raise ValueError(f"Changed v51 target or reward {i}")
        replay(args.data_root / row["game"], row["static"])
        a.append(row["old_hyper_reward"])
        b.append(row["static"]["reward"])
    if report["summary"] != {"n": 134, "old_hyper": sum(a),
                              "static": sum(b), "failures": 0}:
        raise ValueError("Changed v51 summary")
    return {"audit": "passed", "summary": report["summary"],
            "paired": paired(a, b), "replayed_episodes": 134,
            "report_sha256": file_hash(args.output),
            "review_sha256": file_hash(args.review)}


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "evaluate", "audit"))
    paths = {
        "model": "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507",
        "data_root": "ttcl/data/alfworld_delta",
        "train_review": "data/annotations/alfworld_full_lineage_static_v51_20261010.json",
        "train_report": "results/trajectory_hyperlora/alfworld_full_lineage_static_v51_20261010.json",
        "static_model": "results/trajectory_hyperlora/alfworld_full_lineage_static_v51_20261010.pt",
        "old_checkpoint": "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt",
        "old_report": "results/trajectory_hyperlora/alfworld_source_permutation_v42_134_20261010.json",
        "old_review": "data/annotations/alfworld_source_permutation_v42_134_20261010.json",
        "old_audit": "results/trajectory_hyperlora/alfworld_source_permutation_v42_134_audit_20261010.json",
        "review": "data/annotations/alfworld_full_lineage_static_v51_unseen134_20261010.json",
        "output": "results/trajectory_hyperlora/alfworld_full_lineage_static_v51_unseen134_20261010.json",
        "audit_output": "results/trajectory_hyperlora/alfworld_full_lineage_static_v51_unseen134_audit_20261010.json",
    }
    for k, value in paths.items():
        p.add_argument("--" + k.replace("_", "-"), type=Path, default=Path(value))
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--gpu-fraction", type=float, default=.75)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        if args.audit_output.exists():
            raise FileExistsError(args.audit_output)
        result = audit(args)
        save(args.audit_output, result)
        print(json.dumps(result), flush=True)
