"""One-attempt official134 comparison of fresh own-success H and direct LoRA."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_source_fields
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay
from ttcl.trajectory_hyperlora.alfworld_bootstrap_own_success_v43 import (
    checked as bootstrap_checked, parser,
)
from ttcl.trajectory_hyperlora.evaluate_alfworld_bootstrap_own_success_v43 import source_for


def bound(args):
    _, successes, _, train_fixed, _, _ = bootstrap_checked(args)
    train = json.loads(args.train_report.read_text())
    contrast = json.loads(args.contrast_report.read_text())
    full = json.loads(args.full_static_report.read_text())
    original = json.loads(args.original_review.read_text())
    old = json.loads(args.original_report.read_text())
    old_audit = json.loads(args.original_audit.read_text())
    if (train.get("review_sha256") != file_hash(args.review) or
            train.get("hyper_model_sha256") != file_hash(args.hyper_model) or
            any(train.get(k) != v for k, v in train_fixed.items()) or
            contrast.get("model_sha256") != file_hash(args.contrast_model) or
            contrast.get("bootstrap_hyper_model_sha256") != file_hash(args.hyper_model) or
            full.get("model_sha256") != file_hash(args.full_static_model) or
            full.get("bootstrap_review_sha256") != file_hash(args.review) or
            old_audit.get("audit") != "passed" or
            old_audit.get("report_sha256") != file_hash(args.original_report) or
            len(original.get("targets", [])) != 134 or
            len(old.get("rows", [])) != 134):
        raise ValueError("Changed model lineage or official134 reference")
    fixed = {"method": "v47 fresh own-success H vs contrast H vs full static LoRA",
        "bootstrap_review_sha256": file_hash(args.review),
        "bootstrap_report_sha256": file_hash(args.train_report),
        "hyper_model_sha256": file_hash(args.hyper_model),
        "contrast_model_sha256": file_hash(args.contrast_model),
        "contrast_report_sha256": file_hash(args.contrast_report),
        "full_static_model_sha256": file_hash(args.full_static_model),
        "full_static_report_sha256": file_hash(args.full_static_report),
        "original_review_sha256": file_hash(args.original_review),
        "original_report_sha256": file_hash(args.original_report),
        "runner_sha256": file_hash(Path(__file__)),
        "actor_sha256": file_hash(Path(__file__).with_name("alfworld_zero_shot_probe.py")),
        "max_steps": 50, "max_new_tokens": 64,
        "constrain_actions": True, "actor_history_turns": 2,
        "loop_guard_max": 2}
    targets = []
    for i, (row, original_row) in enumerate(zip(
            original["targets"], old["rows"], strict=True)):
        if (row["index"] != i or original_row["index"] != i or
                row["game"] != original_row["game"] or
                row["game_sha256"] != file_hash(args.data_root / row["game"])):
            raise ValueError(f"Changed official target {i}")
        source_index = source_for(row, successes)
        content = {"index": i, "game": row["game"],
            "game_sha256": row["game_sha256"],
            "original_input_content_sha256": row["input_content_sha256"],
            "own_success_source_index": source_index,
            "own_success_source_records_sha256": digest(successes[source_index]["source_records"]),
            "runner_sha256": fixed["runner_sha256"],
            "hyper_model_sha256": fixed["hyper_model_sha256"]}
        targets.append({**content, "input_content_sha256": digest(content),
                        "reviewed_target": True})
    return successes, old, fixed, targets


def prepare(args):
    if args.eval_review.exists():
        raise FileExistsError(args.eval_review)
    _, _, fixed, targets = bound(args)
    save(args.eval_review, {"protocol": "Official valid_unseen134, previously exposed. Fresh models trained exclusively on own successes in official train prefix120; one 50-step attempt per arm; no unseen trajectory in training or source choice.",
        **fixed, "targets": targets, "targets_sha256": digest(targets)})
    print(json.dumps({"reviewed_targets": len(targets)}), flush=True)


def validate_review(args, fixed, targets):
    review = json.loads(args.eval_review.read_text())
    if (review.get("targets") != targets or
            review.get("targets_sha256") != digest(targets) or
            any(review.get(k) != v for k, v in fixed.items())):
        raise ValueError("Changed v47 review")
    return review


def evaluate(args):
    if args.eval_output.exists():
        raise FileExistsError(args.eval_output)
    successes, old, fixed, targets = bound(args)
    review = validate_review(args, fixed, targets)
    agent, tokenizer = load_agent(args.model, args.hyper_model,
                                  args.device, args.gpu_fraction)
    fresh_state = {name: p.detach().clone() for name, p in agent.named_parameters()
                   if p.requires_grad}
    contrast_state = torch.load(args.contrast_model, map_location="cpu",
                                weights_only=True)["trainable_state"]
    static = torch.load(args.full_static_model, map_location="cpu",
                        weights_only=False)
    if (fresh_state.keys() != contrast_state.keys() or
            static["review_sha256"] != file_hash(args.review)):
        raise ValueError("Changed adapter parameters")
    report = {"protocol": review["protocol"],
        "review_sha256": file_hash(args.eval_review),
        **fixed, "rows": [], "failures": []}
    for i, (target, old_row) in enumerate(zip(targets, old["rows"], strict=True)):
        game = args.data_root / target["game"]
        source = successes[target["own_success_source_index"]]
        fields = contextual_source_fields(agent, tokenizer,
            source["source_records"], args.device, 2048, pooling="both")
        fresh = run_episode(agent, tokenizer, game, fields,
            adapter=True, device=args.device, max_steps=50,
            max_new_tokens=64, constrain_actions=True,
            actor_history_turns=2, loop_guard_max=2)
        with torch.no_grad():
            for name, p in agent.named_parameters():
                if p.requires_grad:
                    p.copy_(contrast_state[name].to(p.device, p.dtype))
        contrast = run_episode(agent, tokenizer, game, fields,
            adapter=True, device=args.device, max_steps=50,
            max_new_tokens=64, constrain_actions=True,
            actor_history_turns=2, loop_guard_max=2)
        with torch.no_grad():
            for name, p in agent.named_parameters():
                if p.requires_grad:
                    p.copy_(fresh_state[name])
            for layer, a in zip(agent.adapters, static["A"], strict=True):
                layer.a.copy_(a.to(layer.a.device, layer.a.dtype))
        full_static = run_episode(agent, tokenizer, game, {},
            adapter=False, fixed_adapter=static["factors"],
            device=args.device, max_steps=50, max_new_tokens=64,
            constrain_actions=True, actor_history_turns=2,
            loop_guard_max=2)
        with torch.no_grad():
            for name, p in agent.named_parameters():
                if p.requires_grad:
                    p.copy_(fresh_state[name])
        entry = {"index": i, "game": target["game"],
            "input_content_sha256": target["input_content_sha256"],
            "own_success_source_index": target["own_success_source_index"],
            "old_hyper_reward": old_row["old_hyper_reward"],
            "fresh_hyper": fresh, "contrast_hyper": contrast,
            "full_static": full_static}
        if (any(ep["status"] != "complete" or
                ep["initial_observation"] != fresh["initial_observation"]
                for ep in (fresh, contrast, full_static))):
            report["failures"].append({"index": i})
        report["rows"].append(entry)
        report["summary"] = {"n": len(report["rows"]),
            "old_hyper": sum(x["old_hyper_reward"] for x in report["rows"]),
            "fresh_hyper": sum(x["fresh_hyper"]["reward"] for x in report["rows"]),
            "contrast_hyper": sum(x["contrast_hyper"]["reward"] for x in report["rows"]),
            "full_static": sum(x["full_static"]["reward"] for x in report["rows"]),
            "failures": len(report["failures"])}
        save(args.eval_output, report)
        print(json.dumps(report["summary"]), flush=True)
        if report["failures"]:
            raise RuntimeError("Incomplete v47 episode")


def audit(args):
    _, old, fixed, targets = bound(args)
    validate_review(args, fixed, targets)
    report = json.loads(args.eval_output.read_text())
    if (len(report.get("rows", [])) != 134 or report.get("failures") or
            report.get("review_sha256") != file_hash(args.eval_review) or
            any(report.get(k) != v for k, v in fixed.items())):
        raise ValueError("Changed or incomplete v47 report")
    scores = {"old_hyper": 0., "fresh_hyper": 0.,
              "contrast_hyper": 0., "full_static": 0.}
    for i, (target, row, old_row) in enumerate(zip(
            targets, report["rows"], old["rows"], strict=True)):
        if (row["index"] != i or row["game"] != target["game"] or
                row["input_content_sha256"] != target["input_content_sha256"] or
                row["own_success_source_index"] != target["own_success_source_index"] or
                row["old_hyper_reward"] != old_row["old_hyper_reward"]):
            raise ValueError(f"Changed v47 target {i}")
        scores["old_hyper"] += row["old_hyper_reward"]
        for arm in ("fresh_hyper", "contrast_hyper", "full_static"):
            ep = row[arm]
            if ep["status"] != "complete" or ep["steps"] > 50:
                raise ValueError(f"Invalid episode {i}:{arm}")
            replay(args.data_root / row["game"], ep)
            scores[arm] += ep["reward"]
    if report["summary"] != {"n": 134, **scores, "failures": 0}:
        raise ValueError("Changed v47 score summary")
    return {"audit": "passed", "summary": report["summary"],
        "replayed_episodes": 402,
        "report_sha256": file_hash(args.eval_output),
        "review_sha256": file_hash(args.eval_review)}


if __name__ == "__main__":
    p = parser()
    paths = {
        "contrast_model": "results/trajectory_hyperlora/alfworld_success_failure_contrast_v44_20261010.pt",
        "contrast_report": "results/trajectory_hyperlora/alfworld_success_failure_contrast_v44_20261010.json",
        "full_static_model": "results/trajectory_hyperlora/alfworld_bootstrap_own_success_v43c_full_static_20261010.pt",
        "full_static_report": "results/trajectory_hyperlora/alfworld_bootstrap_own_success_v43c_full_static_20261010.json",
        "original_review": "data/annotations/alfworld_source_permutation_v42_134_20261010.json",
        "original_report": "results/trajectory_hyperlora/alfworld_source_permutation_v42_134_20261010.json",
        "original_audit": "results/trajectory_hyperlora/alfworld_source_permutation_v42_134_audit_20261010.json",
        "eval_review": "data/annotations/alfworld_fresh_own_success_v47_unseen134_20261010.json",
        "eval_output": "results/trajectory_hyperlora/alfworld_fresh_own_success_v47_unseen134_20261010.json",
        "audit_output": "results/trajectory_hyperlora/alfworld_fresh_own_success_v47_unseen134_audit_20261010.json"}
    for name, value in paths.items():
        p.add_argument("--" + name.replace("_", "-"), type=Path, default=Path(value))
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
