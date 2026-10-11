"""Paired suffix evaluation of models trained only on own prefix successes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_source_fields
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay
from ttcl.trajectory_hyperlora.alfworld_bootstrap_own_success_v43 import (
    checked as bootstrap_checked, parser as bootstrap_parser,
)


def source_for(target, successes):
    eligible = [(i, source) for i, source in enumerate(successes)
                if source["family"] == target["family"]]
    if not eligible:
        raise ValueError("No own-success source in task family")
    return min(eligible, key=lambda pair: digest([
        "v43_suffix_source", target["game"], pair[1]["game"]]))[0]


def rotated_source_for(target, successes, original_index):
    same = [i for i, source in enumerate(successes)
            if source["family"] == target["family"] and i != original_index]
    if not same:
        same = [i for i in range(len(successes)) if i != original_index]
    return min(same, key=lambda i: digest([
        "v43_rotated_source", target["game"], successes[i]["game"]]))


def bound(args):
    _, successes, suffix, train_fixed, _, targets = bootstrap_checked(args)
    train = json.loads(args.train_report.read_text())
    if (train.get("review_sha256") != file_hash(args.review) or
            train.get("hyper_model_sha256") != file_hash(args.hyper_model) or
            train.get("static_model_sha256") != file_hash(args.static_model) or
            any(train.get(k) != v for k, v in train_fixed.items())):
        raise ValueError("Changed bootstrap training")
    full_report = json.loads(args.full_static_report.read_text())
    if (full_report.get("bootstrap_review_sha256") != file_hash(args.review) or
            full_report.get("model_sha256") != file_hash(args.full_static_model) or
            full_report.get("action_targets") != train.get("static_actions") or
            full_report.get("steps") != train_fixed["steps"] or
            full_report.get("seed") != train_fixed["seed"]):
        raise ValueError("Changed matched full-LoRA training")
    fixed = {"method": "v43 own-success-prefix fresh hypernetwork vs static LoRA",
        "bootstrap_review_sha256": file_hash(args.review),
        "training_report_sha256": file_hash(args.train_report),
        "hyper_model_sha256": file_hash(args.hyper_model),
        "static_model_sha256": file_hash(args.static_model),
        "full_static_model_sha256": file_hash(args.full_static_model),
        "full_static_report_sha256": file_hash(args.full_static_report),
        "source_report_sha256": file_hash(args.source_report),
        "runner_sha256": file_hash(Path(__file__)),
        "actor_sha256": file_hash(Path(__file__).with_name("alfworld_zero_shot_probe.py")),
        "max_steps": 30, "max_new_tokens": 64,
        "constrain_actions": True, "actor_history_turns": 2,
        "loop_guard_max": 2}
    bindings = []
    for i, (target, row) in enumerate(zip(targets, suffix, strict=True)):
        j = source_for(row, successes)
        content = {"index": i, "game": row["game"],
            "game_sha256": target["game_sha256"],
            "bootstrap_target_input_content_sha256": target["input_content_sha256"],
            "own_success_source_index": j,
            "own_success_source_records_sha256": digest(successes[j]["source_records"]),
            "rotated_source_index": rotated_source_for(row, successes, j),
            "rotated_source_records_sha256": digest(successes[
                rotated_source_for(row, successes, j)]["source_records"]),
            "old_base_episode_sha256": digest(row["arms"]["base"]),
            "runner_sha256": fixed["runner_sha256"],
            "hyper_model_sha256": fixed["hyper_model_sha256"]}
        bindings.append({**content, "input_content_sha256": digest(content),
                         "reviewed_target": True})
    return successes, suffix, fixed, bindings


def prepare(args):
    if args.eval_review.exists():
        raise FileExistsError(args.eval_review)
    _, _, fixed, bindings = bound(args)
    save(args.eval_review, {"protocol": "Previously exposed official train stream suffix after first120 base-own-success bootstrap; one 30-step attempt per newly trained arm plus same-family rotated-source ablation; no suffix actions or rewards used in training; descriptive mechanism test",
        **fixed, "targets": bindings, "targets_sha256": digest(bindings)})
    print(json.dumps({"reviewed_suffix_targets": len(bindings)}), flush=True)


def load_models(args):
    agent, tokenizer = load_agent(args.model, args.hyper_model,
                                  args.device, args.gpu_fraction)
    if not agent.task_conditioned or agent.task_context_scope != "current":
        raise ValueError("Fresh hypernetwork has changed architecture")
    static = torch.load(args.static_model, map_location="cpu", weights_only=False)
    full = torch.load(args.full_static_model, map_location="cpu", weights_only=False)
    if (static["review_sha256"] != file_hash(args.review) or
            static["trainer_sha256"] != file_hash(Path(__file__).with_name(
                "alfworld_bootstrap_own_success_v43.py")) or
            len(static["A"]) != len(agent.adapters) or
            len(static["factors"]) != len(agent.adapters)):
        raise ValueError("Changed fresh static LoRA")
    if (full["review_sha256"] != file_hash(args.review) or
            full["trainer_sha256"] != file_hash(Path(__file__).with_name(
                "train_alfworld_bootstrap_full_lora_v43c.py")) or
            len(full["A"]) != len(agent.adapters) or
            len(full["factors"]) != len(agent.adapters)):
        raise ValueError("Changed full A+B static LoRA")
    trained_a = [layer.a.detach().clone() for layer in agent.adapters]
    return agent, tokenizer, static, full, trained_a


def evaluate(args):
    if args.eval_output.exists():
        raise FileExistsError(args.eval_output)
    successes, suffix, fixed, bindings = bound(args)
    review = json.loads(args.eval_review.read_text())
    if (review.get("targets") != bindings or
            review.get("targets_sha256") != digest(bindings) or
            any(review.get(k) != v for k, v in fixed.items())):
        raise ValueError("Changed reviewed suffix targets")
    agent, tokenizer, static, full, trained_a = load_models(args)
    report = {"protocol": review["protocol"],
        "review_sha256": file_hash(args.eval_review),
        **fixed, "rows": [], "failures": []}
    for i, (row, binding) in enumerate(zip(suffix, bindings, strict=True)):
        game = args.data_root / row["game"]
        j = binding["own_success_source_index"]
        base = run_episode(agent, tokenizer, game, {}, adapter=False,
            device=args.device, max_steps=30, max_new_tokens=64,
            constrain_actions=True, actor_history_turns=2, loop_guard_max=2)
        fields = contextual_source_fields(agent, tokenizer,
            successes[j]["source_records"], args.device, 2048, pooling="both")
        hyper = run_episode(agent, tokenizer, game, fields, adapter=True,
            device=args.device, max_steps=30, max_new_tokens=64,
            constrain_actions=True, actor_history_turns=2, loop_guard_max=2)
        rotated_j = binding["rotated_source_index"]
        rotated_fields = contextual_source_fields(agent, tokenizer,
            successes[rotated_j]["source_records"], args.device, 2048,
            pooling="both")
        rotated = run_episode(agent, tokenizer, game, rotated_fields,
            adapter=True, device=args.device, max_steps=30,
            max_new_tokens=64, constrain_actions=True,
            actor_history_turns=2, loop_guard_max=2)
        with torch.no_grad():
            for layer, a in zip(agent.adapters, static["A"], strict=True):
                layer.a.copy_(a.to(layer.a.device, layer.a.dtype))
        direct = run_episode(agent, tokenizer, game, {}, adapter=False,
            fixed_adapter=static["factors"], device=args.device,
            max_steps=30, max_new_tokens=64, constrain_actions=True,
            actor_history_turns=2, loop_guard_max=2)
        with torch.no_grad():
            for layer, a in zip(agent.adapters, full["A"], strict=True):
                layer.a.copy_(a.to(layer.a.device, layer.a.dtype))
        full_direct = run_episode(agent, tokenizer, game, {}, adapter=False,
            fixed_adapter=full["factors"], device=args.device,
            max_steps=30, max_new_tokens=64, constrain_actions=True,
            actor_history_turns=2, loop_guard_max=2)
        with torch.no_grad():
            for layer, a in zip(agent.adapters, trained_a, strict=True):
                layer.a.copy_(a)
        entry = {"index": i, "game": row["game"],
            "input_content_sha256": binding["input_content_sha256"],
            "source_index": j, "rotated_source_index": rotated_j,
            "old_base_reward": row["arms"]["base"]["reward"],
            "base": base,
            "hyper": hyper, "rotated": rotated, "static": direct,
            "full_static": full_direct}
        if (base["status"] != "complete" or hyper["status"] != "complete" or
                rotated["status"] != "complete" or
                direct["status"] != "complete" or full_direct["status"] != "complete" or
                base["initial_observation"] != row["arms"]["base"]["initial_observation"] or
                hyper["initial_observation"] != row["arms"]["base"]["initial_observation"] or
                rotated["initial_observation"] != row["arms"]["base"]["initial_observation"] or
                direct["initial_observation"] != row["arms"]["base"]["initial_observation"] or
                full_direct["initial_observation"] != row["arms"]["base"]["initial_observation"]):
            report["failures"].append({"index": i})
        report["rows"].append(entry)
        report["summary"] = {"n": len(report["rows"]),
            "historical_base": sum(x["old_base_reward"] for x in report["rows"]),
            "base": sum(x["base"]["reward"] for x in report["rows"]),
            "fresh_hyper": sum(x["hyper"]["reward"] for x in report["rows"]),
            "rotated_source": sum(x["rotated"]["reward"] for x in report["rows"]),
            "fresh_static": sum(x["static"]["reward"] for x in report["rows"]),
            "full_static": sum(x["full_static"]["reward"] for x in report["rows"]),
            "failures": len(report["failures"])}
        save(args.eval_output, report)
        print(json.dumps(report["summary"]), flush=True)
        if report["failures"]:
            raise RuntimeError("Incomplete suffix episode")


def audit(args):
    _, suffix, fixed, bindings = bound(args)
    review = json.loads(args.eval_review.read_text())
    report = json.loads(args.eval_output.read_text())
    if (review.get("targets") != bindings or
            review.get("targets_sha256") != digest(bindings) or
            any(review.get(k) != v or report.get(k) != v for k, v in fixed.items()) or
            report.get("review_sha256") != file_hash(args.eval_review) or
            len(report.get("rows", [])) != 120 or report.get("failures")):
        raise ValueError("Changed or incomplete suffix evaluation")
    scores = {"historical_base": 0., "base": 0.,
              "fresh_hyper": 0., "rotated_source": 0.,
              "fresh_static": 0., "full_static": 0.}
    for i, (source, binding, row) in enumerate(zip(
            suffix, bindings, report["rows"], strict=True)):
        if (row["index"] != i or row["game"] != source["game"] or
                row["input_content_sha256"] != binding["input_content_sha256"] or
                row["source_index"] != binding["own_success_source_index"] or
                row["rotated_source_index"] != binding["rotated_source_index"] or
                row["old_base_reward"] != source["arms"]["base"]["reward"]):
            raise ValueError(f"Changed target at {i}")
        for arm in ("base", "hyper", "rotated", "static", "full_static"):
            ep = row[arm]
            if ep["status"] != "complete" or ep["steps"] > 30:
                raise ValueError(f"Invalid {arm} episode at {i}")
            replay(args.data_root / row["game"], ep)
            scores[{"base": "base", "hyper": "fresh_hyper",
                    "rotated": "rotated_source",
                    "static": "fresh_static", "full_static": "full_static"}[arm]] += ep["reward"]
        scores["historical_base"] += row["old_base_reward"]
    if report["summary"] != {"n": 120, **scores, "failures": 0}:
        raise ValueError("Changed official summary")
    return {"audit": "passed", "summary": report["summary"],
        "replayed_episodes": 600,
        "report_sha256": file_hash(args.eval_output),
        "review_sha256": file_hash(args.eval_review)}


def parser():
    p = bootstrap_parser()
    p.add_argument("--eval-review", type=Path, default=Path(
        "data/annotations/alfworld_bootstrap_own_success_v43_suffix120_20261010.json"))
    p.add_argument("--eval-output", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_bootstrap_own_success_v43_suffix120_20261010.json"))
    p.add_argument("--audit-output", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_bootstrap_own_success_v43_suffix120_audit_20261010.json"))
    p.add_argument("--full-static-model", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_bootstrap_own_success_v43c_full_static_20261010.pt"))
    p.add_argument("--full-static-report", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_bootstrap_own_success_v43c_full_static_20261010.json"))
    return p


if __name__ == "__main__":
    p = parser()
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
