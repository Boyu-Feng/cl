"""Evaluate a frozen reward-trained LoRA policy on fresh valid_seen games."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.paired_reward_gate import visible_family
from ttcl.trajectory_hyperlora.alfworld_frozen_eval_probe import mean_adapter
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.prepare_alf_next_task import validate_review
from ttcl.trajectory_hyperlora.prepare_alf_rl_holdout import validate
from ttcl.trajectory_hyperlora.relational_router_pilot import tokenize_records
from ttcl.trajectory_hyperlora.train_alfworld_reward_policy import (
    ARMS, AdapterPolicy, features, source_latents,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows: list[dict]) -> dict:
    complete = [row for row in rows if all(
        row.get(arm, {}).get("status") == "complete"
        for arm in row["available_arms"])]
    return {"completed": len(complete),
            "policy_successes": sum(row[row["chosen_arm"]]["reward"] for row in complete),
            "base_successes": sum(row["base"]["reward"] for row in complete),
            "mean_successes": sum(row["mean"]["reward"] for row in complete),
            "source_supported": sum("source" in row["available_arms"] for row in complete),
            "source_successes": sum(row["source"]["reward"] for row in complete
                                    if "source" in row["available_arms"]),
            "policy_only_vs_base": sum(row[row["chosen_arm"]]["reward"] == 1
                                       and row["base"]["reward"] == 0
                                       for row in complete),
            "base_only_vs_policy": sum(row[row["chosen_arm"]]["reward"] == 0
                                       and row["base"]["reward"] == 1
                                       for row in complete)}


def run(args: argparse.Namespace) -> dict:
    if args.output.exists() or args.start_index < 0 or args.limit < 0:
        raise ValueError("Fresh test output and nonnegative selection required")
    manifest = json.loads(args.candidates.read_text())
    review = json.loads(args.review.read_text())
    targets = validate(manifest, review, args.plan, args.data_root)
    historical = validate_review(
        json.loads(args.historical_candidates.read_text()),
        json.loads(args.historical_review.read_text()))
    if manifest["historical_review_sha256"] != sha256(args.historical_review):
        raise ValueError("Holdout source lineage changed")
    learned = torch.load(args.policy_checkpoint, map_location="cpu", weights_only=True)
    if (learned["plan_sha256"] != sha256(args.plan) or
            learned["checkpoint_sha256"] != sha256(args.checkpoint) or
            learned["historical_review_sha256"] != sha256(args.historical_review) or
            learned["max_steps"] != 30 or learned["max_new_tokens"] != 64):
        raise ValueError("RL policy or actor lineage changed")
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    means, mean_sha, mean_count = mean_adapter(agent, tokenizer, historical,
                                               args.device)
    if mean_sha != learned["mean_adapter_sha256"]:
        raise ValueError("Public LoRA factor changed")
    train_latents = source_latents(agent, tokenizer, historical, args.device)
    if sorted(train_latents) != learned["source_games"]:
        raise ValueError("RL source universe changed")
    policy = AdapterPolicy(learned["source_basis"].shape[1] + 6)
    policy.load_state_dict(learned["policy_state"])
    policy.eval()
    predictions = []
    with torch.no_grad():
        for row in targets:
            actor_family = visible_family(row["initial_observation"])
            if actor_family != row["family"]:
                raise ValueError("Visible goal and holdout metadata disagree")
            latent = train_latents[row["source_game"]] if row["source_game"] else None
            x = features(actor_family, latent, learned["source_mean"],
                         learned["source_basis"], learned["source_scale"])
            logits = policy(x, has_source=row["source_game"] is not None)
            probabilities = logits.softmax(-1)
            chosen = ARMS[int(logits.argmax())]
            predictions.append({"game": row["target_game"],
                                "input_content_sha256": row["input_content_sha256"],
                                "chosen_arm": chosen,
                                "probabilities": {arm: float(probabilities[index])
                                                  for index, arm in enumerate(ARMS)}})
    prediction_hash = hashlib.sha256(json.dumps(
        predictions, sort_keys=True).encode()).hexdigest()
    report = {"protocol": "Frozen trajectory-conditioned LoRA contextual-bandit policy on newly reviewed ALFWorld valid_seen holdout; real environment won reward; greedy admissible-command Qwen actor; all arms same 30-step/64-token plan budget",
              "plan_sha256": sha256(args.plan),
              "actor_checkpoint_sha256": sha256(args.checkpoint),
              "policy_checkpoint_sha256": sha256(args.policy_checkpoint),
              "holdout_manifest_sha256": sha256(args.candidates),
              "holdout_review_sha256": sha256(args.review),
              "mean_adapter_sha256": mean_sha,
              "mean_source_count": mean_count,
              "max_steps": 30, "max_new_tokens": 64,
              "predictions_sha256": prediction_hash,
              "predictions": predictions, "games": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    selected = list(zip(targets, predictions, strict=True))[args.start_index:]
    if args.limit:
        selected = selected[:args.limit]
    for row, prediction in selected:
        game = row["target_game"]
        fields = tokenize_records(tokenizer, row["source_records"], args.device) \
            if row["source_game"] else {}
        entry = {"game": game, "game_sha256": row["target_game_sha256"],
                 "family": row["family"], "source_game": row["source_game"],
                 "input_content_sha256": row["input_content_sha256"],
                 "chosen_arm": prediction["chosen_arm"],
                 "available_arms": ["base", "mean"] +
                     (["source"] if row["source_game"] else [])}
        for arm in entry["available_arms"]:
            entry[arm] = run_episode(
                agent, tokenizer, args.data_root / game, fields,
                adapter=arm == "source",
                fixed_adapter=means if arm == "mean" else None,
                device=args.device, max_steps=30, max_new_tokens=64,
                constrain_actions=True)
            if entry[arm]["status"] != "complete":
                break
            if entry[arm]["initial_observation"] != row["initial_observation"]:
                entry[arm]["status"] = "binding_failure"
                entry[arm]["error"] = "Actor initial observation differs from review"
                break
        report["games"].append(entry)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        report["summary"] = summarize(report["games"])
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(args.output)
        print(json.dumps({"game": game, "chosen": entry["chosen_arm"],
                          "rewards": {arm: entry.get(arm, {}).get("reward")
                                      for arm in entry["available_arms"]},
                          "summary": report["summary"]}), flush=True)
        if any(entry.get(arm, {}).get("status") != "complete"
               for arm in entry["available_arms"]):
            raise RuntimeError("Holdout environment failure recorded")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_warmstart_20261005/seed42_v2.pt"))
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path(
        "ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json"))
    parser.add_argument("--historical-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_candidates_20261005.json"))
    parser.add_argument("--historical-review", type=Path, default=Path(
        "data/annotations/trajectory_hyperlora_alf_20261005_reviewed_v2.json"))
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_rl_20261005/valid_seen_candidates.json"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/trajectory_hyperlora_alf_rl_valid_seen_20261005_reviewed.json"))
    parser.add_argument("--data-root", type=Path, default=Path("ttcl/data/alfworld_delta"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.40)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
