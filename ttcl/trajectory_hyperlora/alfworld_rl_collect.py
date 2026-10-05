"""Collect real ALFWorld rewards for a train-only LoRA contextual bandit.

The actor and candidate adapters stay frozen. This collector records the
counterfactual reward of each adapter action under identical game budgets.
Training a policy from the rewards is implemented separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_frozen_eval_probe import mean_adapter
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.prepare_alf_next_task import (
    family, partition, validate_review,
)
from ttcl.trajectory_hyperlora.prepare_alf_reward_pairs import validate_reward_review
from ttcl.trajectory_hyperlora.relational_router_pilot import tokenize_records


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_train_targets(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    plan = json.loads(args.plan.read_text())
    train = {game for sequence in plan["training"] for game in sequence["games"]}
    evaluation = {game for sequence in plan["evaluation"] for game in sequence["games"]}
    if train & evaluation or plan["max_steps"] != 30 or plan["actor_max_tokens"] != 64:
        raise ValueError("Frozen plan or budgets changed")
    historical = validate_review(
        json.loads(args.historical_candidates.read_text()),
        json.loads(args.historical_review.read_text()))
    reward = validate_reward_review(
        json.loads(args.reward_candidates.read_text()),
        json.loads(args.reward_review.read_text()), args.data_root)
    if json.loads(args.reward_candidates.read_text())["plan_sha256"] != sha256(args.plan):
        raise ValueError("Reward manifest refers to another plan")
    targets = {}
    for row in [r for r in historical if r["split"] == "train"] + reward:
        game = row["target_game"]
        if game not in train or game in evaluation or row["source_game"] not in train or \
                partition(row["source_game"]) != "train" or \
                row["source_game"] == game or family(game) != row["family"] or \
                family(row["source_game"]) != row["family"]:
            raise ValueError("RL train source/target isolation failed")
        if sha256(args.data_root / game) != row["target_game_sha256"]:
            raise ValueError("RL target game content changed")
        previous = targets.get(game)
        if previous and (previous["source_game"] != row["source_game"] or
                         previous["source_episode_sha256"] != row["source_episode_sha256"]):
            raise ValueError("Conflicting reviewed target binding")
        targets[game] = row
    if len(targets) != 51:
        raise ValueError(f"Expected 51 reviewed training games, found {len(targets)}")
    return [targets[key] for key in sorted(targets)], historical


def run(args: argparse.Namespace) -> dict:
    if (args.limit < 0 or args.start_index < 0 or
            args.gpu_fraction <= 0 or args.gpu_fraction > 1):
        raise ValueError("Invalid collection limit or GPU memory fraction")
    targets, historical = load_train_targets(args)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    means, mean_sha, count = mean_adapter(agent, tokenizer, historical, args.device)
    bindings = {row["target_game"]: row["input_content_sha256"] for row in targets}
    metadata = {"protocol": "Real ALFWorld train-only full-information contextual-bandit rewards; greedy admissible-command actor; frozen plan 30 steps/64 tokens; actor and LoRA generator frozen",
                "plan_sha256": sha256(args.plan),
                "checkpoint_sha256": sha256(args.checkpoint),
                "historical_review_sha256": sha256(args.historical_review),
                "reward_review_sha256": sha256(args.reward_review),
                "mean_adapter_sha256": mean_sha,
                "mean_source_count": count,
                "max_steps": 30, "max_new_tokens": 64,
                "target_bindings": bindings}
    if args.output.exists():
        report = json.loads(args.output.read_text())
        if {key: report.get(key) for key in metadata} != metadata:
            raise ValueError("Existing collection metadata does not match")
    else:
        report = {**metadata, "games": []}
    done = {row["game"] for row in report["games"]}
    if len(done) != len(report["games"]) or not done <= bindings.keys():
        raise ValueError("Existing collection has duplicate or unknown target")
    selected = [row for row in targets[args.start_index:]
                if row["target_game"] not in done]
    if args.limit:
        selected = selected[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for row in selected:
        game = row["target_game"]
        fields = tokenize_records(tokenizer, row["source_records"], args.device)
        entry = {"game": game, "game_sha256": row["target_game_sha256"],
                 "family": row["family"], "source_game": row["source_game"],
                 "source_episode_sha256": row["source_episode_sha256"],
                 "input_content_sha256": row["input_content_sha256"]}
        for arm in ("base", "mean", "source"):
            entry[arm] = run_episode(
                agent, tokenizer, args.data_root / game, fields,
                adapter=arm == "source",
                fixed_adapter=means if arm == "mean" else None,
                device=args.device, max_steps=30, max_new_tokens=64,
                constrain_actions=True)
            if entry[arm]["status"] != "complete":
                break
        report["games"].append(entry)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(args.output)
        print(json.dumps({"game": game,
                          "rewards": {arm: entry[arm].get("reward")
                                      if arm in entry else None
                                      for arm in ("base", "mean", "source")},
                          "completed": len(report["games"])}), flush=True)
        if any(entry.get(arm, {}).get("status") != "complete"
               for arm in ("base", "mean", "source")):
            raise RuntimeError("Environment failure recorded; fix before proceeding")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_warmstart_20261005/seed42_v2.pt"))
    parser.add_argument("--plan", type=Path, default=Path(
        "ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json"))
    parser.add_argument("--historical-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_candidates_20261005.json"))
    parser.add_argument("--historical-review", type=Path, default=Path(
        "data/annotations/trajectory_hyperlora_alf_20261005_reviewed_v2.json"))
    parser.add_argument("--reward-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_reward_pairs_20261005_candidates.json"))
    parser.add_argument("--reward-review", type=Path, default=Path(
        "data/annotations/trajectory_hyperlora_alf_reward_pairs_20261005_reviewed.json"))
    parser.add_argument("--data-root", type=Path, default=Path("ttcl/data/alfworld_delta"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.40)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
