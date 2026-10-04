"""Collect train-only, reviewed source-to-new-game paired ALFWorld rewards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_paired_dev import summarize
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.prepare_alf_reward_pairs import (
    validate_reward_review,
)
from ttcl.trajectory_hyperlora.relational_router_pilot import tokenize_records


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_reward_pairs_20261005_candidates.json"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/trajectory_hyperlora_alf_reward_pairs_20261005_reviewed.json"))
    parser.add_argument("--plan", type=Path, default=Path(
        "ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.40)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.max_steps < 1 or args.max_new_tokens < 1:
        parser.error("Need fresh output path and positive budgets")
    manifest = json.loads(args.candidates.read_text())
    review = json.loads(args.review.read_text())
    approved = validate_reward_review(manifest, review, args.data_root)
    if sha256(args.plan) != manifest["plan_sha256"]:
        raise ValueError("Frozen plan content changed")
    plan = json.loads(args.plan.read_text())
    train_games = {game for sequence in plan["training"]
                   for game in sequence["games"]}
    evaluation_games = {game for sequence in plan["evaluation"]
                        for game in sequence["games"]}
    for row in approved:
        if (row["target_game"] not in train_games or
                row["source_game"] not in train_games or
                row["target_game"] in evaluation_games):
            raise ValueError("New reward game violates frozen train/test plan")
    if args.limit:
        approved = approved[:args.limit]
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    report = {"protocol": "Frozen ALFWorld train-only true-reward pairs; reviewed task/source binding; same reset, greedy actor, admissible-command constrained decoding; no dev/test rewards for training",
              "split": "train", "constrain_actions": True,
              "adapter_scale": 1.0,
              "checkpoint_sha256": sha256(args.checkpoint),
              "candidate_manifest_sha256": sha256(args.candidates),
              "reviewed_annotations_sha256": sha256(args.review),
              "plan_sha256": sha256(args.plan),
              "max_steps": args.max_steps,
              "max_new_tokens": args.max_new_tokens,
              "games": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for row in approved:
        fields = tokenize_records(tokenizer, row["source_records"], args.device)
        pair = {"game": row["target_game"],
                "game_sha256": row["target_game_sha256"],
                "source_game": row["source_game"],
                "source_episode_sha256": row["source_episode_sha256"],
                "input_content_sha256": row["input_content_sha256"]}
        for arm, enabled in (("base", False), ("generated", True)):
            pair[arm] = run_episode(
                agent, tokenizer, args.data_root / row["target_game"], fields,
                adapter=enabled, device=args.device,
                max_steps=args.max_steps,
                max_new_tokens=args.max_new_tokens,
                constrain_actions=True)
        report["games"].append(pair)
        report["summary"] = summarize(report["games"])
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"game": row["target_game"],
                          "base": pair["base"].get("reward"),
                          "generated": pair["generated"].get("reward"),
                          "summary": report["summary"]}), flush=True)


if __name__ == "__main__":
    main()
