"""Audit existing ALFWorld train trajectories for future experience learning.

This creates candidate metadata, not new supervision labels or benchmark
results. Train/evaluation game separation is checked against the frozen plan.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.paired_utility import content_hash


def audit(root: Path) -> dict:
    plan = json.loads((root / "plan.json").read_text())
    train_games = {game for seq in plan["training"] for game in seq["games"]}
    eval_games = {game for seq in plan["evaluation"] for game in seq["games"]}
    if train_games & eval_games:
        raise ValueError("Frozen train/evaluation game sets overlap")
    rows = []
    counts = Counter()
    for source_path in sorted((root / "training" / "delta").glob(
            "batch_*/seq_*/task_*/episode.json")):
        stage = int(source_path.parent.name.split("_")[1])
        if stage >= 2:
            continue
        future_path = source_path.parent.parent / f"task_{stage + 1}" / "episode.json"
        if not future_path.exists():
            counts["missing_future"] += 1
            continue
        source = json.loads(source_path.read_text())
        future = json.loads(future_path.read_text())
        if source["game"] not in train_games or future["game"] not in train_games:
            raise ValueError("Episode game is outside the frozen train pool")
        if source["game"] == future["game"]:
            raise ValueError("Source and future task are the same game")
        public_source = {
            "initial_observation": source["initial_observation"],
            "trajectory": source["trajectory"],
            "reward": source["reward"],
            "termination": source["termination"],
        }
        source_content = json.dumps(public_source, sort_keys=True, ensure_ascii=False)
        future_content = future["initial_observation"]
        if not future_content:
            raise ValueError("Future task lacks an initial public observation")
        rows.append({
            "source_path": str(source_path.relative_to(root)),
            "future_path": str(future_path.relative_to(root)),
            "source_content_sha256": content_hash(source_content),
            "future_initial_observation_sha256": content_hash(future_content),
            "source_game": source["game"],
            "future_game": future["game"],
            "source_reward": source["reward"],
            "future_reward_historical": future["reward"],
            "source_steps": len(source["trajectory"]),
            "source_invalid_commands": sum(
                step.get("valid_command") is False for step in source["trajectory"]),
            "future_seed_historical": future["seed"],
            "reviewed_target": False,
        })
        counts[f"stage_{stage}"] += 1
        counts["source_success" if source["reward"] > 0 else "source_failure"] += 1
        counts["future_success_historical" if future["reward"] > 0
               else "future_failure_historical"] += 1
    return {
        "plan_path": str(root / "plan.json"),
        "plan_sha256": content_hash((root / "plan.json").read_text()),
        "train_games": len(train_games),
        "evaluation_games": len(eval_games),
        "train_evaluation_overlap": 0,
        "counts": dict(counts),
        "pairs": rows,
        "warning": "Historical future rewards are descriptive only. No reviewed target or paired LoRA/no-LoRA rollout exists.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(
        "ttcl/results/experience_evolution/alfworld_delta_20260922"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_train_pair_audit.json"))
    args = parser.parse_args()
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"counts": result["counts"],
                      "train_games": result["train_games"],
                      "evaluation_games": result["evaluation_games"],
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
