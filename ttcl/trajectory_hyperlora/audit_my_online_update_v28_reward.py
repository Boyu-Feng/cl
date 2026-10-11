"""Replay every sampled action and verify reward-training input bindings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.experience_evolution.environment import make_env
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash


def replay(game: Path, episode: dict) -> None:
    env = make_env(game)
    try:
        state = env.reset()
        if str(state["feedback"]) != episode["initial_observation"]:
            raise ValueError("Initial observation changed")
        done = False
        for step in episode["trajectory"]:
            command = step["command"]
            if (done or command not in state["admissible_commands"] or
                    not step["valid"]):
                raise ValueError("Invalid or extra sampled action")
            state, _, done = env.step(command)
            if (str(state["feedback"]) != step["observation"] or
                    bool(state["won"]) != step["won"]):
                raise ValueError("Environment feedback changed")
        if (float(bool(state["won"])) != episode["reward"] or
                episode["termination"] != ("success" if state["won"]
                    else "budget_or_environment_done") or
                episode["steps"] != len(episode["trajectory"]) or
                episode["steps"] > 50 or episode["invalid_commands"] != 0):
            raise ValueError("Official reward or budget changed")
    finally:
        env.close()


def audit(args):
    review = json.loads(args.review.read_text())
    report = json.loads(args.report.read_text())
    if (report["review_sha256"] != file_hash(args.review) or
            report["pretrained_sha256"] != review["pretrained_sha256"] or
            report["checkpoint_sha256"] != review["checkpoint_sha256"] or
            report["actor_sha256"] != review["actor_sha256"] or
            report["action_guard_sha256"] != review["action_guard_sha256"] or
            review["runner_sha256"] != file_hash(args.runner) or
            review["actor_sha256"] != file_hash(args.actor) or
            review["action_guard_sha256"] != file_hash(args.guard) or
            review["model_code_sha256"] != file_hash(args.model_code) or
            review["checkpoint_sha256"] != file_hash(args.checkpoint) or
            review["pretrained_sha256"] != file_hash(args.pretrained) or
            review["pairs_sha256"] != digest(review["pairs"]) or
            len(report["train"]) > len(review["pairs"]) or
            report["failures"]):
        raise ValueError("Changed reward training lineage or failure record")
    if not args.partial and len(report["train"]) != len(review["pairs"]):
        raise ValueError("Incomplete reward training")
    for row, binding in zip(report["train"], review["pairs"]):
        content = {key: val for key, val in binding.items()
                   if key not in ("input_content_sha256", "reviewed_target")}
        if (not binding["reviewed_target"] or
                binding["input_content_sha256"] != digest(content) or
                binding["fixed_sha256"] != digest({key: review[key]
                    for key in review if key not in
                    ("protocol", "pairs", "pairs_sha256")}) or
                row["index"] != binding["index"] or
                row["source_id"] != binding["source_id"] or
                row["target_id"] != binding["target_id"] or
                row["input_content_sha256"] != binding["input_content_sha256"] or
                binding["target_game_sha256"] != file_hash(args.data_root /
                    binding["target_game"])):
            raise ValueError("Changed source-target binding")
        replay(args.data_root / binding["target_game"], row["episode"])
    if not args.partial:
        if (report["trained_model_sha256"] != file_hash(args.trained_model)
                or report["summary"]["rollouts"] != len(report["train"])
                or report["summary"]["failures"] != 0):
            raise ValueError("Changed trained artifact or final summary")
    return {"audit": "passed", "rollouts": len(report["train"]),
        "wins": sum(row["sampled_reward"] for row in report["train"]),
        "report_sha256": file_hash(args.report),
        "review_sha256": file_hash(args.review)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/my_online_update_v28_reward_train_20261009.json"))
    parser.add_argument("--report", type=Path, default=Path(
        "results/trajectory_hyperlora/my_online_update_v28_reward_train_20261009.json"))
    parser.add_argument("--trained-model", type=Path, default=Path(
        "results/trajectory_hyperlora/my_online_update_v28_reward_train_20261009.pt"))
    parser.add_argument("--runner", type=Path, default=Path(
        "ttcl/trajectory_hyperlora/train_my_online_update_v28_reward.py"))
    parser.add_argument("--actor", type=Path, default=Path(
        "ttcl/trajectory_hyperlora/alfworld_actor_generic_v28.py"))
    parser.add_argument("--guard", type=Path, default=Path(
        "ttcl/trajectory_hyperlora/action_feedback_guard_v28.py"))
    parser.add_argument("--model-code", type=Path, default=Path(
        "ttcl/trajectory_hyperlora/my_online_update_v28_model.py"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--pretrained", type=Path, default=Path(
        "results/trajectory_hyperlora/my_online_update_v28_utility_train_20261009.pt"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result), flush=True)
