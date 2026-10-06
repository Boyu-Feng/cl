"""Review reward-selected ALFWorld actions from the frozen policy's own rollouts.

Only completed train-domain rollouts with official wins provide targets. For
each task, prefer the shortest successful own-source trajectory, then wrong
source, then no-LoRA; this selects by observed reward, not future test labels.
The selected route is replayed to bind every action prompt to the target game.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.experience_evolution.environment import ACTOR_SYSTEM, make_env
from ttcl.trajectory_hyperlora.alfworld_heldout_retry import replay_episode
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import records_from_episode
from ttcl.trajectory_hyperlora.alfworld_same_game_retry import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_sibling_transfer import checked_review
from ttcl.trajectory_hyperlora.train_alf_sibling_labels import utility
from ttcl.trajectory_hyperlora.train_alfworld_retry_reward import label_content


def selected_arm(arms: dict) -> str | None:
    choices = [arm for arm in ("own", "wrong", "base")
               if arms[arm]["status"] == "complete" and
               arms[arm]["reward"] == 1.0]
    return max(choices, key=lambda arm: (utility(1, arms[arm]["steps"]),
                                       -("own", "wrong", "base").index(arm))) \
        if choices else None


def checked_rollouts(args):
    report = json.loads(args.rollouts.read_text())
    if (report["candidates_sha256"] != file_hash(args.candidates) or
            report["source_review_sha256"] != file_hash(args.source_review) or
            report["checkpoint_sha256"] != file_hash(args.checkpoint) or
            report["summary"]["n"] != args.expected_games or
            len(report["games"]) != args.expected_games or
            report["failures"] or
            "train_large" not in report["protocol"] or
            report.get("per_family_limit") != args.per_family_limit):
        raise ValueError("On-policy reward rollout lineage/budget changed")
    source_rows = checked_review(argparse.Namespace(
        candidates=args.candidates, source_review=args.source_review,
        retry_candidates=args.retry_candidates, review=args.retry_review,
        all_source_review=args.all_source_review, data_root=args.data_root,
        split="train_large"))
    by_game = {row["target_game"]: (row, note) for row, note in source_rows}
    if len({game["game"] for game in report["games"]}) != args.expected_games:
        raise ValueError("Duplicate on-policy target")
    for game in report["games"]:
        pair = by_game.get(game["game"])
        if (pair is None or game["input_content_sha256"] !=
                pair[0]["input_content_sha256"] or
                any(game["arms"][arm]["status"] != "complete" or
                    game["arms"][arm]["steps"] > 30 or
                    game["arms"][arm]["invalid_commands"] != 0
                    for arm in ("base", "own", "wrong"))):
            raise ValueError("Incomplete or changed on-policy target")
    return report, by_game


def build(args) -> dict:
    report, by_game = checked_rollouts(args)
    source_mode = getattr(args, "source_mode", "sibling_expert")
    tasks = []
    for item in report["games"]:
        row, note = by_game[item["game"]]
        arm = selected_arm(item["arms"])
        if arm is None:
            continue
        episode = item["arms"][arm]
        game = args.data_root / item["game"]
        if file_hash(game) != row["target_game_sha256"]:
            raise ValueError("On-policy target game changed")
        base = item["arms"]["base"]
        if source_mode == "base_episode":
            if base["reward"]:
                continue
            replay_episode(game, base)
            source_records = records_from_episode(base)
            source_episode_sha256 = digest(source_records)
        else:
            source_records = note["source_records"]
            source_episode_sha256 = note["source_records_sha256"]
        env = make_env(game)
        try:
            state = env.reset()
            if str(state["feedback"]) != episode["initial_observation"]:
                raise ValueError("On-policy reset changed")
            messages = [{"role": "system", "content": ACTOR_SYSTEM}]
            labels = []
            for index, step in enumerate(episode["trajectory"]):
                available = list(state["admissible_commands"])
                command = step["command"]
                if (not step["valid"] or command not in available or
                        step["response"] != command):
                    raise ValueError("On-policy command or response changed")
                messages.append({"role": "user", "content":
                    str(state["feedback"]) + "\nAvailable commands:\n" +
                    "\n".join(available)})
                content = {"target_game": item["game"],
                    "target_game_sha256": row["target_game_sha256"],
                    "source_episode_sha256": source_episode_sha256,
                    "source_records": source_records,
                    "teacher_arm": arm,
                    "target_messages": list(messages),
                    "target_action": command}
                labels.append({**content, "input_content_sha256":
                    digest(content)})
                state, _, done = env.step(command)
                if (str(state["feedback"]) != step["observation"] or
                        bool(state["won"]) != bool(step["won"]) or
                        done and index != len(episode["trajectory"]) - 1):
                    raise ValueError("On-policy winning transition changed")
                messages.append({"role": "assistant", "content": command})
            if not state["won"] or len(labels) != episode["steps"]:
                raise ValueError("On-policy teacher did not win")
        finally:
            env.close()
        own = item["arms"]["own"]
        tasks.append({"target_game": item["game"],
            "target_game_sha256": row["target_game_sha256"],
            "source_episode_sha256": source_episode_sha256,
            "teacher_arm": arm,
            "own_utility": utility(own["reward"], own["steps"]),
            "baseline_utility": utility(base["reward"], base["steps"]),
            "teacher_utility": utility(1, episode["steps"]),
            "first_changed_action": 0,
            "wrong_records": note["wrong_records"],
            "labels": labels})
    if len(tasks) < args.minimum_winners:
        raise ValueError("Too few official on-policy wins for training")
    return {"protocol": ("Train-only official-won on-policy best-arm reward selection; every selected action replayed and freshly content-bound"
            if source_mode == "sibling_expert" else
            "Train-only official-won retry reward selection from replayed failed no-LoRA trajectories; every selected action freshly content-bound"),
        "source_scope": ("own_failed_retry" if source_mode ==
                         "base_episode" else "sibling_expert"),
        **({"source_mode": source_mode} if source_mode ==
           "base_episode" else {}), "split": "train_large",
        "teacher_mode": "on_policy_best",
        "sibling_candidates_sha256": file_hash(args.candidates),
        "sibling_source_review_sha256": file_hash(args.source_review),
        "first_attempts_sha256": None,
        "on_policy_rollouts_sha256": file_hash(args.rollouts),
        "on_policy_checkpoint_sha256": file_hash(args.checkpoint),
        "on_policy_expected_games": args.expected_games,
        "on_policy_per_family_limit": args.per_family_limit,
        "tasks": tasks}


def prepare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    dataset = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"winning_tasks": len(dataset["tasks"]),
        "actions": sum(len(task["labels"]) for task in dataset["tasks"]),
        "teacher_arms": {arm: sum(task["teacher_arm"] == arm
            for task in dataset["tasks"]) for arm in ("own", "wrong", "base")}}),
        flush=True)


def review(args):
    if args.output_review.exists():
        raise FileExistsError(args.output_review)
    dataset = json.loads(args.output.read_text())
    if dataset != build(args):
        raise ValueError("On-policy selected labels changed")
    notes = []
    for task in dataset["tasks"]:
        for label in task["labels"]:
            if digest(label_content(label)) != label["input_content_sha256"]:
                raise ValueError("On-policy action content changed")
            notes.append({"target_game": task["target_game"],
                "input_content_sha256": label["input_content_sha256"],
                "source_episode_sha256": task["source_episode_sha256"],
                "approved": True,
                "review_basis": "Frozen policy official-won arm; exact target reset, command availability, per-step feedback and terminal win replayed; full prompt/source content bound"})
    result = {"protocol": "Fresh train-only on-policy reward-selected ALFWorld action review",
        "labels_sha256": file_hash(args.output),
        "on_policy_rollouts_sha256": file_hash(args.rollouts),
        "annotations": notes}
    args.output_review.parent.mkdir(parents=True, exist_ok=True)
    args.output_review.write_text(json.dumps(result, ensure_ascii=False,
                                               indent=2) + "\n")
    print(json.dumps({"approved_actions": len(notes)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "review"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_taskpair_current1000_20261006.pt"))
    parser.add_argument("--rollouts", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train240_taskpair_current1000_20261006.json"))
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train600_reviewed_20261006.json"))
    parser.add_argument("--retry-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_same_game_retry_candidates_20261006.json"))
    parser.add_argument("--retry-review", type=Path, default=Path(
        "data/annotations/alf_same_game_retry_reviewed_20261006.json"))
    parser.add_argument("--all-source-review", type=Path, default=Path(
        "data/annotations/alf_expert_first_attempts_reviewed_20261006.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--expected-games", type=int, default=240)
    parser.add_argument("--per-family-limit", type=int, default=40)
    parser.add_argument("--minimum-winners", type=int, default=20)
    parser.add_argument("--source-mode", choices=("sibling_expert",
                        "base_episode"), default="sibling_expert")
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train240_onpolicy_best_labels_20261006.json"))
    parser.add_argument("--output-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train240_onpolicy_best_reviewed_20261006.json"))
    args = parser.parse_args()
    if (args.expected_games < 1 or args.per_family_limit < 1 or
            args.minimum_winners < 1):
        parser.error("Invalid frozen on-policy reward selection")
    (prepare if args.command == "prepare" else review)(args)


if __name__ == "__main__":
    main()
