"""Fresh reviewed action labels for cross-game ALFWorld sibling transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.experience_evolution.environment import ACTOR_SYSTEM, make_env
from ttcl.trajectory_hyperlora.alfworld_same_game_retry import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_sibling_transfer import checked_review


def label_content(row):
    return {key: row[key] for key in ("target_game", "target_game_sha256",
        "source_episode_sha256", "source_records", "teacher_arm",
        "target_messages", "target_action")}


def utility(reward, steps):
    return float(reward) * (1.0 - .25 * (steps - 1) / 29.0)


def build(args):
    if args.split not in ("train", "train_large"):
        raise ValueError("Sibling expert labels are train-only")
    pairs = checked_review(args)
    first = ({item["game"]: item["base"] for item in
             json.loads(args.first_attempts.read_text())["games"]}
             if args.split == "train" else {})
    tasks = []
    for row, note in pairs:
        game = args.data_root / row["target_game"]
        commands = json.loads(game.read_text())["walkthrough"]
        if not commands or len(commands) > 30:
            raise ValueError("Train target expert exceeds budget")
        env = make_env(game)
        try:
            state = env.reset()
            messages = [{"role": "system", "content": ACTOR_SYSTEM}]
            labels = []
            for index, command in enumerate(commands):
                available = list(state["admissible_commands"])
                if command not in available:
                    raise ValueError("Train target expert command inadmissible")
                messages.append({"role": "user", "content":
                    str(state["feedback"]) + "\nAvailable commands:\n" +
                    "\n".join(available)})
                content = {"target_game": row["target_game"],
                    "target_game_sha256": row["target_game_sha256"],
                    "source_episode_sha256": note["source_records_sha256"],
                    "source_records": note["source_records"],
                    "teacher_arm": "walkthrough",
                    "target_messages": list(messages),
                    "target_action": command}
                labels.append({**content,
                    "input_content_sha256": digest(content)})
                state, _, done = env.step(command)
                if done and index != len(commands) - 1:
                    raise ValueError("Train target expert ended early")
                messages.append({"role": "assistant", "content": command})
            if not state["won"]:
                raise ValueError("Train target expert did not win")
        finally:
            env.close()
        first_episode = first.get(row["target_game"])
        divergence = (next((index for index, (expert, source_step)
            in enumerate(zip(commands, first_episode["trajectory"]))
            if expert != source_step["command"]), None)
            if first_episode is not None else None)
        tasks.append({"target_game": row["target_game"],
            "target_game_sha256": row["target_game_sha256"],
            "source_episode_sha256": note["source_records_sha256"],
            "teacher_arm": "walkthrough",
            "own_utility": (utility(first_episode["reward"],
                                    first_episode["steps"])
                            if first_episode is not None else None),
            "teacher_utility": utility(1, len(commands)),
            "first_changed_action": divergence if divergence is not None else 0,
            "wrong_records": note["wrong_records"],
            "labels": labels})
    expected = 42 if args.split == "train" else 6 * args.per_family
    if len(tasks) != expected:
        raise ValueError("Expanded sibling train targets incomplete")
    return tasks


def prepare(args):
    if args.labels.exists():
        raise FileExistsError(args.labels)
    tasks = build(args)
    dataset = {"protocol": "Train-only cross-game expert actions with freshly reviewed successful sibling expert source and same-family unrelated expert control; target expert replayed against new source-content bindings",
        "source_scope": "sibling_expert",
        "split": args.split,
        "teacher_mode": "walkthrough",
        "sibling_candidates_sha256": file_hash(args.candidates),
        "sibling_source_review_sha256": file_hash(args.source_review),
        "first_attempts_sha256": (file_hash(args.first_attempts)
            if args.split == "train" else None),
        "tasks": tasks}
    args.labels.parent.mkdir(parents=True, exist_ok=True)
    args.labels.write_text(json.dumps(dataset, ensure_ascii=False,
                                      indent=2) + "\n")
    print(json.dumps({"tasks": len(tasks),
        "actions": sum(len(task["labels"]) for task in tasks)}), flush=True)


def review(args):
    if args.label_review.exists():
        raise FileExistsError(args.label_review)
    dataset = json.loads(args.labels.read_text())
    if (dataset["source_scope"] != "sibling_expert" or
            dataset["sibling_candidates_sha256"] !=
            file_hash(args.candidates) or
            dataset["sibling_source_review_sha256"] !=
            file_hash(args.source_review) or
            dataset["first_attempts_sha256"] !=
            (file_hash(args.first_attempts) if args.split == "train" else None) or
            dataset.get("split", "train") != args.split or
            dataset["tasks"] != build(args)):
        raise ValueError("Sibling action target review or contents changed")
    notes = []
    for task in dataset["tasks"]:
        for label in task["labels"]:
            if digest(label_content(label)) != label["input_content_sha256"]:
                raise ValueError("Sibling action input-content binding changed")
            notes.append({"target_game": task["target_game"],
                "input_content_sha256": label["input_content_sha256"],
                "source_episode_sha256": task["source_episode_sha256"],
                "approved": True,
                "review_basis": "New sibling expert source bound to train target expert action; source and target separately replayed with terminal won"})
    result = {"protocol": "Fresh action-level reviewed bindings for cross-game ALFWorld expert distillation",
        "labels_sha256": file_hash(args.labels),
        "annotations": notes}
    args.label_review.parent.mkdir(parents=True, exist_ok=True)
    args.label_review.write_text(json.dumps(result, ensure_ascii=False,
                                      indent=2) + "\n")
    print(json.dumps({"approved_actions": len(notes)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "review"))
    parser.add_argument("--plan", type=Path, default=Path(
        "ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json"))
    parser.add_argument("--first-attempts", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_rl_20261005/train_rewards_complete.json"))
    parser.add_argument("--retry-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_same_game_retry_candidates_20261006.json"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/alf_same_game_retry_reviewed_20261006.json"))
    parser.add_argument("--all-source-review", type=Path, default=Path(
        "data/annotations/alf_expert_first_attempts_reviewed_20261006.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--split", choices=("train", "train_large"),
                        default="train")
    parser.add_argument("--per-family", type=int, default=20)
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train42_candidates_20261006.json"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train42_reviewed_20261006.json"))
    parser.add_argument("--labels", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train42_labels_20261006.json"))
    parser.add_argument("--label-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train42_labels_reviewed_20261006.json"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        review(args)


if __name__ == "__main__":
    main()
