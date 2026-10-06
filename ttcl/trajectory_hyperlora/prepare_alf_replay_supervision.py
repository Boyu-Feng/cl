"""Fresh action labels from ALFWorld cross-trial source plans that win.

For each of 600 reviewed train targets, use the sibling source's commands only
if independently replayed in that target with official ``won``. Otherwise use
the target train walkthrough. All selected plans are replayed again while
building and reviewing their own action-level input-content bindings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.experience_evolution.environment import ACTOR_SYSTEM, make_env
from ttcl.trajectory_hyperlora.alfworld_same_game_retry import digest, file_hash
from ttcl.trajectory_hyperlora.calibrate_alf_sibling_gate import checked_tasks
from ttcl.trajectory_hyperlora.train_alf_sibling_labels import label_content, utility


def build(args) -> dict:
    old_tasks = checked_tasks(args, expected_tasks=args.expected_tasks)
    replay = json.loads(args.replay_audit.read_text())
    if (replay["labels_sha256"] != file_hash(args.labels) or
            replay["label_review_sha256"] != file_hash(args.label_review) or
            replay["source_review_sha256"] != file_hash(args.source_review) or
            replay["summary"]["tasks"] != args.expected_tasks or
            len(replay["tasks"]) != args.expected_tasks):
        raise ValueError("Source replay reward lineage changed")
    reward_by_game = {row["target_game"]: row for row in replay["tasks"]}
    if len(reward_by_game) != args.expected_tasks:
        raise ValueError("Duplicate replay target")
    tasks = []
    for old in old_tasks:
        reward = reward_by_game.get(old["target_game"])
        first = old["labels"][0]
        if (reward is None or reward["target_game_sha256"] !=
                old["target_game_sha256"] or
                reward["source_episode_sha256"] !=
                    old["source_episode_sha256"] or
                reward["wrong_records_sha256"] !=
                    digest(old["wrong_records"])):
            raise ValueError("Source replay reward/target content changed")
        records = first["source_records"]
        if reward["own"]["won"]:
            commands = [step["action"] for step in records][
                :reward["own"]["valid_prefix"]]
            teacher_arm = "source_replay"
        else:
            commands = [label["target_action"] for label in old["labels"]]
            teacher_arm = "walkthrough"
        if not commands or len(commands) > 30:
            raise ValueError("Reward-selected plan exceeds actor budget")
        game = args.data_root / old["target_game"]
        if file_hash(game) != old["target_game_sha256"]:
            raise ValueError("Reward-selected target game changed")
        env = make_env(game)
        try:
            state = env.reset()
            initial = first["target_messages"][1]["content"].split(
                "\nAvailable commands:\n", 1)[0]
            if str(state["feedback"]) != initial:
                raise ValueError("Reward-selected target reset changed")
            messages = [{"role": "system", "content": ACTOR_SYSTEM}]
            labels = []
            for index, command in enumerate(commands):
                available = list(state["admissible_commands"])
                if command not in available:
                    raise ValueError("Reward-selected plan command inadmissible")
                messages.append({"role": "user", "content":
                    str(state["feedback"]) + "\nAvailable commands:\n" +
                    "\n".join(available)})
                content = {"target_game": old["target_game"],
                    "target_game_sha256": old["target_game_sha256"],
                    "source_episode_sha256": old["source_episode_sha256"],
                    "source_records": records,
                    "teacher_arm": teacher_arm,
                    "target_messages": list(messages),
                    "target_action": command}
                labels.append({**content,
                    "input_content_sha256": digest(content)})
                state, _, done = env.step(command)
                if done and index != len(commands) - 1:
                    raise ValueError("Reward-selected plan ended early")
                messages.append({"role": "assistant", "content": command})
            if not state["won"]:
                raise ValueError("Reward-selected plan failed target replay")
        finally:
            env.close()
        tasks.append({"target_game": old["target_game"],
            "target_game_sha256": old["target_game_sha256"],
            "source_episode_sha256": old["source_episode_sha256"],
            "teacher_arm": teacher_arm,
            "own_utility": None,
            "teacher_utility": utility(1, len(commands)),
            "first_changed_action": 0,
            "wrong_records": old["wrong_records"],
            "wrong_replay_won": bool(reward["wrong"]["won"]),
            "labels": labels})
    return {"protocol": "Train-only source-plan official-won selection, fallback target walkthrough; fresh action targets replayed and content-bound against reviewed sibling source",
        "source_scope": "sibling_expert", "split": "train_large",
        "teacher_mode": "replay_or_walkthrough",
        "sibling_candidates_sha256": file_hash(args.candidates),
        "sibling_source_review_sha256": file_hash(args.source_review),
        "first_attempts_sha256": None,
        "predecessor_labels_sha256": file_hash(args.labels),
        "source_replay_audit_sha256": file_hash(args.replay_audit),
        "tasks": tasks}


def prepare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    dataset = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, ensure_ascii=False,
                                      indent=2) + "\n")
    print(json.dumps({"tasks": len(dataset["tasks"]),
        "source_replay_tasks": sum(task["teacher_arm"] == "source_replay"
                                   for task in dataset["tasks"]),
        "actions": sum(len(task["labels"]) for task in dataset["tasks"])}),
        flush=True)


def review(args):
    if args.output_review.exists():
        raise FileExistsError(args.output_review)
    dataset = json.loads(args.output.read_text())
    if dataset != build(args):
        raise ValueError("Reward-selected action dataset changed")
    notes = []
    for task in dataset["tasks"]:
        for label in task["labels"]:
            if digest(label_content(label)) != label["input_content_sha256"]:
                raise ValueError("Reward-selected action content changed")
            notes.append({"target_game": task["target_game"],
                "input_content_sha256": label["input_content_sha256"],
                "source_episode_sha256": task["source_episode_sha256"],
                "approved": True,
                "review_basis": "Fresh source-plan won or target walkthrough; command admissible and complete target replay won; full action prompt/source content bound"})
    result = {"protocol": "Fresh reviewed source-replay reward selected ALFWorld train action targets",
        "labels_sha256": file_hash(args.output),
        "source_replay_audit_sha256": file_hash(args.replay_audit),
        "annotations": notes}
    args.output_review.parent.mkdir(parents=True, exist_ok=True)
    args.output_review.write_text(json.dumps(result, ensure_ascii=False,
                                              indent=2) + "\n")
    print(json.dumps({"approved_actions": len(notes)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "review"))
    parser.add_argument("--labels", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train600_labels_20261006.json"))
    parser.add_argument("--label-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train600_labels_reviewed_20261006.json"))
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train600_reviewed_20261006.json"))
    parser.add_argument("--replay-audit", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train600_cross_trial_replay_20261006.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--expected-tasks", type=int, default=600)
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train600_replay_selected_labels_20261006.json"))
    parser.add_argument("--output-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train600_replay_selected_labels_reviewed_20261006.json"))
    args = parser.parse_args()
    if args.expected_tasks < 12 or args.expected_tasks % 6:
        parser.error("Expected tasks must cover six families")
    (prepare if args.command == "prepare" else review)(args)


if __name__ == "__main__":
    main()
