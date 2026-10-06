"""Replay reviewed sibling expert commands in a different train trial.

This checks whether a single target walkthrough wrongly treats a valid
alternative source plan as an incorrect action sequence. All target trials
are train-only and action labels remain separately reviewed/content-bound.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ttcl.experience_evolution.environment import make_env
from ttcl.trajectory_hyperlora.alfworld_same_game_retry import digest, file_hash
from ttcl.trajectory_hyperlora.calibrate_alf_sibling_gate import checked_tasks


def replay(game: Path, expected_initial: str, commands: list[str]) -> dict:
    env = make_env(game)
    try:
        state = env.reset()
        if str(state["feedback"]) != expected_initial:
            raise ValueError("Sibling replay target reset changed")
        for index, command in enumerate(commands):
            if command not in state["admissible_commands"]:
                return {"won": False, "valid_prefix": index,
                        "first_invalid_action": command,
                        "environment_done": False}
            state, _, done = env.step(command)
            if done or state["won"]:
                return {"won": bool(state["won"]),
                        "valid_prefix": index + 1,
                        "first_invalid_action": None,
                        "environment_done": bool(done)}
        return {"won": bool(state["won"]),
                "valid_prefix": len(commands),
                "first_invalid_action": None,
                "environment_done": False}
    finally:
        env.close()


def run(args) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    tasks = checked_tasks(args, expected_tasks=args.expected_tasks)
    rows = []
    for task in tasks:
        label = task["labels"][0]
        initial = label["target_messages"][1]["content"].split(
            "\nAvailable commands:\n", 1)[0]
        game = args.data_root / task["target_game"]
        if file_hash(game) != task["target_game_sha256"]:
            raise ValueError("Sibling replay target game bytes changed")
        own = [step["action"] for step in label["source_records"]]
        wrong = [step["action"] for step in task["wrong_records"]]
        expert = [step["target_action"] for step in task["labels"]]
        if not own or not wrong or not expert:
            raise ValueError("Sibling replay has empty plan")
        row = {"target_game": task["target_game"],
            "target_game_sha256": task["target_game_sha256"],
            "source_episode_sha256": task["source_episode_sha256"],
            "wrong_records_sha256": digest(task["wrong_records"]),
            "target_expert_first_action": expert[0],
            "own_first_action": own[0],
            "wrong_first_action": wrong[0],
            "own": replay(game, initial, own),
            "wrong": replay(game, initial, wrong)}
        rows.append(row)
    def summary(arm: str) -> dict:
        count = Counter()
        for row in rows:
            result = row[arm]
            count["won"] += result["won"]
            count["first_action_admissible"] += result["valid_prefix"] >= 1
            count["all_source_actions_admissible"] += (
                result["first_invalid_action"] is None)
            count["first_action_equals_target_walkthrough"] += (
                row[f"{arm}_first_action"] ==
                row["target_expert_first_action"])
            count["won_with_different_first_action"] += (
                result["won"] and row[f"{arm}_first_action"] !=
                row["target_expert_first_action"])
        return dict(count)
    output = {"protocol": "Read-only cross-trial source-command replay on 600 reviewed ALFWorld train targets; official environment won, distinct target walkthrough used only for agreement diagnostic; no policy training",
        "labels_sha256": file_hash(args.labels),
        "label_review_sha256": file_hash(args.label_review),
        "source_review_sha256": file_hash(args.source_review),
        "summary": {"tasks": len(rows), "own": summary("own"),
                    "wrong": summary("wrong")},
        "tasks": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False,
                                      indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train600_labels_20261006.json"))
    parser.add_argument("--label-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train600_labels_reviewed_20261006.json"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train600_reviewed_20261006.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--expected-tasks", type=int, default=600)
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train600_cross_trial_replay_20261006.json"))
    args = parser.parse_args()
    if args.expected_tasks < 12 or args.expected_tasks % 6:
        parser.error("Expected tasks must cover six families")
    print(json.dumps(run(args)["summary"]), flush=True)


if __name__ == "__main__":
    main()
