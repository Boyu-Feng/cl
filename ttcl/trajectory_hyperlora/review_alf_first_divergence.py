"""Bind on-policy terminal preferences to the first shared-state action split.

Only train tasks with an exclusive own or control win are candidates. Review
replays both full trajectories and checks the two divergent commands were
admissible in the same state. This is a credit-assignment heuristic, not a
proof that the first differing action caused the terminal outcome.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.experience_evolution.environment import ACTOR_SYSTEM, make_env
from ttcl.trajectory_hyperlora.alfworld_same_game_retry import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_sibling_transfer import checked_review


def candidate_content(row: dict) -> dict:
    return {key: value for key, value in row.items()
            if key != "input_content_sha256"}


def candidates(args):
    report = json.loads(args.rollouts.read_text())
    if (report["candidates_sha256"] != file_hash(args.source_candidates) or
            report["source_review_sha256"] != file_hash(args.source_review) or
            report["summary"]["n"] != args.expected_games or
            len(report["games"]) != args.expected_games or
            report["failures"] or
            "train_large" not in report["protocol"]):
        raise ValueError("First-divergence reward source changed")
    reviewed = checked_review(argparse.Namespace(
        candidates=args.source_candidates,
        source_review=args.source_review,
        retry_candidates=args.retry_candidates,
        review=args.retry_review,
        all_source_review=args.all_source_review,
        data_root=args.data_root, split="train_large"))
    by_game = {row["target_game"]: (row, note) for row, note in reviewed}
    rows = []
    for game in report["games"]:
        source_row, note = by_game[game["game"]]
        if (game["input_content_sha256"] !=
                source_row["input_content_sha256"] or
                len({arm["initial_observation"] for arm
                     in game["arms"].values()}) != 1 or
                len({arm["initial_commands_sha256"] for arm
                     in game["arms"].values()}) != 1):
            raise ValueError("Source/target reset binding changed")
        arms = game["arms"]
        own_only = (arms["own"]["reward"] == 1. and
                    arms["base"]["reward"] == 0. and
                    arms["wrong"]["reward"] == 0.)
        wrong_only = (arms["wrong"]["reward"] == 1. and
                      arms["base"]["reward"] == 0. and
                      arms["own"]["reward"] == 0.)
        if not (own_only or wrong_only):
            continue
        own = arms["own"]["trajectory"]
        wrong = arms["wrong"]["trajectory"]
        index = next((i for i, (left, right) in enumerate(zip(own, wrong))
                      if left["command"] != right["command"]), None)
        if index is None or any(
                own[i]["observation"] != wrong[i]["observation"]
                for i in range(index)):
            raise ValueError("Preference lacks a shared-state action split")
        before = (arms["own"]["initial_observation"] if index == 0 else
                  own[index - 1]["observation"])
        row = {"target_game": game["game"],
            "target_game_sha256": source_row["target_game_sha256"],
            "source_input_content_sha256": game["input_content_sha256"],
            "source_records_sha256": note["source_records_sha256"],
            "wrong_records_sha256": note["wrong_records_sha256"],
            "first_divergence": index,
            "prefix_commands": [step["command"] for step in own[:index]],
            "before_observation_sha256": digest(before),
            "preferred_arm": "own" if own_only else "wrong",
            "preferred_action": (own if own_only else wrong)[index]["command"],
            "rejected_action": (wrong if own_only else own)[index]["command"],
            "own_trajectory_sha256": digest(own),
            "wrong_trajectory_sha256": digest(wrong),
            "base_trajectory_sha256": digest(arms["base"]["trajectory"])}
        row["input_content_sha256"] = digest(candidate_content(row))
        rows.append(row)
    if not rows or len({row["target_game"] for row in rows}) != len(rows):
        raise ValueError("No unique reviewed terminal preference candidates")
    return rows, report, by_game


def prepare(args):
    if args.candidates.exists():
        raise FileExistsError(args.candidates)
    rows, report, _ = candidates(args)
    result = {"protocol": "Train-only terminal exclusive wins; first common-state own/wrong action divergence; content bound, pending replay review",
        "rollouts_sha256": file_hash(args.rollouts),
        "source_candidates_sha256": file_hash(args.source_candidates),
        "source_review_sha256": file_hash(args.source_review),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "rows": rows}
    args.candidates.parent.mkdir(parents=True, exist_ok=True)
    args.candidates.write_text(json.dumps(result, ensure_ascii=False,
                                          indent=2) + "\n")
    print(json.dumps({"candidates": len(rows),
        "own_preferred": sum(row["preferred_arm"] == "own" for row in rows)}),
        flush=True)


def replay(game: Path, episode: dict):
    env = make_env(game)
    try:
        state = env.reset()
        if str(state["feedback"]) != episode["initial_observation"]:
            raise ValueError("Preference replay reset changed")
        for index, step in enumerate(episode["trajectory"]):
            if step["command"] not in state["admissible_commands"]:
                raise ValueError("Preference replay command became inadmissible")
            state, _, done = env.step(step["command"])
            if (str(state["feedback"]) != step["observation"] or
                    bool(state["won"]) != bool(step["won"]) or
                    done and index != len(episode["trajectory"]) - 1):
                raise ValueError("Preference replay transition changed")
        if bool(state["won"]) != bool(episode["reward"]):
            raise ValueError("Preference replay terminal changed")
    finally:
        env.close()


def review(args):
    if args.selection_review.exists():
        raise FileExistsError(args.selection_review)
    selected = json.loads(args.candidates.read_text())
    rows, report, by_game = candidates(args)
    if (selected["rollouts_sha256"] != file_hash(args.rollouts) or
            selected["source_candidates_sha256"] !=
                file_hash(args.source_candidates) or
            selected["source_review_sha256"] != file_hash(args.source_review) or
            selected["checkpoint_sha256"] != report["checkpoint_sha256"] or
            selected["rows"] != rows):
        raise ValueError("First-divergence candidates changed before review")
    episodes = {row["game"]: row for row in report["games"]}
    notes = []
    for row in rows:
        source, note = by_game[row["target_game"]]
        game = args.data_root / row["target_game"]
        if (file_hash(game) != source["target_game_sha256"] or
                digest(candidate_content(row)) != row["input_content_sha256"] or
                digest(note["source_records"]) != row["source_records_sha256"] or
                digest(note["wrong_records"]) != row["wrong_records_sha256"]):
            raise ValueError("Preference target or source content changed")
        arms = episodes[row["target_game"]]["arms"]
        for arm in ("base", "own", "wrong"):
            replay(game, arms[arm])
        env = make_env(game)
        try:
            state = env.reset()
            messages = [{"role": "system", "content": ACTOR_SYSTEM}]
            for command in row["prefix_commands"]:
                available = list(state["admissible_commands"])
                messages.append({"role": "user", "content":
                    str(state["feedback"]) + "\nAvailable commands:\n" +
                    "\n".join(available)})
                if command not in available:
                    raise ValueError("Preference prefix command inadmissible")
                state, _, _ = env.step(command)
                messages.append({"role": "assistant", "content": command})
            available = list(state["admissible_commands"])
            if (digest(str(state["feedback"])) !=
                    row["before_observation_sha256"] or
                    row["preferred_action"] not in available or
                    row["rejected_action"] not in available):
                raise ValueError("Preference candidates not in shared state")
            messages.append({"role": "user", "content":
                str(state["feedback"]) + "\nAvailable commands:\n" +
                "\n".join(available)})
            reviewed = {"target_game": row["target_game"],
                "selection_input_content_sha256": row["input_content_sha256"],
                "source_records_sha256": row["source_records_sha256"],
                "wrong_records_sha256": row["wrong_records_sha256"],
                "target_messages": messages,
                "preferred_action": row["preferred_action"],
                "rejected_action": row["rejected_action"],
                "preferred_arm": row["preferred_arm"]}
            reviewed["input_content_sha256"] = digest(reviewed)
            notes.append({**reviewed, "approved": True,
                "review_basis": "All three terminal trajectories replayed; common prefix replayed; both differing commands admissible in identical public state"})
        finally:
            env.close()
    result = {"protocol": "Fresh train-only first-divergence preference labels bound to complete terminal rollouts",
        "candidates_sha256": file_hash(args.candidates),
        "annotations": notes}
    args.selection_review.parent.mkdir(parents=True, exist_ok=True)
    args.selection_review.write_text(json.dumps(result, ensure_ascii=False,
                                                indent=2) + "\n")
    print(json.dumps({"approved_preferences": len(notes)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "review"))
    parser.add_argument("--rollouts", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train60_taskpair_current1000_20261006.json"))
    parser.add_argument("--expected-games", type=int, default=60)
    parser.add_argument("--source-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train120_candidates_20261006.json"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train120_reviewed_20261006.json"))
    parser.add_argument("--retry-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_same_game_retry_candidates_20261006.json"))
    parser.add_argument("--retry-review", type=Path, default=Path(
        "data/annotations/alf_same_game_retry_reviewed_20261006.json"))
    parser.add_argument("--all-source-review", type=Path, default=Path(
        "data/annotations/alf_expert_first_attempts_reviewed_20261006.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_first_divergence_train60_candidates_20261006.json"))
    parser.add_argument("--selection-review", type=Path, default=Path(
        "data/annotations/alf_first_divergence_train60_reviewed_20261006.json"))
    args = parser.parse_args()
    if args.expected_games < 1:
        parser.error("Expected game count must be positive")
    (prepare if args.command == "prepare" else review)(args)


if __name__ == "__main__":
    main()
