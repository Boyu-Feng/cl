"""Review more ALFWorld train first attempts against official walkthroughs.

This expands reward-confirmed expert distillation beyond the small retry-win
subset. The old model's first attempt is the LoRA source; only train-game
walkthroughs supply target actions. The retry development games stay excluded.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from ttcl.experience_evolution.environment import ACTOR_SYSTEM, make_env
from ttcl.trajectory_hyperlora.alfworld_heldout_retry import replay_episode
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import records_from_episode
from ttcl.trajectory_hyperlora.alfworld_same_game_retry import digest, file_hash


def row_content(row):
    return {key: value for key, value in row.items()
            if key != "input_content_sha256"}


def label_content(row):
    return {key: row[key] for key in ("target_game", "target_game_sha256",
        "source_episode_sha256", "source_records", "teacher_arm",
        "target_messages", "target_action")}


def prepare(args):
    if args.candidates.exists():
        raise FileExistsError(args.candidates)
    plan = json.loads(args.plan.read_text())
    first = json.loads(args.first_attempts.read_text())
    retry = json.loads(args.retry_candidates.read_text())
    if (first["plan_sha256"] != file_hash(args.plan) or
            retry["plan_sha256"] != file_hash(args.plan) or
            first["max_steps"] != 30 or first["max_new_tokens"] != 64):
        raise ValueError("First-attempt or split lineage changed")
    training = {game for sequence in plan["training"]
                for game in sequence["games"]}
    evaluation = {game for sequence in plan["evaluation"]
                  for game in sequence["games"]}
    held = {row["target_game"] for row in retry["rows"]
            if row["split"] == "dev"}
    if training & evaluation or not held <= training:
        raise ValueError("Train/development/test isolation changed")
    selected = []
    for item in first["games"]:
        game = item["game"]
        if game in held:
            continue
        episode = item["base"]
        if (game not in training or game in evaluation or
                episode["status"] != "complete" or
                episode["steps"] != len(episode["trajectory"]) or
                file_hash(args.data_root / game) != item["game_sha256"]):
            raise ValueError("Non-training or incomplete first attempt")
        selected.append({"target_game": game,
            "target_game_sha256": item["game_sha256"],
            "family": item["family"],
            "source_episode_sha256": digest(episode),
            "source_records": records_from_episode(episode),
            "source_reward": episode["reward"],
            "source_steps": episode["steps"]})
    groups = defaultdict(list)
    for row in selected:
        groups[row["family"]].append(row)
    for row in selected:
        others = [item for item in groups[row["family"]]
                  if item["target_game"] != row["target_game"]]
        if not others:
            raise ValueError("No same-family train wrong-history control")
        wrong = min(others, key=lambda item: digest(
            ["expert_all_wrong", row["target_game"], item["target_game"]]))
        row["wrong_source_game"] = wrong["target_game"]
        row["wrong_source_episode_sha256"] = wrong["source_episode_sha256"]
        row["wrong_records"] = wrong["source_records"]
        row["input_content_sha256"] = digest(row_content(row))
    selected.sort(key=lambda row: row["target_game"])
    if len(selected) != 42 or len({row["target_game"] for row in selected}) != 42:
        raise ValueError(f"Expected 42 distinct train sources, got {len(selected)}")
    result = {"protocol": "Forty-two frozen ALFWorld train first attempts, excluding nine retry development games and all official evaluation games; own and wrong histories bound to full contents",
              "plan_sha256": file_hash(args.plan),
              "first_attempts_sha256": file_hash(args.first_attempts),
              "retry_candidates_sha256": file_hash(args.retry_candidates),
              "rows": selected}
    args.candidates.parent.mkdir(parents=True, exist_ok=True)
    args.candidates.write_text(json.dumps(result, ensure_ascii=False,
                                           indent=2) + "\n")
    print(json.dumps({"sources": len(selected),
        "families": {key: len(value) for key, value in groups.items()}}),
        flush=True)


def checked_all_sources(args):
    candidates = json.loads(args.candidates.read_text())
    source_review = json.loads(args.source_review.read_text())
    first = json.loads(args.first_attempts.read_text())
    if (candidates["plan_sha256"] != file_hash(args.plan) or
            candidates["first_attempts_sha256"] !=
            file_hash(args.first_attempts) or
            candidates["retry_candidates_sha256"] !=
            file_hash(args.retry_candidates) or
            source_review["candidates_sha256"] != file_hash(args.candidates)):
        raise ValueError("Expanded source lineage changed")
    by_game = {item["game"]: item for item in first["games"]}
    approved = {note["input_content_sha256"]: note
                for note in source_review["annotations"]}
    if len(approved) != len(candidates["rows"]):
        raise ValueError("Expanded source review is incomplete")
    for row in candidates["rows"]:
        own = by_game[row["target_game"]]
        wrong = by_game[row["wrong_source_game"]]
        note = approved.get(row["input_content_sha256"])
        if (digest(row_content(row)) != row["input_content_sha256"] or
                file_hash(args.data_root / row["target_game"]) !=
                row["target_game_sha256"] or
                file_hash(args.data_root / row["wrong_source_game"]) !=
                wrong["game_sha256"] or
                digest(own["base"]) != row["source_episode_sha256"] or
                row["source_reward"] != own["base"]["reward"] or
                row["source_steps"] != own["base"]["steps"] or
                records_from_episode(own["base"]) != row["source_records"] or
                digest(wrong["base"]) !=
                row["wrong_source_episode_sha256"] or
                records_from_episode(wrong["base"]) != row["wrong_records"] or
                own["family"] != row["family"] or
                wrong["family"] != row["family"] or
                note is None or note["approved"] is not True or
                note["target_game"] != row["target_game"]):
            raise ValueError("Expanded source content or review changed")
    return candidates["rows"]


def utility(reward, steps):
    return float(reward) * (1.0 - .25 * (steps - 1) / 29.0)


def review(args):
    if any(path.exists() for path in (args.source_review, args.labels,
                                     args.label_review)):
        raise FileExistsError("Fresh expanded source/label paths required")
    candidates = json.loads(args.candidates.read_text())
    first = {item["game"]: item for item in
             json.loads(args.first_attempts.read_text())["games"]}
    notes, tasks, label_notes = [], [], []
    for row in candidates["rows"]:
        if (digest(row_content(row)) != row["input_content_sha256"] or
                file_hash(args.data_root / row["target_game"]) !=
                row["target_game_sha256"] or
                digest(first[row["target_game"]]["base"]) !=
                row["source_episode_sha256"] or
                digest(first[row["wrong_source_game"]]["base"]) !=
                row["wrong_source_episode_sha256"]):
            raise ValueError("Expanded source input changed before review")
        game = args.data_root / row["target_game"]
        replay_episode(game, first[row["target_game"]]["base"])
        commands = json.loads(game.read_text())["walkthrough"]
        if not commands or len(commands) > 30:
            raise ValueError("Train expert exceeds declared budget")
        env = make_env(game)
        try:
            state = env.reset()
            messages = [{"role": "system", "content": ACTOR_SYSTEM}]
            labels = []
            for index, command in enumerate(commands):
                available = list(state["admissible_commands"])
                if command not in available:
                    raise ValueError("Official train walkthrough inadmissible")
                messages.append({"role": "user", "content":
                    str(state["feedback"]) + "\nAvailable commands:\n" +
                    "\n".join(available)})
                content = {"target_game": row["target_game"],
                    "target_game_sha256": row["target_game_sha256"],
                    "source_episode_sha256": row["source_episode_sha256"],
                    "source_records": row["source_records"],
                    "teacher_arm": "walkthrough",
                    "target_messages": list(messages),
                    "target_action": command}
                label = {**content, "input_content_sha256": digest(content)}
                labels.append(label)
                label_notes.append({"target_game": row["target_game"],
                    "input_content_sha256": label["input_content_sha256"],
                    "source_episode_sha256": row["source_episode_sha256"],
                    "approved": True,
                    "review_basis": "Fresh official train expert action; admissible command and terminal won replayed after content-bound own first attempt"})
                state, _, done = env.step(command)
                if done and index != len(commands) - 1:
                    raise ValueError("Official train walkthrough ended early")
                messages.append({"role": "assistant", "content": command})
            if not state["won"]:
                raise ValueError("Official train walkthrough did not win")
        finally:
            env.close()
        failed_steps = first[row["target_game"]]["base"]["trajectory"]
        divergence = next((index for index, (expert, source_step)
            in enumerate(zip(commands, failed_steps))
            if expert != source_step["command"]), None)
        tasks.append({"target_game": row["target_game"],
            "target_game_sha256": row["target_game_sha256"],
            "source_episode_sha256": row["source_episode_sha256"],
            "teacher_arm": "walkthrough",
            "own_utility": utility(row["source_reward"], row["source_steps"]),
            "teacher_utility": utility(1, len(commands)),
            "first_changed_action": divergence if divergence is not None else 0,
            "has_changed_action": divergence is not None,
            "wrong_records": row["wrong_records"],
            "labels": labels})
        notes.append({"target_game": row["target_game"],
            "input_content_sha256": row["input_content_sha256"],
            "approved": True,
            "review_basis": "First train episode replayed exactly in official environment; same-family wrong episode content and train game hash checked"})
    args.source_review.parent.mkdir(parents=True, exist_ok=True)
    source_review = {"protocol": "Fresh reviewed source contents for all-first-attempt ALFWorld expert training",
                     "candidates_sha256": file_hash(args.candidates),
                     "annotations": notes}
    args.source_review.write_text(json.dumps(source_review,
        ensure_ascii=False, indent=2) + "\n")
    dataset = {"protocol": "Environment-confirmed train expert actions from 42 task-bound own first attempts; no retry development or official evaluation action labels",
               "source_scope": "all_first_attempts",
               "teacher_mode": "walkthrough",
               "source_review_sha256": file_hash(args.source_review),
               "tasks": tasks}
    args.labels.parent.mkdir(parents=True, exist_ok=True)
    args.labels.write_text(json.dumps(dataset, ensure_ascii=False,
                                      indent=2) + "\n")
    label_review = {"protocol": "Fresh action-level reviewed input-content bindings for all-first-attempt ALFWorld expert training",
                    "labels_sha256": file_hash(args.labels),
                    "annotations": label_notes}
    args.label_review.parent.mkdir(parents=True, exist_ok=True)
    args.label_review.write_text(json.dumps(label_review,
        ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"reviewed_sources": len(notes),
        "reviewed_actions": len(label_notes),
        "first_action_changes": sum(task["has_changed_action"]
                                     for task in tasks)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "review"))
    parser.add_argument("--plan", type=Path, default=Path(
        "ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json"))
    parser.add_argument("--first-attempts", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_rl_20261005/train_rewards_complete.json"))
    parser.add_argument("--retry-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_same_game_retry_candidates_20261006.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_expert_first_attempts_candidates_20261006.json"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alf_expert_first_attempts_reviewed_20261006.json"))
    parser.add_argument("--labels", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_expert_first_attempts_labels_20261006.json"))
    parser.add_argument("--label-review", type=Path, default=Path(
        "data/annotations/alf_expert_first_attempts_labels_reviewed_20261006.json"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        review(args)


if __name__ == "__main__":
    main()
