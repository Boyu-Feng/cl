"""Prepare auditable ALFWorld train-only source -> future action candidates.

The output is *unreviewed*. Training code must require separately approved
annotation targets bound to each candidate's complete input-content hash.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from ttcl.experience_evolution.environment import ACTOR_SYSTEM, make_env
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import source_records


def digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()


def family(game: str) -> str:
    return game.split("/", 3)[2].split("-", 1)[0]


def partition(game: str) -> str:
    return "dev" if int(hashlib.sha256(game.encode()).hexdigest(), 16) % 5 == 0 else "train"


def successful_episodes(root: Path, train_games: set[str]) -> dict[str, Path]:
    chosen = {}
    for path in sorted((root / "training" / "delta").glob(
            "batch_*/seq_*/*/episode.json")):
        episode = json.loads(path.read_text())
        game = episode.get("game")
        if (game not in train_games or episode.get("status") != "complete" or
                episode.get("reward") != 1 or not episode.get("trajectory")):
            continue
        key = (sum(step.get("valid_command") is False
                   for step in episode["trajectory"]),
               episode["steps"], str(path))
        previous = chosen.get(game)
        if previous is None or key < previous[0]:
            chosen[game] = (key, path)
    return {game: path for game, (_, path) in chosen.items()}


def replay_targets(game_path: Path, episode: dict,
                   max_targets: int, selection: str = "last") -> list[dict]:
    environment = make_env(game_path)
    try:
        state = environment.reset()
        if str(state["feedback"]) != episode["initial_observation"]:
            raise ValueError("Initial observation mismatch")
        messages = [{"role": "system", "content": ACTOR_SYSTEM}]
        eligible = []
        for turn, step in enumerate(episode["trajectory"]):
            available = list(state["admissible_commands"])
            messages.append({"role": "user", "content": str(state["feedback"]) +
                             "\nAvailable commands:\n" + "\n".join(available)})
            command = step["action"]
            if (command in available) != bool(step["valid_command"]):
                raise ValueError(f"Admissibility mismatch at turn {turn}")
            if step["valid_command"] and command != "look":
                eligible.append({"turn": turn, "messages": list(messages),
                                 "target_action": command,
                                 "available_commands": available})
            state, _, _ = environment.step(command)
            if str(state["feedback"]) != step["observation"]:
                raise ValueError(f"Observation mismatch at turn {turn}")
            messages.append({"role": "assistant", "content": command})
        if not state["won"]:
            raise ValueError("Historical success did not replay as a win")
        if selection == "last":
            return eligible[-max_targets:]
        if selection == "even":
            count = min(max_targets, len(eligible))
            if count == 1:
                return eligible[-1:]
            return [eligible[(index * (len(eligible) - 1) + (count - 1) // 2)
                             // (count - 1)] for index in range(count)]
        raise ValueError(f"Unknown target selection: {selection}")
    finally:
        environment.close()


def prepare(root: Path, data_root: Path, *, max_targets: int = 2,
            target_selection: str = "last") -> dict:
    if max_targets < 1 or target_selection not in {"last", "even"}:
        raise ValueError("Invalid target selection or count")
    plan_path = root / "plan.json"
    plan = json.loads(plan_path.read_text())
    train_games = {game for sequence in plan["training"] for game in sequence["games"]}
    eval_games = {game for sequence in plan["evaluation"] for game in sequence["games"]}
    if train_games & eval_games:
        raise ValueError("Frozen plan has overlapping games")
    episodes = successful_episodes(root, train_games)
    sources_by_family = defaultdict(list)
    for game in sorted(episodes):
        if partition(game) == "train":
            sources_by_family[family(game)].append(game)
    candidates, failures = [], []
    for target_game, target_path in sorted(episodes.items()):
        source_choices = [game for game in sources_by_family[family(target_game)]
                          if game != target_game]
        if not source_choices:
            failures.append({"game": target_game, "reason": "no_distinct_train_source"})
            continue
        source_game = min(source_choices, key=lambda game: digest_json(
            [target_game, game]))
        source_path = episodes[source_game]
        source_episode = json.loads(source_path.read_text())
        target_episode = json.loads(target_path.read_text())
        game_path = data_root / target_game
        if not game_path.is_file():
            failures.append({"game": target_game, "reason": "missing_game"})
            continue
        try:
            targets = replay_targets(game_path, target_episode, max_targets,
                                     target_selection)
        except Exception as error:
            failures.append({"game": target_game, "reason": repr(error)})
            continue
        source_public = source_records(source_episode)
        for target in targets:
            input_content = {"source_records": source_public,
                             "target_game": target_game,
                             "target_messages": target["messages"],
                             "target_action": target["target_action"]}
            candidates.append({
                "split": partition(target_game),
                "family": family(target_game),
                "source_game": source_game,
                "target_game": target_game,
                "source_episode": str(source_path),
                "target_episode": str(target_path),
                "source_episode_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "target_episode_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
                "target_game_sha256": hashlib.sha256(game_path.read_bytes()).hexdigest(),
                "turn": target["turn"],
                "source_records": source_public,
                "target_messages": target["messages"],
                "target_action": target["target_action"],
                "input_content_sha256": digest_json(input_content),
                "reviewed_target": False,
            })
    return {"protocol": "Unreviewed ALFWorld train-only source -> future successful action candidates; official test games excluded; replay-verified; no training permission inferred",
            "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            "train_games": len(train_games), "official_evaluation_games": len(eval_games),
            "successful_train_games": len(episodes),
            "candidate_count": len(candidates), "candidates": candidates,
            "failures": failures}


def validate_review(manifest: dict, review: dict) -> list[dict]:
    """Require explicit reviewed targets bound to current source/query content."""
    if review.get("plan_sha256") != manifest.get("plan_sha256"):
        raise ValueError("Reviewed plan hash does not match candidate manifest")
    by_hash = {}
    for row in manifest["candidates"]:
        content = {"source_records": row["source_records"],
                   "target_game": row["target_game"],
                   "target_messages": row["target_messages"],
                   "target_action": row["target_action"]}
        if digest_json(content) != row["input_content_sha256"]:
            raise ValueError("Candidate content hash mismatch")
        key = row["input_content_sha256"]
        if key in by_hash:
            raise ValueError("Duplicate candidate content hash")
        by_hash[key] = row
    approved = []
    seen = set()
    for annotation in review["annotations"]:
        key = annotation["input_content_sha256"]
        if key in seen or key not in by_hash:
            raise ValueError("Duplicate or unknown reviewed input")
        seen.add(key)
        row = by_hash[key]
        if (annotation.get("approved") is not True or
                annotation.get("target_action") != row["target_action"] or
                annotation.get("source_episode_sha256") != row["source_episode_sha256"] or
                annotation.get("target_episode_sha256") != row["target_episode_sha256"] or
                not annotation.get("review_note")):
            raise ValueError("Reviewed target or source binding mismatch")
        for path_key, hash_key in (("source_episode", "source_episode_sha256"),
                                   ("target_episode", "target_episode_sha256")):
            path = Path(row[path_key])
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row[hash_key]:
                raise ValueError(f"Reviewed source changed: {path}")
        approved.append(row)
    if not approved:
        raise ValueError("No reviewed targets")
    train_targets = {row["target_game"] for row in approved if row["split"] == "train"}
    dev_targets = {row["target_game"] for row in approved if row["split"] == "dev"}
    train_sources = {row["source_game"] for row in approved if row["split"] == "train"}
    if train_targets & dev_targets or train_sources & dev_targets:
        raise ValueError("Reviewed train/dev game leakage")
    return approved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(
        "ttcl/results/experience_evolution/alfworld_delta_20260922"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--max-targets", type=int, default=2)
    parser.add_argument("--target-selection", choices=("last", "even"),
                        default="last")
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_candidates_20261005.json"))
    parser.add_argument("--reviewed-annotations", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        if args.reviewed_annotations is None:
            parser.error("--verify-only requires --reviewed-annotations")
        manifest = json.loads(args.output.read_text())
        review = json.loads(args.reviewed_annotations.read_text())
        approved = validate_review(manifest, review)
        print(json.dumps({"approved": len(approved),
                          "train": sum(row["split"] == "train" for row in approved),
                          "dev": sum(row["split"] == "dev" for row in approved)}))
        return
    result = prepare(args.root, args.data_root, max_targets=args.max_targets,
                     target_selection=args.target_selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"successful_train_games": result["successful_train_games"],
                      "candidate_count": result["candidate_count"],
                      "failures": len(result["failures"]),
                      "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
