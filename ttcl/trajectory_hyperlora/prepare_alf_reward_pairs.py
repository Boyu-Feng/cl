"""Select new ALFWorld train games for reviewed paired reward collection.

These candidates have no action label. The target annotation reviews the game
and source binding; the reward is obtained only by executing the environment.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.prepare_alf_next_task import (
    digest_json, family, partition, validate_review,
)


DEFAULT_QUOTAS = {
    "pick_and_place_simple": 8,
    "pick_cool_then_place_in_recep": 4,
    "pick_heat_then_place_in_recep": 4,
    "pick_two_obj_and_place": 4,
}


def prepare(plan_path: Path, candidate_path: Path, review_path: Path,
            data_root: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    historical = json.loads(candidate_path.read_text())
    approved = validate_review(historical, json.loads(review_path.read_text()))
    train_games = {game for sequence in plan["training"]
                   for game in sequence["games"]}
    evaluation_games = {game for sequence in plan["evaluation"]
                        for game in sequence["games"]}
    if train_games & evaluation_games:
        raise ValueError("Frozen training and evaluation games overlap")
    historical_targets = {row["target_game"] for row in historical["candidates"]}
    source_by_family = defaultdict(dict)
    for row in approved:
        if row["split"] != "train":
            continue
        prior = source_by_family[row["family"]].get(row["source_game"])
        if prior and (prior["source_records"] != row["source_records"] or
                      prior["source_episode_sha256"] != row["source_episode_sha256"]):
            raise ValueError("Conflicting reviewed source versions")
        source_by_family[row["family"]][row["source_game"]] = row
    candidates = []
    for task_family, quota in DEFAULT_QUOTAS.items():
        sources = source_by_family[task_family]
        if not sources:
            raise ValueError(f"No reviewed train source for {task_family}")
        eligible = [game for game in train_games
                    if family(game) == task_family and partition(game) == "train"
                    and game not in historical_targets]
        eligible.sort(key=lambda game: digest_json(["reward_pair", game]))
        if len(eligible) < quota:
            raise ValueError(f"Not enough eligible train games: {task_family}")
        for target_game in eligible[:quota]:
            source_game = min(sources, key=lambda game: digest_json(
                ["reward_source", target_game, game]))
            source = sources[source_game]
            game_path = data_root / target_game
            target_sha = hashlib.sha256(game_path.read_bytes()).hexdigest()
            content = {"source_game": source_game,
                       "source_episode_sha256": source["source_episode_sha256"],
                       "source_records": source["source_records"],
                       "target_game": target_game,
                       "target_game_sha256": target_sha}
            candidates.append({"split": "train", "family": task_family,
                               **content,
                               "source_episode": source["source_episode"],
                               "input_content_sha256": digest_json(content),
                               "reviewed_target": False})
    return {"protocol": "New train-only ALFWorld paired reward targets; reviewed source trajectory; no action labels or benchmark evaluation games",
            "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            "historical_candidates_sha256": hashlib.sha256(
                candidate_path.read_bytes()).hexdigest(),
            "historical_review_sha256": hashlib.sha256(
                review_path.read_bytes()).hexdigest(),
            "quotas": DEFAULT_QUOTAS, "candidates": candidates}


def validate_reward_review(manifest: dict, review: dict,
                           data_root: Path) -> list[dict]:
    if review.get("plan_sha256") != manifest.get("plan_sha256") or \
            review.get("historical_review_sha256") != manifest.get(
                "historical_review_sha256"):
        raise ValueError("Reviewed plan/source lineage mismatch")
    by_hash = {}
    for row in manifest["candidates"]:
        content = {key: row[key] for key in (
            "source_game", "source_episode_sha256", "source_records",
            "target_game", "target_game_sha256")}
        key = digest_json(content)
        if key != row["input_content_sha256"] or key in by_hash:
            raise ValueError("Candidate input-content binding mismatch")
        by_hash[key] = row
    approved = []
    seen = set()
    for note in review["annotations"]:
        key = note["input_content_sha256"]
        if key in seen or key not in by_hash:
            raise ValueError("Duplicate or unknown reviewed target")
        seen.add(key)
        row = by_hash[key]
        if (note.get("approved") is not True or not note.get("review_note") or
                note.get("source_episode_sha256") != row["source_episode_sha256"] or
                note.get("target_game_sha256") != row["target_game_sha256"]):
            raise ValueError("Reviewed target or source hash mismatch")
        if (row["split"] != "train" or partition(row["target_game"]) != "train" or
                family(row["source_game"]) != row["family"] or
                family(row["target_game"]) != row["family"] or
                row["source_game"] == row["target_game"]):
            raise ValueError("Source/target train isolation mismatch")
        if hashlib.sha256((data_root / row["target_game"]).read_bytes()).hexdigest() \
                != row["target_game_sha256"]:
            raise ValueError("Target game content changed")
        if hashlib.sha256(Path(row["source_episode"]).read_bytes()).hexdigest() \
                != row["source_episode_sha256"]:
            raise ValueError("Source episode content changed")
        approved.append(row)
    if len(approved) != len(manifest["candidates"]):
        raise ValueError("Every new reward target needs a reviewed annotation")
    return approved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path(
        "ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json"))
    parser.add_argument("--historical-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_candidates_20261005.json"))
    parser.add_argument("--historical-review", type=Path, default=Path(
        "data/annotations/trajectory_hyperlora_alf_20261005_reviewed_v2.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_reward_pairs_20261005_candidates.json"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Candidate manifest exists; do not overwrite frozen inputs")
    manifest = prepare(args.plan, args.historical_candidates,
                       args.historical_review, args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"candidates": len(manifest["candidates"]),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
