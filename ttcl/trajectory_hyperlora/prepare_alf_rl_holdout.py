"""Prepare fresh valid_seen ALFWorld task/source bindings for RL evaluation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from ttcl.experience_evolution.environment import make_env
from ttcl.trajectory_hyperlora.prepare_alf_next_task import (
    digest_json, family, partition, validate_review,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(plan_path: Path, historical_candidates: Path,
            historical_review: Path, data_root: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    train = {game for item in plan["training"] for game in item["games"]}
    evaluation = {game for item in plan["evaluation"] for game in item["games"]}
    approved = validate_review(json.loads(historical_candidates.read_text()),
                               json.loads(historical_review.read_text()))
    sources = defaultdict(dict)
    for row in approved:
        if row["source_game"] not in train or partition(row["source_game"]) != "train":
            raise ValueError("Historical source outside frozen training partition")
        previous = sources[row["family"]].get(row["source_game"])
        if previous and (previous["source_episode_sha256"] != row["source_episode_sha256"]
                         or previous["source_records"] != row["source_records"]):
            raise ValueError("Conflicting reviewed source contents")
        sources[row["family"]][row["source_game"]] = row
    paths = list((data_root / "json_2.1.1" / "valid_seen").glob("*/*/game.tw-pddl"))
    grouped = defaultdict(list)
    for path in paths:
        game = str(path.relative_to(data_root))
        if game in train or game in evaluation:
            raise ValueError("Valid_seen holdout overlaps frozen plan")
        grouped[family(game)].append(game)
    candidates = []
    for task_family in sorted(grouped):
        games = sorted(grouped[task_family], key=lambda game: digest_json(
            ["alf_rl_valid_seen_holdout_v1", game]))[:6]
        if len(games) != 6:
            raise ValueError("Need six valid_seen games per family")
        for target in games:
            choices = sources[task_family]
            source = choices[min(choices, key=lambda game: digest_json(
                ["alf_rl_holdout_source_v1", target, game]))] if choices else None
            environment = make_env(data_root / target)
            try:
                state = environment.reset()
                observation = str(state["feedback"])
            finally:
                environment.close()
            content = {"target_game": target,
                       "target_game_sha256": sha256(data_root / target),
                       "initial_observation": observation,
                       "source_game": source["source_game"] if source else None,
                       "source_episode_sha256": source["source_episode_sha256"]
                           if source else None,
                       "source_records": source["source_records"] if source else None}
            candidates.append({"family": task_family, **content,
                               "source_episode": source["source_episode"]
                                   if source else None,
                               "input_content_sha256": digest_json(content),
                               "reviewed_target": False})
    if len(candidates) != 36:
        raise ValueError("Expected 36 fresh holdout bindings")
    return {"protocol": "Fresh official valid_seen six-per-family holdout; no test actions or rewards; deterministic target/source mapping; all source trajectories reviewed train-only",
            "plan_sha256": sha256(plan_path),
            "historical_review_sha256": sha256(historical_review),
            "candidates": candidates}


def validate(manifest: dict, review: dict, plan_path: Path,
             data_root: Path) -> list[dict]:
    if (manifest["plan_sha256"] != sha256(plan_path) or
            review.get("plan_sha256") != manifest["plan_sha256"] or
            review.get("historical_review_sha256") != manifest["historical_review_sha256"]):
        raise ValueError("Holdout review lineage mismatch")
    plan = json.loads(plan_path.read_text())
    train = {game for item in plan["training"] for game in item["games"]}
    evaluation = {game for item in plan["evaluation"] for game in item["games"]}
    by_hash = {}
    for row in manifest["candidates"]:
        content = {key: row[key] for key in (
            "target_game", "target_game_sha256", "initial_observation",
            "source_game", "source_episode_sha256", "source_records")}
        key = digest_json(content)
        if key != row["input_content_sha256"] or key in by_hash:
            raise ValueError("Holdout content binding mismatch")
        by_hash[key] = row
    approved = []
    seen = set()
    for note in review["annotations"]:
        key = note["input_content_sha256"]
        if key not in by_hash or key in seen:
            raise ValueError("Unknown or duplicate holdout review")
        seen.add(key)
        row = by_hash[key]
        if (note.get("approved") is not True or not note.get("review_note") or
                note.get("target_game_sha256") != row["target_game_sha256"] or
                note.get("source_episode_sha256") != row["source_episode_sha256"]):
            raise ValueError("Holdout review source/target mismatch")
        target = row["target_game"]
        if ("/valid_seen/" not in target or target in train or target in evaluation or
                family(target) != row["family"] or
                sha256(data_root / target) != row["target_game_sha256"]):
            raise ValueError("Holdout target split or content changed")
        if row["source_game"] is not None:
            if (row["source_game"] not in train or
                    partition(row["source_game"]) != "train" or
                    family(row["source_game"]) != row["family"] or
                    sha256(Path(row["source_episode"])) != row["source_episode_sha256"]):
                raise ValueError("Holdout source changed")
        elif any(row[key] is not None for key in (
                "source_episode", "source_episode_sha256", "source_records")):
            raise ValueError("Unsupported source binding inconsistent")
        approved.append(row)
    if len(approved) != 36 or len(by_hash) != 36:
        raise ValueError("Every holdout task requires new reviewed binding")
    return approved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path(
        "ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json"))
    parser.add_argument("--historical-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_candidates_20261005.json"))
    parser.add_argument("--historical-review", type=Path, default=Path(
        "data/annotations/trajectory_hyperlora_alf_20261005_reviewed_v2.json"))
    parser.add_argument("--data-root", type=Path, default=Path("ttcl/data/alfworld_delta"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_rl_20261005/valid_seen_candidates.json"))
    parser.add_argument("--review", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        if not args.review:
            parser.error("Verification needs --review")
        rows = validate(json.loads(args.output.read_text()),
                        json.loads(args.review.read_text()), args.plan, args.data_root)
        print(json.dumps({"approved": len(rows)}))
        return
    if args.output.exists():
        parser.error("Fresh output required")
    manifest = prepare(args.plan, args.historical_candidates,
                       args.historical_review, args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"candidates": len(manifest["candidates"]),
                      "with_source": sum(row["source_game"] is not None
                                         for row in manifest["candidates"])}))


if __name__ == "__main__":
    main()
