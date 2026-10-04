"""Bind frozen ALFWorld evaluation games to reviewed train-only LoRA sources.

No test action label or reward is read here. Each task/source binding requires
a separate reviewed record before any evaluation rollout can start.
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(plan_path: Path, historical_candidates: Path,
            historical_review: Path, data_root: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    train = {game for item in plan["training"] for game in item["games"]}
    evaluation = {game for item in plan["evaluation"] for game in item["games"]}
    if train & evaluation or len(evaluation) != 36:
        raise ValueError("Frozen evaluation split changed or overlaps training")
    approved = validate_review(
        json.loads(historical_candidates.read_text()),
        json.loads(historical_review.read_text()))
    sources = defaultdict(dict)
    for row in approved:
        game = row["source_game"]
        if game not in train or partition(game) != "train":
            raise ValueError("Reviewed source is outside train partition")
        prior = sources[row["family"]].get(game)
        if prior and (prior["source_episode_sha256"] != row["source_episode_sha256"]
                      or prior["source_records"] != row["source_records"]):
            raise ValueError("Conflicting reviewed source versions")
        sources[row["family"]][game] = row
    candidates = []
    for target in sorted(evaluation):
        task_family = family(target)
        choices = sources[task_family]
        source = choices[min(choices, key=lambda game: digest_json(
            ["frozen_eval_source", target, game]))] if choices else None
        content = {"target_game": target,
                   "target_game_sha256": sha256(data_root / target),
                   "source_game": source["source_game"] if source else None,
                   "source_episode_sha256": source["source_episode_sha256"]
                       if source else None,
                   "source_records": source["source_records"] if source else None}
        candidates.append({"family": task_family, **content,
                           "source_episode": source["source_episode"]
                               if source else None,
                           "input_content_sha256": digest_json(content),
                           "reviewed_target": False})
    return {"protocol": "Frozen official ALFWorld evaluation split task/source bindings; train-reviewed sources only; no target actions or rewards",
            "plan_sha256": sha256(plan_path),
            "historical_candidates_sha256": sha256(historical_candidates),
            "historical_review_sha256": sha256(historical_review),
            "train_games": len(train), "evaluation_games": len(evaluation),
            "candidates": candidates}


def validate_eval_review(manifest: dict, review: dict,
                         plan_path: Path, data_root: Path) -> list[dict]:
    if (review.get("plan_sha256") != manifest.get("plan_sha256") or
            review.get("historical_review_sha256") != manifest.get(
                "historical_review_sha256") or
            sha256(plan_path) != manifest["plan_sha256"]):
        raise ValueError("Reviewed evaluation lineage mismatch")
    plan = json.loads(plan_path.read_text())
    train = {game for item in plan["training"] for game in item["games"]}
    evaluation = {game for item in plan["evaluation"] for game in item["games"]}
    if train & evaluation:
        raise ValueError("Frozen train/test split overlaps")
    by_hash = {}
    for row in manifest["candidates"]:
        content = {key: row[key] for key in (
            "target_game", "target_game_sha256", "source_game",
            "source_episode_sha256", "source_records")}
        key = digest_json(content)
        if key != row["input_content_sha256"] or key in by_hash:
            raise ValueError("Evaluation input-content binding changed")
        by_hash[key] = row
    approved = []
    seen = set()
    for note in review["annotations"]:
        key = note["input_content_sha256"]
        if key not in by_hash or key in seen:
            raise ValueError("Unknown or duplicate evaluation review")
        seen.add(key)
        row = by_hash[key]
        if (note.get("approved") is not True or not note.get("review_note") or
                note.get("target_game_sha256") != row["target_game_sha256"] or
                note.get("source_episode_sha256") != row["source_episode_sha256"]):
            raise ValueError("Evaluation task/source review mismatch")
        if (row["target_game"] not in evaluation or
                row["target_game"] in train or
                family(row["target_game"]) != row["family"]):
            raise ValueError("Target is outside frozen evaluation split")
        if sha256(data_root / row["target_game"]) != row["target_game_sha256"]:
            raise ValueError("Frozen evaluation game content changed")
        if row["source_game"] is not None:
            if (row["source_game"] not in train or
                    partition(row["source_game"]) != "train" or
                    family(row["source_game"]) != row["family"] or
                    sha256(Path(row["source_episode"])) != row["source_episode_sha256"]):
                raise ValueError("Reviewed evaluation source changed")
        elif row["source_episode"] is not None or \
                row["source_episode_sha256"] is not None or \
                row["source_records"] is not None:
            raise ValueError("Unpaired evaluation task has inconsistent source")
        approved.append(row)
    if len(approved) != len(evaluation) or len(by_hash) != len(evaluation):
        raise ValueError("Every frozen evaluation game must be reviewed")
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
        "results/trajectory_hyperlora/alf_eval_pairs_20261005_candidates.json"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; do not overwrite frozen bindings")
    manifest = prepare(args.plan, args.historical_candidates,
                       args.historical_review, args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"evaluation_games": len(manifest["candidates"]),
                      "with_reviewed_source": sum(row["source_game"] is not None
                                                  for row in manifest["candidates"])},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
