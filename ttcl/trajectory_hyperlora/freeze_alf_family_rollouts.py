"""Freeze one fully completed ALFWorld train family from a larger live run.

The parent evaluator writes atomic progress snapshots. This makes a new,
content-bound derived artifact only after every task in the requested family
has all three official rollout arms, while the other families may still run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_same_game_retry import file_hash


def freeze(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.parent.read_bytes()
    parent = json.loads(raw)
    if (parent["summary"]["n"] != len(parent["games"]) or
            parent["failures"] or
            "train_large" not in parent["protocol"] or
            parent["candidates_sha256"] != file_hash(args.candidates) or
            parent["source_review_sha256"] != file_hash(args.source_review) or
            parent.get("per_family_limit") != args.expected_games):
        raise ValueError("Parent train rollout snapshot changed")
    games = [game for game in parent["games"]
             if game["family"] == args.family]
    if (len(games) != args.expected_games or
            len({game["game"] for game in games}) != len(games) or
            any(set(game["arms"]) != {"base", "own", "wrong"} or
                any(arm["status"] != "complete" or
                    arm["reward"] not in (0.0, 1.0) or
                    arm["invalid_commands"] != 0
                    for arm in game["arms"].values())
                for game in games)):
        raise ValueError("Selected family lacks complete official paired rollouts")
    frozen = {key: value for key, value in parent.items()
              if key not in ("protocol", "games", "failures", "summary")}
    frozen.update({"protocol": parent["protocol"] +
        "; immutable completed-family subset of live train run",
        "parent_snapshot_sha256": hashlib.sha256(raw).hexdigest(),
        "parent_completed_games": len(parent["games"]),
        "selection_family": args.family,
        "games": games, "failures": [],
        "summary": {"n": len(games), "failures": 0,
            **{arm: sum(game["arms"][arm]["reward"] for game in games)
               for arm in ("base", "own", "wrong")}}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(frozen, ensure_ascii=False,
                                      indent=2) + "\n")
    print(json.dumps({"family": args.family,
        "parent_completed_games": len(parent["games"]),
        "summary": frozen["summary"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train240_taskpair_current1000_20261006.json"))
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train600_reviewed_20261006.json"))
    parser.add_argument("--family", default="pick_and_place_simple")
    parser.add_argument("--expected-games", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple_train40_frozen_taskpair_current1000_20261006.json"))
    args = parser.parse_args()
    if args.expected_games < 1:
        parser.error("Expected completed family count must be positive")
    freeze(args)


if __name__ == "__main__":
    main()
