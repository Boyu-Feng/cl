"""Read-only source-plan transfer ceiling on disjoint ALFWorld train tasks."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ttcl.experience_evolution.environment import make_env
from ttcl.trajectory_hyperlora.alfworld_same_game_retry import file_hash
from ttcl.trajectory_hyperlora.alfworld_sibling_transfer import checked_review
from ttcl.trajectory_hyperlora.audit_alf_sibling_replay import replay


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    reviewed = checked_review(args)
    if len(reviewed) != args.expected_tasks:
        raise ValueError("Holdout source replay task count changed")
    rows = []
    by_family = defaultdict(lambda: {"n": 0, "own_won": 0, "wrong_won": 0})
    for source, note in reviewed:
        game = args.data_root / source["target_game"]
        env = make_env(game)
        try:
            initial = str(env.reset()["feedback"])
        finally:
            env.close()
        own = replay(game, initial,
                     [step["action"] for step in note["source_records"]])
        wrong = replay(game, initial,
                       [step["action"] for step in note["wrong_records"]])
        rows.append({"target_game": source["target_game"],
            "input_content_sha256": source["input_content_sha256"],
            "own": own, "wrong": wrong})
        family = by_family[source["family"]]
        family["n"] += 1
        family["own_won"] += own["won"]
        family["wrong_won"] += wrong["won"]
    result = {"protocol": "Disjoint train-domain holdout, reviewed sibling expert commands replayed verbatim in target TextWorld environment; official won, no target walkthrough or hypernetwork inference",
        "candidates_sha256": file_hash(args.candidates),
        "source_review_sha256": file_hash(args.source_review),
        "summary": {"tasks": len(rows),
            "own_won": sum(row["own"]["won"] for row in rows),
            "wrong_won": sum(row["wrong"]["won"] for row in rows),
            "families": dict(by_family)},
        "tasks": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False,
                                      indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_holdout60_candidates_20261006.json"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alf_sibling_holdout60_reviewed_20261006.json"))
    parser.add_argument("--retry-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_same_game_retry_candidates_20261006.json"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/alf_same_game_retry_reviewed_20261006.json"))
    parser.add_argument("--all-source-review", type=Path, default=Path(
        "data/annotations/alf_expert_first_attempts_reviewed_20261006.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--split", choices=("train_large",), default="train_large")
    parser.add_argument("--large-offset", type=int, default=100)
    parser.add_argument("--family", type=str,
                        help="Optional single train_large family in reviewed candidates")
    parser.add_argument("--expected-tasks", type=int, default=60)
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_holdout60_source_replay_20261006.json"))
    args = parser.parse_args()
    if args.expected_tasks < 6 or args.large_offset < 0:
        parser.error("Invalid holdout protocol")
    print(json.dumps(run(args)["summary"]), flush=True)


if __name__ == "__main__":
    main()
