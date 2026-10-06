"""Paired official-win audit for frozen ALFWorld LoRA versus no-LoRA runs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from math import comb
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_same_game_retry import file_hash


def exact_discordant_p(improved: int, regressed: int) -> float:
    """Two-sided exact sign/McNemar probability for paired binary rewards."""
    total = improved + regressed
    if total == 0:
        return 1.0
    tail = sum(comb(total, k) for k in range(min(improved, regressed) + 1))
    return min(1.0, 2.0 * tail / 2**total)


def arm_summary(games: list[dict], arm: str) -> dict:
    improved = sum(game["arms"][arm]["reward"] == 1.0 and
                   game["arms"]["base"]["reward"] == 0.0 for game in games)
    regressed = sum(game["arms"][arm]["reward"] == 0.0 and
                    game["arms"]["base"]["reward"] == 1.0 for game in games)
    wins = sum(game["arms"][arm]["reward"] for game in games)
    return {"games": len(games), "wins": int(wins),
            "wins_above_base": int(wins - sum(
                game["arms"]["base"]["reward"] for game in games)),
            "improved_games": improved, "regressed_games": regressed,
            "discordant_exact_two_sided_p": exact_discordant_p(
                improved, regressed)}


def analyze(path: Path, *, expected_games: int,
            candidates: Path, source_review: Path) -> dict:
    report = json.loads(path.read_text())
    games = report["games"]
    if (report["summary"]["n"] != expected_games or
            len(games) != expected_games or report["failures"] or
            len({game["game"] for game in games}) != expected_games or
            report["candidates_sha256"] != file_hash(candidates) or
            report["source_review_sha256"] != file_hash(source_review) or
            "official won" not in report["protocol"]):
        raise ValueError("Overall gain audit needs complete frozen paired rollouts")
    by_family = defaultdict(list)
    for game in games:
        arms = game["arms"]
        if (set(arms) != {"base", "own", "wrong"} or
                len({arm["initial_observation"] for arm in arms.values()}) != 1 or
                len({arm["initial_commands_sha256"] for arm in arms.values()}) != 1 or
                any(arm["status"] != "complete" or
                    arm["reward"] not in (0.0, 1.0) or
                    arm["invalid_commands"] != 0 or
                    not 1 <= arm["steps"] <= 30 or
                    arm["steps"] != len(arm["trajectory"]) or
                    bool(arm["trajectory"][-1]["won"]) != bool(arm["reward"])
                    for arm in arms.values())):
            raise ValueError("Paired official-win record is incomplete")
        by_family[game["family"]].append(game)
    return {"protocol": "Read-only paired official ALFWorld terminal-success gain audit; exact discordant sign test is descriptive; no model selection or training",
        "rollouts_sha256": file_hash(path),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "candidates_sha256": report["candidates_sha256"],
        "source_review_sha256": report["source_review_sha256"],
        "actor_history_turns": report.get("actor_history_turns", 0),
        "summary": {"base_wins": int(sum(g["arms"]["base"]["reward"]
                                           for g in games)),
                    "own_vs_base": arm_summary(games, "own"),
                    "wrong_vs_base": arm_summary(games, "wrong")},
        "families": {family: {"base_wins": int(sum(
            g["arms"]["base"]["reward"] for g in rows)),
            "own_vs_base": arm_summary(rows, "own"),
            "wrong_vs_base": arm_summary(rows, "wrong")}
            for family, rows in sorted(by_family.items())}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--expected-games", type=int, default=60)
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_holdout60_candidates_20261006.json"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alf_sibling_holdout60_reviewed_20261006.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.rollouts, expected_games=args.expected_games,
                     candidates=args.candidates, source_review=args.source_review)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
