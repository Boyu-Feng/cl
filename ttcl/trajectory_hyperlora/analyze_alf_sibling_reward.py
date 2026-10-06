"""Audit source-specific terminal advantages in paired ALFWorld rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_same_game_retry import digest, file_hash


def utility(episode: dict) -> float:
    if (episode["status"] != "complete" or
            episode["steps"] != len(episode["trajectory"]) or
            not 1 <= episode["steps"] <= 30 or
            episode["reward"] not in (0., 1.) or
            bool(episode["trajectory"][-1]["won"]) != bool(episode["reward"]) or
            episode["invalid_commands"] != 0):
        raise ValueError("Malformed or incomplete official rollout")
    return float(episode["reward"]) * (
        1. - .25 * (episode["steps"] - 1) / 29.)


def analyze(source: Path) -> dict:
    data = json.loads(source.read_text())
    if (data.get("summary", {}).get("n") != len(data["games"]) or
            data["failures"] or
            len({row["game"] for row in data["games"]}) != len(data["games"]) or
            len(data["games"]) == 0 or
            "official won" not in data["protocol"]):
        raise ValueError("Reward analysis needs complete unique official rollouts")
    rows = []
    for game in data["games"]:
        arms = game["arms"]
        if set(arms) != {"base", "own", "wrong"} or len({
            episode["initial_observation"] for episode in arms.values()}) != 1:
            raise ValueError("Paired arms do not share one target reset")
        values = {arm: utility(episode) for arm, episode in arms.items()}
        advantage = values["own"] - max(values["base"], values["wrong"])
        rows.append({"game": game["game"], "family": game["family"],
            "arm_trajectory_sha256": {arm: digest(episode["trajectory"])
                                      for arm, episode in arms.items()},
            "utility": values, "source_specific_advantage": advantage,
            "own_only_terminal_win": bool(arms["own"]["reward"] and
                not arms["base"]["reward"] and
                not arms["wrong"]["reward"]),
            "wrong_only_terminal_win": bool(arms["wrong"]["reward"] and
                not arms["base"]["reward"] and
                not arms["own"]["reward"])})
    return {"protocol": "Read-only official terminal advantage audit; same paired reset and budget, no policy fitting",
        "rollouts_sha256": file_hash(source),
        "checkpoint_sha256": data["checkpoint_sha256"],
        "candidates_sha256": data["candidates_sha256"],
        "source_review_sha256": data["source_review_sha256"],
        "summary": {"games": len(rows),
            "terminal_wins": {arm: sum(game["arms"][arm]["reward"]
                for game in data["games"]) for arm in ("base", "own", "wrong")},
            "own_only_terminal_wins": sum(row["own_only_terminal_win"]
                                           for row in rows),
            "wrong_only_terminal_wins": sum(row["wrong_only_terminal_win"]
                                             for row in rows),
            "positive_advantage_games": sum(
                row["source_specific_advantage"] > 0 for row in rows),
            "negative_advantage_games": sum(
                row["source_specific_advantage"] < 0 for row in rows),
            "advantage_sum": sum(row["source_specific_advantage"]
                                 for row in rows)},
        "games": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.rollouts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False,
                                      indent=2) + "\n")
    print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
