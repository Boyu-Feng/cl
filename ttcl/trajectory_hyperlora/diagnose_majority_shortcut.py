"""Audit whether a trajectory-to-LoRA result merely copies its majority action."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def diagnose(rows: list[dict]) -> dict:
    episodes: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        episodes[(row.get("cue_count", 4), row["rule"],
                  row["source_seed"])].append(row)
    if not episodes:
        raise ValueError("No scored episodes")
    cue_sensitive = 0
    correct = wrong = majority_correct = 0
    for (count, rule, _), items in episodes.items():
        predicted_by_cue: dict[str, set[str]] = defaultdict(set)
        for row in items:
            predicted_by_cue[row["cue"]].add(row["correct"]["first_token"])
            correct += row["correct"]["correct"]
            wrong += row["wrong"]["correct"]
        if len(predicted_by_cue) != count:
            raise ValueError("Episode does not cover every cue")
        if len(set.union(*predicted_by_cue.values())) > 1:
            cue_sensitive += 1
        right_count = rule.bit_count()
        if right_count != count - right_count:
            majority_correct += sum(
                row["expected"] ==
                ("RIGHT" if right_count > count - right_count else "LEFT")
                for row in items)
    return {
        "episodes": len(episodes), "queries": len(rows),
        "cue_sensitive_episodes": cue_sensitive,
        "correct_source_success": correct,
        "wrong_source_success": wrong,
        "majority_action_baseline_success": majority_correct,
        "majority_defined_for_every_episode": all(
            rule.bit_count() != count - rule.bit_count()
            for count, rule, _ in episodes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.results:
        data = json.loads(path.read_text())
        rows = data["test"]["rows"] if "test" in data else data["rows"]
        print(json.dumps({"path": str(path), **diagnose(rows)}))


if __name__ == "__main__":
    main()
