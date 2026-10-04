"""Fit a conservative family gate from paired next-game environment rewards.

For each family f, maximize the observed contextual-bandit objective
    sum_i gate(f_i) * (reward_lora_i - reward_base_i)
over binary gates, breaking ties in favor of the frozen base actor. The gate
changes only whether a trajectory-generated LoRA is mounted; no development or
official evaluation rewards enter fitting.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def game_family(game: str) -> str:
    return game.split("/", 3)[2].split("-", 1)[0]


def visible_family(initial_observation: str) -> str:
    """Classify the task from the actor-visible ALFWorld goal sentence."""
    marker = "Your task is to:"
    if marker not in initial_observation:
        raise ValueError("Initial observation lacks actor-visible goal")
    task = initial_observation.split(marker, 1)[1].split("\n", 1)[0].lower()
    if ("examine " in task or "look at " in task) and "desklamp" in task:
        return "look_at_obj_in_light"
    if "cool " in task or " cold " in task:
        return "pick_cool_then_place_in_recep"
    if "heat " in task or " hot " in task:
        return "pick_heat_then_place_in_recep"
    if "clean " in task or "wash " in task:
        return "pick_clean_then_place_in_recep"
    if "two " in task:
        return "pick_two_obj_and_place"
    if "put " in task:
        return "pick_and_place_simple"
    raise ValueError(f"Unrecognized actor-visible ALFWorld task: {task!r}")


def row_family(row: dict) -> str:
    family = visible_family(row["base"]["initial_observation"])
    if family != game_family(row["game"]):
        raise ValueError("Visible goal and frozen task family disagree")
    return family


def read_pairs(paths: list[Path], *, split: str,
               expected_checkpoint: str | None = None) -> tuple[list[dict], dict]:
    if not paths:
        raise ValueError("At least one paired report is required")
    rows = []
    seen = set()
    protocol = None
    for path in paths:
        report = json.loads(path.read_text())
        if report["split"] != split or report.get("constrain_actions") is not True:
            raise ValueError("Wrong split or unconstrained report")
        setting = {key: report[key] for key in (
            "checkpoint_sha256", "adapter_scale", "max_steps",
            "max_new_tokens", "constrain_actions")}
        if setting["adapter_scale"] != 1.0 or setting["max_steps"] != 50:
            raise ValueError("Gate requires the frozen scale and episode budget")
        if protocol is not None and setting != protocol:
            raise ValueError("Paired rollout settings differ")
        protocol = setting
        if expected_checkpoint and setting["checkpoint_sha256"] != expected_checkpoint:
            raise ValueError("Development checkpoint differs from fitted gate")
        for row in report["games"]:
            game = row["game"]
            if game in seen:
                raise ValueError("Duplicate target game across paired reports")
            seen.add(game)
            if row["base"]["status"] != "complete" or \
                    row["generated"]["status"] != "complete":
                raise ValueError("Incomplete pair is not a zero reward")
            if row["source_game"] == game:
                raise ValueError("Source and target must be distinct")
            for arm in ("base", "generated"):
                if row[arm]["reward"] not in (0.0, 1.0):
                    raise ValueError("Environment reward is missing or invalid")
                if row[arm]["initial_observation"] != row["base"]["initial_observation"]:
                    raise ValueError("Paired resets differ in initial observation")
            row_family(row)
            rows.append(row)
    return rows, protocol


def fit_gate(train: list[dict]) -> dict[str, dict]:
    by_family = defaultdict(list)
    for row in train:
        by_family[row_family(row)].append(
            row["generated"]["reward"] - row["base"]["reward"])
    return {family: {"n": len(deltas),
                     "positive": sum(delta > 0 for delta in deltas),
                     "negative": sum(delta < 0 for delta in deltas),
                     "reward_delta_sum": sum(deltas),
                     "use_lora": sum(deltas) > 0}
            for family, deltas in sorted(by_family.items())}


def score_gate(rows: list[dict], gate: dict[str, dict]) -> dict:
    details = []
    for row in rows:
        family = row_family(row)
        enabled = gate.get(family, {}).get("use_lora", False)
        details.append({"game": row["game"], "family": family,
                        "use_lora": enabled,
                        "base_reward": row["base"]["reward"],
                        "lora_reward": row["generated"]["reward"],
                        "gated_reward": row["generated" if enabled else "base"]["reward"]})
    return {"n": len(details),
            "base_successes": sum(x["base_reward"] for x in details),
            "always_lora_successes": sum(x["lora_reward"] for x in details),
            "gated_successes": sum(x["gated_reward"] for x in details),
            "details": details}


def leave_one_game_out_successes(rows: list[dict]) -> float:
    """Evaluate a gate fitted without each held-out training game."""
    return sum(score_gate([row], fit_gate(rows[:index] + rows[index + 1:]))[
        "gated_successes"] for index, row in enumerate(rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-report", action="append", type=Path, required=True)
    parser.add_argument("--development-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; do not overwrite fitted gate")
    train, settings = read_pairs(args.train_report, split="train")
    gate = fit_gate(train)
    result = {"protocol": "Family-level binary use gate fit only from full paired ALFWorld train-game rewards; positive signed sum enables LoRA, tie defaults to base; exploratory heldout development reporting",
              "settings": settings,
              "train_report_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in args.train_report},
              "gate": gate,
              "train": score_gate(train, gate),
              "train_leave_one_game_out_successes":
                  leave_one_game_out_successes(train)}
    if args.development_report:
        dev, dev_settings = read_pairs(
            [args.development_report], split="dev",
            expected_checkpoint=settings["checkpoint_sha256"])
        if dev_settings != settings:
            raise ValueError("Train/development rollout settings differ")
        if {row["game"] for row in train} & {row["game"] for row in dev}:
            raise ValueError("Train/development target overlap")
        result["development_report_sha256"] = hashlib.sha256(
            args.development_report.read_bytes()).hexdigest()
        result["development"] = score_gate(dev, gate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"gate": gate,
                      "train": {key: value for key, value in result["train"].items()
                                if key != "details"},
                      "train_leave_one_game_out_successes":
                          result["train_leave_one_game_out_successes"],
                      "development": {key: value for key, value in
                                      result.get("development", {}).items()
                                      if key != "details"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
