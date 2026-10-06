"""Read-only lexical control for ALFWorld sibling-source matching.

The control only uses the task sentence in the initial observation. It never
reads a source action, feedback, target expert action, or environment reward.
It is a diagnostic baseline, not a rule used by the LoRA policy.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from ttcl.experience_evolution.environment import make_env
from ttcl.trajectory_hyperlora.alfworld_same_game_retry import digest, file_hash
from ttcl.trajectory_hyperlora.calibrate_alf_sibling_gate import checked_tasks
from ttcl.trajectory_hyperlora.train_alf_pair_matcher import dev_pairs


TASK = re.compile(r"Your task is to:\s*([^\n]+)", re.IGNORECASE)
WORD = re.compile(r"[a-z0-9]+")


def task_sentence(observation: str) -> str:
    match = TASK.search(observation)
    if match is None:
        raise ValueError("Public initial observation lacks task sentence")
    return match.group(1).strip().casefold()


def similarity(target: str, source: str) -> float:
    left = set(WORD.findall(task_sentence(target)))
    right = set(WORD.findall(task_sentence(source)))
    if not left or not right:
        raise ValueError("Task sentence has no words")
    return len(left & right) / len(left | right)


def summarize(rows: list[dict]) -> dict:
    return {"pairs": len(rows),
        "own_above_wrong": sum(row["own"] > row["wrong"] for row in rows),
        "tie": sum(row["own"] == row["wrong"] for row in rows),
        "own_exact_sentence": sum(row["own_sentence"] == row["target_sentence"]
                                  for row in rows),
        "wrong_exact_sentence": sum(row["wrong_sentence"] == row["target_sentence"]
                                    for row in rows)}


def score_row(game: str, target: str, own_records: list,
              wrong_records: list) -> dict:
    if not own_records or not wrong_records:
        raise ValueError("Reviewed source history is empty")
    own = own_records[0]["observation"]
    wrong = wrong_records[0]["observation"]
    return {"target_game": game,
        "target_sentence": task_sentence(target),
        "own_sentence": task_sentence(own),
        "wrong_sentence": task_sentence(wrong),
        "own": similarity(target, own),
        "wrong": similarity(target, wrong)}


def run(args) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    tasks = checked_tasks(args, expected_tasks=args.expected_tasks)
    groups = defaultdict(list)
    for task in tasks:
        family = task["target_game"].split("/")[2].split("-", 1)[0]
        label = task["labels"][0]
        observation = label["target_messages"][1]["content"].split(
            "\nAvailable commands:\n", 1)[0]
        groups[family].append(score_row(task["target_game"], observation,
            label["source_records"], task["wrong_records"]))
    fit, holdout = [], []
    for family, rows in sorted(groups.items()):
        if len(rows) != args.expected_tasks // 6:
            raise ValueError("Lexical control family size changed")
        ordered = sorted(rows, key=lambda row: digest([
            "pair_matcher_split", row["target_game"]]))
        count = 4 * len(ordered) // 5
        fit.extend(ordered[:count])
        holdout.extend(ordered[count:])
    development = []
    for row, note in dev_pairs(args):
        env = make_env(args.data_root / row["target_game"])
        try:
            observation = str(env.reset()["feedback"])
        finally:
            env.close()
        development.append(score_row(row["target_game"], observation,
            note["source_records"], note["wrong_records"]))
    result = {"protocol": "Read-only first-observation task-sentence Jaccard control; same train task-directory split as linear matcher; no source actions, target walkthrough or rewards",
        "labels_sha256": file_hash(args.labels),
        "label_review_sha256": file_hash(args.label_review),
        "source_review_sha256": file_hash(args.source_review),
        "dev_candidates_sha256": file_hash(args.dev_candidates),
        "dev_source_review_sha256": file_hash(args.dev_source_review),
        "expected_tasks": args.expected_tasks,
        "fit": summarize(fit),
        "internal_holdout": summarize(holdout),
        "development": summarize(development),
        "development_scores": development}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False,
                                      indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train600_labels_20261006.json"))
    parser.add_argument("--label-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train600_labels_reviewed_20261006.json"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train600_reviewed_20261006.json"))
    parser.add_argument("--dev-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_dev_candidates_20261006.json"))
    parser.add_argument("--dev-source-review", type=Path, default=Path(
        "data/annotations/alf_sibling_dev_reviewed_20261006.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--expected-tasks", type=int, default=600)
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_goal_overlap600_20261006.json"))
    args = parser.parse_args()
    if args.expected_tasks < 12 or args.expected_tasks % 6:
        parser.error("Expected tasks must cover six families")
    result = run(args)
    print(json.dumps({key: result[key] for key in (
        "fit", "internal_holdout", "development")}), flush=True)


if __name__ == "__main__":
    main()
