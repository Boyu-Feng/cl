"""Review train-only actions whose correct sibling history is informative.

An action is selected when the same-index command in a distinct successful
sibling trajectory matches the target expert action while the reviewed wrong
history does not. This is a data selection criterion, not an ALFWorld rule or
an evaluation label; every selection binds the full source, control, and
target-action content hashes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_same_game_retry import digest, file_hash
from ttcl.trajectory_hyperlora.train_alfworld_retry_reward import label_content


def content(row):
    return {key: row[key] for key in (
        "target_game", "target_game_sha256", "action_index",
        "action_input_sha256", "source_records_sha256",
        "wrong_records_sha256", "expert_action", "wrong_action",
        "wrong_action_admissible")}


def selected_rows(args):
    dataset = json.loads(args.labels.read_text())
    review = json.loads(args.label_review.read_text())
    if (dataset["source_scope"] != "sibling_expert" or
            dataset.get("split") != "train_large" or
            dataset["sibling_source_review_sha256"] !=
                file_hash(args.source_review) or
            review["labels_sha256"] != file_hash(args.labels)):
        raise ValueError("Sibling action evidence lineage changed")
    approved = {note["input_content_sha256"]: note
                for note in review["annotations"]}
    rows = []
    for task in dataset["tasks"]:
        own = task["labels"][0]["source_records"]
        wrong = task["wrong_records"]
        for index, label in enumerate(task["labels"]):
            note = approved.get(label["input_content_sha256"])
            if (digest(label_content(label)) !=
                    label["input_content_sha256"] or
                    note is None or note["approved"] is not True or
                    note["target_game"] != task["target_game"]):
                raise ValueError("Unreviewed sibling target action")
            if (index >= len(own) or index >= len(wrong) or
                    own[index]["action"] != label["target_action"] or
                    wrong[index]["action"] == label["target_action"]):
                continue
            available = label["target_messages"][-1]["content"].split(
                "\nAvailable commands:\n", 1)[-1].splitlines()
            row = {"target_game": task["target_game"],
                "target_game_sha256": task["target_game_sha256"],
                "action_index": index,
                "action_input_sha256": label["input_content_sha256"],
                "source_records_sha256": digest(own),
                "wrong_records_sha256": digest(wrong),
                "expert_action": label["target_action"],
                "wrong_action": wrong[index]["action"],
                "wrong_action_admissible": wrong[index]["action"] in available}
            row["selection_content_sha256"] = digest(content(row))
            rows.append(row)
    if not rows or len({row["selection_content_sha256"] for row in rows}) != len(rows):
        raise ValueError("No unique train-only sibling action evidence")
    return rows


def prepare(args):
    if args.candidates.exists():
        raise FileExistsError(args.candidates)
    rows = selected_rows(args)
    result = {"protocol": "Train-only same-index action evidence from reviewed sibling expert sources and separately reviewed target expert actions; no development input",
        "labels_sha256": file_hash(args.labels),
        "label_review_sha256": file_hash(args.label_review),
        "source_review_sha256": file_hash(args.source_review),
        "rows": rows}
    args.candidates.parent.mkdir(parents=True, exist_ok=True)
    args.candidates.write_text(json.dumps(result, ensure_ascii=False,
                                          indent=2) + "\n")
    print(json.dumps({"selected_actions": len(rows),
        "target_games": len({row["target_game"] for row in rows}),
        "admissible_wrong_actions": sum(row["wrong_action_admissible"]
                                         for row in rows)}), flush=True)


def review(args):
    if args.selection_review.exists():
        raise FileExistsError(args.selection_review)
    candidate = json.loads(args.candidates.read_text())
    if (candidate["labels_sha256"] != file_hash(args.labels) or
            candidate["label_review_sha256"] != file_hash(args.label_review) or
            candidate["source_review_sha256"] != file_hash(args.source_review) or
            candidate["rows"] != selected_rows(args)):
        raise ValueError("Sibling action evidence changed before review")
    notes = [{"target_game": row["target_game"],
        "selection_content_sha256": row["selection_content_sha256"],
        "action_input_sha256": row["action_input_sha256"],
        "approved": True,
        "review_basis": "Matched source action, mismatched control action, and target expert action freshly checked against content-bound reviewed train trajectories"}
        for row in candidate["rows"]]
    result = {"protocol": "Fresh selection-level annotation for train-only sibling action evidence",
        "candidates_sha256": file_hash(args.candidates),
        "annotations": notes}
    args.selection_review.parent.mkdir(parents=True, exist_ok=True)
    args.selection_review.write_text(json.dumps(result, ensure_ascii=False,
                                                indent=2) + "\n")
    print(json.dumps({"approved_actions": len(notes)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "review"))
    parser.add_argument("--labels", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train120_labels_20261006.json"))
    parser.add_argument("--label-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train120_labels_reviewed_20261006.json"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train120_reviewed_20261006.json"))
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_discriminative_candidates_20261006.json"))
    parser.add_argument("--selection-review", type=Path, default=Path(
        "data/annotations/alf_sibling_discriminative_reviewed_20261006.json"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        review(args)


if __name__ == "__main__":
    main()
