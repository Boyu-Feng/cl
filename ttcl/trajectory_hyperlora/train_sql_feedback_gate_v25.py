"""Bind and train the SQL feedback operator from earlier public trajectories.

Indices 0-12 train, 13-17 develop. Indices 18-29 are not used for fitting.
The target is immediate query feedback; final official reward is a separate
online evaluation. Source and target action/feedback content are hash-bound.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v2 import ROOT
from ttcl.trajectory_hyperlora.general_evidence_v22 import (
    digest, extract_records, verify,
)
from ttcl.trajectory_hyperlora.sql_feedback_evidence_v25 import (
    candidate, feedback_grade, fit, missing_identifiers,
)


SOURCE_REPORT = ROOT / "results/trajectory_hyperlora/clbench_general_evidence_v22_database_exploration_full_20261009.json"
EPISODES = ROOT / "results/trajectory_hyperlora/clbench_general_evidence_v22_database_exploration_full_20261009_episodes/lora"


def labels() -> dict:
    report = json.loads(SOURCE_REPORT.read_text())
    if len(report["rows"]) != 30:
        raise ValueError("Expected completed source sequence")
    prior = []
    bad = set()
    rows = []
    for index, outcome in enumerate(report["rows"]):
        path = EPISODES / f"episode_{index+1:03d}" / "trajectory.json"
        episode = json.loads(path.read_text())
        if (not episode["completed"] or
                float(episode["reward"]) != float(outcome["lora"]["reward"])):
            raise ValueError("Changed official source outcome")
        records = [{"observation": step["query"], "action": step["action"],
                    "feedback": step.get("public_feedback", "")}
                   for step in episode["steps"]]
        for step_index, record in enumerate(records):
            selected = candidate(record["action"], prior, index, bad)
            if selected is None or index >= 18:
                continue
            source = selected.source
            source_path = EPISODES / f"episode_{source.source_index+1:03d}" / "trajectory.json"
            content = {"split": "train" if index <= 12 else "dev",
                "target_index": index, "target_step_index": step_index,
                "target_instance_id": outcome["target"]["instance_id"],
                "target_initial_query_sha256": outcome["target"]["initial_query_sha256"],
                "target_trajectory_sha256": file_hash(path),
                "target_action_sha256": digest(record["action"]),
                "target_feedback_sha256": digest(record["feedback"]),
                "source_index": source.source_index,
                "source_step_index": source.step_index,
                "source_trajectory_sha256": file_hash(source_path),
                "source_action_sha256": source.action_sha256,
                "source_feedback_sha256": source.feedback_sha256,
                "source_record_sha256": source.source_sha256,
                "failed_identifier": selected.failed_identifier,
                "candidate_action_sha256": digest(selected.action),
                "operator": "sql_feedback_schema_query_copy",
                "feedback_delta": feedback_grade(source.feedback) -
                    feedback_grade(record["feedback"])}
            if not verify(source, [{"observation": s["query"],
                    "action": s["action"], "feedback": s.get("public_feedback", "")}
                    for s in json.loads(source_path.read_text())["steps"]],
                    source.source_sha256):
                raise ValueError("Source provenance failed")
            rows.append({**content, "input_content_sha256": digest(content)})
        added = extract_records(records, source_index=index,
            source_sha256=digest(records), reward=float(episode["reward"]))
        prior.extend(added)
        for item in added:
            bad.update(missing_identifiers(item.feedback))
    return {"protocol": "CLBench SQL feedback operator; train indices 0-12, dev 13-17; immediate public feedback labels, not official final reward",
        "source_report_sha256": file_hash(SOURCE_REPORT),
        "operator_sha256": file_hash(Path(__file__).with_name("sql_feedback_evidence_v25.py")),
        "labels": rows, "input_content_sha256": digest(rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "train"))
    parser.add_argument("--candidates", type=Path, default=ROOT /
        "data/annotations/clbench_sql_feedback_v25_candidates_20261009.json")
    parser.add_argument("--review", type=Path, default=ROOT /
        "data/annotations/clbench_sql_feedback_v25_review_20261009.json")
    parser.add_argument("--output", type=Path, default=ROOT /
        "results/trajectory_hyperlora/clbench_sql_feedback_gate_v25_20261009.json")
    args = parser.parse_args()
    data = labels()
    if args.command == "prepare":
        if args.candidates.exists() or args.review.exists():
            raise FileExistsError("Use new reviewed annotation paths")
        args.candidates.parent.mkdir(parents=True, exist_ok=True)
        args.candidates.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        review = {"candidates_sha256": file_hash(args.candidates),
            "annotations": [{"input_content_sha256": row["input_content_sha256"],
                "approved": False, "review_basis": "pending"}
                for row in data["labels"]]}
        args.review.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"labels": len(data["labels"]),
            "train": sum(r["split"] == "train" for r in data["labels"]),
            "dev": sum(r["split"] == "dev" for r in data["labels"])}))
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    saved = json.loads(args.candidates.read_text())
    review = json.loads(args.review.read_text())
    if saved != data or review["candidates_sha256"] != file_hash(args.candidates):
        raise ValueError("Changed source trajectory, operator, or reviewed labels")
    approvals = {row["input_content_sha256"]: row for row in review["annotations"]}
    if len(approvals) != len(data["labels"]) or any(
            not approvals[row["input_content_sha256"]]["approved"] or
            approvals[row["input_content_sha256"]]["review_basis"] == "pending"
            for row in data["labels"]):
        raise ValueError("Incomplete label review")
    model = fit(data["labels"])
    model.update({"candidates_sha256": file_hash(args.candidates),
        "review_sha256": file_hash(args.review),
        "source_report_sha256": data["source_report_sha256"],
        "operator_sha256": data["operator_sha256"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(model))


if __name__ == "__main__":
    main()
