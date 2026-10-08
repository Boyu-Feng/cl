"""Create reviewed scalar-operator utility labels on official Cohort train tasks.

Only task indices 0-7 are scored here. The operators and features come from
typed_evidence_v17; no prediction field or disease rule is hand coded.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v2 import BENCH, ROOT, benchmark
from ttcl.trajectory_hyperlora.typed_evidence_v17 import extract, features, proposals


STEM = "clbench_cohort_train_collect0_7_20261008"
ROOT_EPISODES = ROOT / "results/trajectory_hyperlora" / (STEM + "_episodes") / "cohort_studies"
REPORT = ROOT / "results/trajectory_hyperlora" / (STEM + ".json")


def prepare_rows():
    from src.tasks.cohort_studies.tool_schemas import (
        build_submission_schema, parse_flat_submission,
    )
    schema_class = build_submission_schema()
    schema = schema_class.model_json_schema()
    report = json.loads(REPORT.read_text())
    public = {}
    rows = []
    task_factory = benchmark()
    for target_index in range(8):
        path = ROOT_EPISODES / f"episode_{target_index+1:03}" / "trajectory.json"
        episode = json.loads(path.read_text())
        if not episode["completed"]:
            continue
        step = episode["steps"][-1]
        current = step["action"]
        if not isinstance(current, dict) or len(current) != 108:
            raise ValueError("Changed Cohort final submission")
        task = task_factory.make_task("cohort_studies", 42, independent=True)
        query = task.reset_baseline_instance(target_index)
        official = lambda action: float(task._score_submission(
            parse_flat_submission(schema_class.model_validate(action))).score)
        baseline = official(current)
        if abs(baseline - float(episode["reward"])) > 1e-6:
            raise ValueError("Official source reward changed")
        for source_index, source_step, source_path, source_reward in public.values():
            evidence = extract(source_step["action"], source_index=source_index,
                trajectory_sha256=file_hash(source_path),
                feedback=source_step.get("public_feedback"),
                reward=source_reward, query=source_step["query"])
            for item in evidence:
                if item.kind != "number":
                    continue
                for proposal in proposals(current, item, schema):
                    if proposal.operator not in ("numeric_mean", "scalar_copy"):
                        continue
                    reward = official(proposal.action)
                    content = {"split": "train" if target_index <= 5 else "dev",
                        "target_index": target_index, "source_index": source_index,
                        "target_instance_id": query.instance_id,
                        "target_initial_query_sha256": digest(query.prompt),
                        "target_trajectory_sha256": file_hash(path),
                        "source_trajectory_sha256": file_hash(source_path),
                        "source_action_sha256": item.action_sha256,
                        "source_feedback_sha256": item.feedback_sha256,
                        "schema_path": item.path, "operator": proposal.operator,
                        "features": features(proposal, current,
                            target_index=target_index,
                            target_query=step["query"]).tolist(),
                        "base_reward": baseline, "candidate_reward": reward,
                        "utility_delta": reward-baseline}
                    rows.append({**content, "input_content_sha256": digest(content)})
        public[target_index] = (target_index, step, path, baseline)
        connection = getattr(task, "_conn", None)
        if connection is not None:
            connection.close()
    return {"protocol": "Official Cohort train 0-5 / dev 6-7 scalar counterfactuals",
        "report_sha256": file_hash(REPORT), "labels": rows,
        "input_content_sha256": digest(rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=ROOT /
        "data/annotations/clbench_typed_numeric_v19_candidates_20261008.json")
    parser.add_argument("--review", type=Path, default=ROOT /
        "data/annotations/clbench_typed_numeric_v19_review_20261008.json")
    args = parser.parse_args()
    if args.candidates.exists() or args.review.exists():
        raise FileExistsError("Use fresh candidate and review files")
    os.chdir(BENCH)
    data = prepare_rows()
    args.candidates.parent.mkdir(parents=True, exist_ok=True)
    args.candidates.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    review = {"candidates_sha256": file_hash(args.candidates),
        "annotations": [{"input_content_sha256": row["input_content_sha256"],
            "approved": False, "review_basis": "pending"}
            for row in data["labels"]]}
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"train": sum(x["split"] == "train" for x in data["labels"]),
        "dev": sum(x["split"] == "dev" for x in data["labels"]),
        "report_sha256": data["report_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
