"""Read-only lineage and outcome audit for the Cohort cold-start run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v2 import ROOT
from ttcl.trajectory_hyperlora.train_clbench_hyperlora_v1 import clean_records


def audit(report_path: Path, review_path: Path):
    report = json.loads(report_path.read_text())
    review = json.loads(review_path.read_text())
    binding = review["binding"]
    if (review["input_content_sha256"] != digest({
            "binding": binding, "targets": review["targets"]}) or
            report["review_sha256"] != file_hash(review_path) or
            report["checkpoint_sha256"] != binding["checkpoint_sha256"] or
            binding["actor_script_sha256"] != file_hash(
                ROOT / "ttcl/trajectory_hyperlora/clbench_online_parameter_memory_v9.py")):
        raise ValueError("Review or actor binding changed")
    targets = {(x["domain"], x["index"]): x for x in review["targets"]}
    if len(targets) != len(review["targets"]):
        raise ValueError("Duplicate target")
    outcome = {}
    for domain in binding["domains"]:
        rows = [x for x in report["rows"] if x["domain"] == domain]
        if [x["index"] for x in rows] != list(range(binding["start"], binding["stop"])):
            raise ValueError("Missing or reordered task")
        written = 0
        rewards = []
        written_records = []
        for row in rows:
            target = targets[(domain, row["index"])]
            if row["target"] != target:
                raise ValueError("Task target changed")
            for arm in ("base", "online"):
                directory = (report_path.parent /
                    (report_path.stem + "_episodes") / domain / arm /
                    f"episode_{row['index']+1:03}")
                item = row[arm]
                episode = json.loads((directory / "trajectory.json").read_text())
                if json.loads((directory / "row.json").read_text()) != item:
                    raise ValueError("Episode row changed")
                if (item["instance_id"] != target["instance_id"] or
                        item["initial_query_sha256"] != target["initial_query_sha256"]):
                    raise ValueError("Official task input changed")
                if item["status"] == "complete":
                    if not episode["completed"] or float(episode["reward"]) != float(item["reward"]):
                        raise ValueError("Official reward changed")
                elif item["reward"] is not None or episode["completed"]:
                    raise ValueError("Failure was assigned reward")
            online = row["online"]
            should_write = (online["status"] == "complete" and
                row["index"] + 1 < binding["stop"] and
                (written == 0 or float(online["reward"]) > 0) and
                (binding.get("max_writes") is None or written < binding["max_writes"]))
            if bool(row["write"]["written"]) != should_write:
                raise ValueError("Cold-start write chronology changed")
            if should_write:
                directory = (report_path.parent /
                    (report_path.stem + "_episodes") / domain / "online" /
                    f"episode_{row['index']+1:03}")
                episode = json.loads((directory / "trajectory.json").read_text())
                records = clean_records(episode)
                if not records:
                    raise ValueError("Written episode has no public action")
                written_records.extend(records)
                if row["write"]["source_records_sha256"] != digest(written_records):
                    raise ValueError("Written source is not the audited public history")
                written += 1
            if online["status"] == "complete":
                rewards.append(float(online["reward"]))
        paired = [x for x in rows if x["base"]["status"] ==
                  x["online"]["status"] == "complete"]
        summary = report["summary"][domain]
        if (summary["paired"] != len(paired) or summary["writes"] !=
                sum(x["write"]["written"] for x in paired) or
                summary["base_mean"] != (statistics.mean(x["base"]["reward"]
                    for x in paired) if paired else None) or
                summary["online_mean"] != (statistics.mean(x["online"]["reward"]
                    for x in paired) if paired else None)):
            raise ValueError("Summary changed")
        outcome[domain] = {"paired": len(paired),
                           "failed_pairs": len(rows) - len(paired),
                           "writes": written}
    return {"audit": "passed", "report_sha256": file_hash(report_path),
            "review_sha256": file_hash(review_path), "domains": outcome}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.report, args.review)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
