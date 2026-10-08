"""Read-only content and chronology audit for the CLBench v1 held-out run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v2 import ROOT
from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v3 import reward_gate


def audit(report_path: Path, review_path: Path):
    report = json.loads(report_path.read_text())
    review = json.loads(review_path.read_text())
    binding = review["binding"]
    if binding["start"] < 8 or binding["stop"] <= binding["start"]:
        raise ValueError("Training/test index separation changed")
    if review["input_content_sha256"] != digest({"binding": binding,
                                                   "targets": review["targets"]}):
        raise ValueError("Target content binding changed")
    if report["review_sha256"] != file_hash(review_path) or \
       report["checkpoint_sha256"] != binding["checkpoint_sha256"]:
        raise ValueError("Evaluation lineage mismatch")
    checked = []
    for domain in binding["domains"]:
        rewards = []
        entries = 0
        domain_rows = [x for x in report["rows"] if x["domain"] == domain]
        for row in domain_rows:
            index = row["index"]
            target = next(x for x in review["targets"] if x["domain"] == domain and
                          x["index"] == index)
            if target != row["target"] or index < binding["start"] or index >= binding["stop"]:
                raise ValueError("Unreviewed held-out target")
            if (row["base"]["instance_id"] != target["instance_id"] or
                    row["online"]["instance_id"] != target["instance_id"] or
                    row["base"]["initial_query_sha256"] != target["initial_query_sha256"] or
                    row["online"]["initial_query_sha256"] != target["initial_query_sha256"]):
                raise ValueError("Paired target identity mismatch")
            if bool(row["online"]["adapter_enabled"]) != (entries > 0):
                raise ValueError("Adapter read chronology mismatch")
            reward = row["online"]["reward"]
            expected = row["online"]["status"] == "complete" and \
                index + 1 < binding["stop"] and reward_gate(float(reward), rewards) and \
                (binding.get("max_writes") is None or entries < binding["max_writes"])
            if bool(row["write"]["written"]) != expected:
                raise ValueError("Reward-gated write chronology mismatch")
            if expected:
                episode = report_path.parent / (report_path.stem + "_episodes") / domain / "online" / f"episode_{index+1:03}" / "trajectory.json"
                from ttcl.trajectory_hyperlora.train_clbench_hyperlora_v1 import clean_records
                if row["write"]["source_records_sha256"] != digest(clean_records(json.loads(episode.read_text()))):
                    raise ValueError("Written public source changed")
                entries += 1
            if row["online"]["status"] == "complete":
                rewards.append(float(reward))
        if len(domain_rows) != binding["stop"] - binding["start"]:
            raise ValueError("Missing held-out domain rows")
        paired = [x for x in domain_rows if
                  x["base"]["status"] == x["online"]["status"] == "complete"]
        summary = report["summary"][domain]
        if (summary["paired"] != len(paired) or summary["writes"] != entries or
            summary["base_mean"] != (statistics.mean(x["base"]["reward"] for x in paired) if paired else None) or
            summary["online_mean"] != (statistics.mean(x["online"]["reward"] for x in paired) if paired else None)):
            raise ValueError("Official summary inconsistent with rows")
        checked.append({"domain": domain, "paired": len(paired), "writes": entries})
    return {"audit": "passed", "report_sha256": file_hash(report_path),
            "review_sha256": file_hash(review_path), "domains": checked}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=ROOT / "results/trajectory_hyperlora/clbench_hyperlora_v1_test_20261008.json")
    parser.add_argument("--review", type=Path, default=ROOT / "data/annotations/clbench_hyperlora_v1_test_20261008.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/trajectory_hyperlora/clbench_hyperlora_v1_test_audit_20261008.json")
    args = parser.parse_args()
    result = audit(args.report, args.review)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
