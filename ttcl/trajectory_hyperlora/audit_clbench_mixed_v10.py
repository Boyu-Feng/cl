"""Read-only target, public-history, and outcome audit for mixed CLBench runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v2 import ROOT, benchmark
from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v3 import reward_gate
from ttcl.trajectory_hyperlora.train_clbench_cumulative_hyperlora_v4 import merge_object_arrays
from ttcl.trajectory_hyperlora.train_clbench_hyperlora_v1 import clean_records


def audit(report_path: Path, review_path: Path, actor_script: Path | None = None):
    report = json.loads(report_path.read_text())
    review = json.loads(review_path.read_text())
    binding = review["binding"]
    if (review["input_content_sha256"] != digest({
            "binding": binding, "targets": review["targets"]}) or
            report["review_sha256"] != file_hash(review_path) or
            report["checkpoint_sha256"] != binding["checkpoint_sha256"] or
            ("actor_script_sha256" in binding and
             binding["actor_script_sha256"] != file_hash(actor_script or
                 ROOT / "ttcl/trajectory_hyperlora/clbench_online_parameter_memory_v10.py"))):
        raise ValueError("Run binding or script hash changed")
    target_lookup = {(x["domain"], x["index"]): x for x in review["targets"]}
    if len(target_lookup) != len(review["targets"]):
        raise ValueError("Duplicate target binding")
    result = {}
    for domain in binding["domains"]:
        rows = [x for x in report["rows"] if x["domain"] == domain]
        if [x["index"] for x in rows] != list(range(binding["start"], binding["stop"])):
            raise ValueError("Missing or reordered domain rows")
        rewards = []
        writes = 0
        public_actions = []
        for row in rows:
            target = target_lookup[(domain, row["index"])]
            if row["target"] != target:
                raise ValueError("Reviewed target changed")
            for arm in ("base", "online"):
                directory = (report_path.parent /
                    (report_path.stem + "_episodes") / domain / arm /
                    f"episode_{row['index']+1:03}")
                outcome = row[arm]
                if json.loads((directory / "row.json").read_text()) != outcome:
                    raise ValueError("Episode row was modified")
                if (outcome["instance_id"] != target["instance_id"] or
                        outcome["initial_query_sha256"] != target["initial_query_sha256"]):
                    raise ValueError("Official task target changed")
                episode = json.loads((directory / "trajectory.json").read_text())
                if outcome["status"] == "complete":
                    if (not episode["completed"] or
                            float(episode["reward"]) != float(outcome["reward"])):
                        raise ValueError("Official outcome changed")
                elif outcome["reward"] is not None or episode["completed"]:
                    raise ValueError("Failed episode was counted")
                if arm == "online":
                    for line in (directory / "responses.jsonl").read_text().splitlines():
                        event = json.loads(line)
                        if "generated_raw_response" in event:
                            original = json.loads(event["generated_raw_response"])
                            if json.loads(event["raw_response"]) != \
                                    merge_object_arrays(original, public_actions):
                                raise ValueError("Event action used future or unbound evidence")
                    if outcome["status"] == "complete":
                        public_actions.extend(step["action"] for step in episode["steps"]
                            if isinstance(step.get("action"), dict))
            online = row["online"]
            expected_write = (online["status"] == "complete" and
                row["index"] + 1 < binding["stop"] and
                reward_gate(float(online["reward"]), rewards) and
                (binding.get("max_writes") is None or writes < binding["max_writes"]))
            if bool(row["write"]["written"]) != expected_write:
                raise ValueError("Write chronology changed")
            if expected_write:
                trajectory = json.loads((report_path.parent /
                    (report_path.stem + "_episodes") / domain / "online" /
                    f"episode_{row['index']+1:03}" / "trajectory.json").read_text())
                if row["write"]["source_records_sha256"] != digest(clean_records(trajectory)):
                    raise ValueError("Written source is not own public trajectory")
                writes += 1
            if online["status"] == "complete":
                rewards.append(float(online["reward"]))
        paired = [x for x in rows if x["base"]["status"] ==
                  x["online"]["status"] == "complete"]
        summary = report["summary"][domain]
        if (summary["paired"] != len(paired) or summary["writes"] != writes or
                summary["base_mean"] != (statistics.mean(x["base"]["reward"]
                    for x in paired) if paired else None) or
                summary["online_mean"] != (statistics.mean(x["online"]["reward"]
                    for x in paired) if paired else None)):
            raise ValueError("Summary disagrees with official episode rows")
        result[domain] = {"paired": len(paired), "writes": writes,
                          "failed_pairs": len(rows) - len(paired)}
    return {"audit": "passed", "report_sha256": file_hash(report_path),
            "review_sha256": file_hash(review_path), "domains": result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actor-script", type=Path,
                        help="Frozen actor entrypoint; defaults to v10")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.report, args.review, args.actor_script)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
