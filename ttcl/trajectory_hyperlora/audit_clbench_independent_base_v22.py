"""Audit frozen independent CLBench outcomes and absence of memory use."""

import argparse
import json
import math
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash


def audit(args):
    report = json.loads(args.report.read_text())
    review = json.loads(args.review.read_text())
    binding = review["binding"]
    if (binding["method"] != "independent_frozen_base_v22" or
            binding["runner_sha256"] != file_hash(args.runner) or
            review["input_content_sha256"] != digest({
                "binding": binding, "targets": review["targets"]}) or
            report["review_sha256"] != file_hash(args.review) or
            report["binding"] != binding or
            len(report["rows"]) != len(review["targets"])):
        raise ValueError("Changed frozen baseline review or coverage")
    scored = []
    failures = []
    for row, target in zip(report["rows"], review["targets"]):
        index = target["index"]
        folder = (args.report.parent / (args.report.stem + "_episodes") /
                  f"episode_{index+1:03}")
        outcome = json.loads((folder / "row.json").read_text())
        episode = json.loads((folder / "trajectory.json").read_text())
        response_path = folder / "responses.jsonl"
        if not response_path.exists() and outcome["actor_calls"]:
            raise ValueError("Missing actor response log")
        responses = ([json.loads(line) for line in
            response_path.read_text().splitlines()]
            if response_path.exists() else [])
        if len(responses) != outcome["actor_calls"]:
            raise ValueError("Actor response count changed")
        if (row != {"index": index, "target": target, "base": outcome} or
                outcome["instance_id"] != target["instance_id"] or
                outcome["initial_query_sha256"] != target["initial_query_sha256"] or
                outcome["adapter_enabled"] or outcome["read"]["entries"] or
                any(x["adapter_enabled"] for x in responses)):
            raise ValueError("Changed episode, target, or memory usage")
        if outcome["status"] == "complete":
            if not episode["completed"] or episode["reward"] != outcome["reward"]:
                raise ValueError("Changed official outcome")
            scored.append(outcome["reward"])
        else:
            if outcome["reward"] is not None or episode["completed"]:
                raise ValueError("Failed episode has score")
            failures.append(index)
    summary = report["summary"]
    mean = sum(scored) / len(scored) if scored else None
    if (summary["completed_targets"] != len(report["rows"]) or
            summary["scored"] != len(scored) or
            (mean is not None and not math.isclose(summary["base_mean"], mean,
                                                   abs_tol=1e-12, rel_tol=0)) or
            len(report["failures"]) != len(failures)):
        raise ValueError("Summary or failures changed")
    return {"audit": "passed", "report_sha256": file_hash(args.report),
            "review_sha256": file_hash(args.review), "scored": len(scored),
            "failures": len(failures), "base_mean": mean}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--runner", type=Path, default=Path(
        "ttcl/trajectory_hyperlora/clbench_independent_base_v22.py"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    value = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    main()
