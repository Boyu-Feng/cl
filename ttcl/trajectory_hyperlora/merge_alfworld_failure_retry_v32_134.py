"""Merge stopped v32 prefix with independently bound disjoint retry shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora import evaluate_alfworld_failure_retry_v32_134 as v32
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.shard_alfworld_failure_retry_v32_134 import frozen_args


def merge(args):
    if args.manifest.exists():
        raise FileExistsError(args.manifest)
    frozen = frozen_args(argparse.Namespace(device="cuda:3", gpu_fraction=.65))
    _, first, fixed, bindings = v32.checked_input(frozen)
    report = json.loads(args.report.read_text())
    if (report["review_sha256"] != file_hash(frozen.review) or
            any(report[k] != value for k, value in fixed.items()) or
            report["failures"] or
            [row["index"] for row in report["games"]] !=
                list(range(len(report["games"])))):
        raise ValueError("Changed or nonsequential existing v32 prefix")
    prefix = len(report["games"])
    replacement = {}
    provenance = []
    for path in args.shards:
        part = json.loads(path.read_text())
        if (part["review_sha256"] != file_hash(frozen.review) or
                part["original_runner_sha256"] != file_hash(Path(v32.__file__)) or
                part["shard_runner_sha256"] != file_hash(Path(
                    "ttcl/trajectory_hyperlora/shard_alfworld_failure_retry_v32_134.py")) or
                part["checkpoint_sha256"] != fixed["checkpoint_sha256"] or
                part["failures"] or
                len(part["rows"]) != part["end"] - part["start"] or
                [row["index"] for row in part["rows"]] !=
                    list(range(part["start"], part["end"]))):
            raise ValueError(f"Changed or incomplete v32 shard: {path}")
        provenance.append({"path": str(path), "sha256": file_hash(path),
                           "start": part["start"], "end": part["end"]})
        for row in part["rows"]:
            index = row["index"]
            if index in replacement:
                raise ValueError("Overlapping retry shards")
            replacement[index] = row
    for index in range(134):
        row = report["games"][index] if index < prefix else replacement.get(index)
        if row is None:
            raise ValueError(f"Missing retry target {index}")
        if (row["game"] != bindings[index]["game"] or
                row["input_content_sha256"] != bindings[index]["input_content_sha256"] or
                row["first_episode_sha256"] != v32.digest(first[index]["lora"]) or
                row["first_reward"] != first[index]["lora"]["reward"] or
                (not row["first_reward"] and tuple(row["retries"]) != v32.ARMS)):
            raise ValueError(f"Changed bound retry row {index}")
        if index in replacement and index < prefix and row != replacement[index]:
            raise ValueError(f"Overlapping retry results differ at {index}")
        if index >= prefix:
            report["games"].append(row)
    report["summary"] = {"n": 134,
        "first_wins": sum(row["first_reward"] for row in report["games"]),
        "first_failures": sum(not row["first_reward"] for row in report["games"]),
        "two_attempt_wins": {arm: sum(row["first_reward"] or
            row["retries"].get(arm, {}).get("reward", 0)
            for row in report["games"]) for arm in v32.ARMS},
        "failures": 0}
    prior_sha = file_hash(args.report)
    v32.save(args.report, report)
    manifest = {"protocol": "Stopped sequential v32 prefix plus independently bound disjoint retry shards; final report audited separately",
        "prefix": prefix, "prefix_report_sha256": prior_sha,
        "shards": provenance, "original_runner_sha256": file_hash(Path(v32.__file__)),
        "resume_runner_sha256": file_hash(Path(
            "ttcl/trajectory_hyperlora/resume_alfworld_failure_retry_v32_134.py")),
        "shard_runner_sha256": file_hash(Path(
            "ttcl/trajectory_hyperlora/shard_alfworld_failure_retry_v32_134.py")),
        "merge_runner_sha256": file_hash(Path(__file__)),
        "final_report_sha256": file_hash(args.report), "summary": report["summary"]}
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False,
                                         indent=2) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_failure_retry_v32_134_20261009.json"))
    parser.add_argument("--shards", type=Path, nargs=2, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_failure_retry_v32_merge_20261009.json"))
    merge(parser.parse_args())
