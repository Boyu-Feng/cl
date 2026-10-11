"""Merge frozen disjoint v51 shards and replay every selected environment episode."""

from __future__ import annotations

import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.evaluate_alfworld_full_lineage_static_v51 import audit, checked, parser


def main():
    p = parser()
    p.add_argument("--primary", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_full_lineage_static_v51_unseen134_20261010.json"))
    p.add_argument("--shard-70-102", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_full_lineage_static_v51_shard_70_102_20261010.json"))
    p.add_argument("--shard-102-134", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_full_lineage_static_v51_shard_102_134_20261010.json"))
    args = p.parse_args()
    if args.command != "audit" or args.output.exists() or args.audit_output.exists():
        raise ValueError("Use fresh merged output and audit paths")
    old, fixed, targets = checked(args)
    primary = json.loads(args.primary.read_text())
    shard_paths = [args.shard_70_102, args.shard_102_134]
    shards = [json.loads(path.read_text()) for path in shard_paths]
    for key, value in fixed.items():
        if primary.get(key) != value or any(shard.get(key) != value for shard in shards):
            raise ValueError(f"Changed frozen v51 field {key}")
    review_hash = file_hash(args.review)
    if primary.get("review_sha256") != review_hash or any(
            shard.get("review_sha256") != review_hash for shard in shards):
        raise ValueError("Changed reviewed target list")
    if len(primary.get("rows", [])) < 70 or primary.get("failures"):
        raise ValueError("Incomplete primary prefix")
    shard_runner_hash = file_hash(Path(__file__).with_name(
        "evaluate_alfworld_full_lineage_static_v51_shard.py"))
    for shard, (start, end) in zip(shards, ((70, 102), (102, 134)), strict=True):
        if (shard.get("start") != start or shard.get("end") != end or
                len(shard.get("rows", [])) != end - start or shard.get("failures") or
                shard.get("shard_runner_sha256") != shard_runner_hash):
            raise ValueError("Changed or incomplete shard")
    rows = primary["rows"][:70] + shards[0]["rows"] + shards[1]["rows"]
    if len(rows) != 134:
        raise ValueError("Missing v51 target")
    for i, (target, row, old_row) in enumerate(zip(
            targets, rows, old["rows"], strict=True)):
        if (row["index"] != i or row["game"] != target["game"] or
                row["input_content_sha256"] != target["input_content_sha256"] or
                row["old_hyper_reward"] != old_row["old_hyper_reward"] or
                row["static"]["status"] != "complete"):
            raise ValueError(f"Changed selected v51 row {i}")
    report = {"protocol": "v51 full-lineage static vs old H; first70 primary episodes plus two frozen disjoint shards",
              "review_sha256": review_hash, **fixed,
              "merge_runner_sha256": file_hash(Path(__file__)),
              "source_reports": [
                  {"path": str(args.primary), "sha256": file_hash(args.primary),
                   "selected_start": 0, "selected_end": 70,
                   "runner_sha256": fixed["runner_sha256"]},
                  *[{"path": str(path), "sha256": file_hash(path),
                     "selected_start": start, "selected_end": end,
                     "runner_sha256": shard_runner_hash}
                    for path, (start, end) in zip(shard_paths, ((70, 102), (102, 134)), strict=True)]],
              "rows": rows, "failures": [],
              "summary": {"n": 134,
                  "old_hyper": sum(row["old_hyper_reward"] for row in rows),
                  "static": sum(row["static"]["reward"] for row in rows),
                  "failures": 0}}
    save(args.output, report)
    checked_result = audit(args)
    save(args.audit_output, checked_result)
    print(json.dumps(checked_result), flush=True)


if __name__ == "__main__":
    main()
