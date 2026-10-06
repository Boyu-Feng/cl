"""Audit v5 probe/final separation and reward-only adapter selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.audit_xland_official_reward_v4 import audit_episode
from ttcl.trajectory_hyperlora.evaluate_xland_official_live import digest, sha256


ARMS = ("none", "all_history", "reward_gated")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = json.loads(args.result.read_text())
    source = json.loads(args.source_result.read_text())
    audit = json.loads(args.source_audit.read_text())
    if (result["source_result_sha256"] != sha256(args.source_result) or
            result["source_audit_sha256"] != sha256(args.source_audit) or
            audit["result_sha256"] != sha256(args.source_result) or
            not audit["passed"] or result["rule_ids"] != source["rule_ids"] or
            result["failures"] or
            len(result["rows"]) != len(source["rows"])):
        raise ValueError("Source lineage or result completion changed")
    for row, source_row in zip(result["rows"], source["rows"], strict=True):
        if (row["ruleset_id"] != source_row["ruleset_id"] or
                row["source_sha256"] != source_row["source_sha256"] or
                row["source_positive"] != source_row["source_positive_so_far"]):
            raise ValueError("Source record changed")
        if row["probe"]["reset_seed"] == row["final"]["reset_seed"]:
            raise ValueError("Probe and final used same reset")
        for stage in ("probe", "final"):
            block = row[stage]
            initial = block["arms"]["none"]["steps"][0]["state"]
            if block["target_initial_sha256"] != digest(initial):
                raise ValueError("Initial observation hash changed")
            for arm in ARMS:
                ep = block["arms"][arm]
                audit_episode(ep, result["budget"])
                if ep["steps"][0]["state"] != initial:
                    raise ValueError("Arms used different reset")
            if row["source_positive"] == 0 and \
                    block["reward_gated_input_sha256"] is not None:
                raise ValueError("Reward gate used unsuccessful source")
        selected = max(ARMS, key=lambda arm:
                       row["probe"]["arms"][arm]["return"])
        if row["selected_arm"] != selected:
            raise ValueError("Selected arm used final reward or wrong tie order")
    record = {"result_sha256": sha256(args.result),
              "source_result_sha256": sha256(args.source_result),
              "rows": len(result["rows"]), "passed": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
