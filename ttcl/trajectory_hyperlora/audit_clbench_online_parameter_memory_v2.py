"""Read-only online-lineage audit for official CLBench transfer pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v2 import ROOT, checked, public_records
from ttcl.trajectory_hyperlora.online_parameter_memory_v2 import ParameterMemory


def audit(args):
    if args.audit_output.exists():
        raise FileExistsError(args.audit_output)
    review = checked(args)
    report = json.loads(args.output.read_text())
    if (report["review_sha256"] != file_hash(args.review) or
            report["checkpoint_sha256"] != file_hash(args.checkpoint) or
            report["domains"] != args.domains or
            report["episodes_per_domain"] != args.episodes or
            len(report["rows"]) != len(review["targets"])):
        raise ValueError("Changed CLBench transfer result")
    summary = {}
    failures = []
    index_in_report = 0
    for domain in args.domains:
        state_hash = ParameterMemory().digest()
        writes = 0
        paired = []
        for index in range(args.episodes):
            entry = report["rows"][index_in_report]
            target = review["targets"][index_in_report]
            index_in_report += 1
            if (entry["domain"] != domain or entry["index"] != index or
                    entry["target"] != target or
                    entry["memory_sha256_before"] != state_hash or
                    entry["base"]["instance_id"] != target["instance_id"] or
                    entry["online"]["instance_id"] != target["instance_id"] or
                    entry["base"]["initial_query_sha256"] != target["initial_query_sha256"] or
                    entry["online"]["initial_query_sha256"] != target["initial_query_sha256"]):
                raise ValueError("Target or parameter-memory chronology changed")
            if index == 0 and (entry["base"]["first_prompt_sha256"] !=
                               entry["online"]["first_prompt_sha256"] or
                               entry["online"]["adapter_enabled"]):
                raise ValueError("First CLBench target was not an empty-memory pair")
            if (entry["base"]["adapter_enabled"] or
                    entry["online"]["read"]["entries"] != writes):
                raise ValueError("Incorrect adapter mounting or memory count")
            directory = args.output.parent / (args.output.stem + "_episodes") / domain
            episodes = {}
            for arm in ("base", "online"):
                row = entry[arm]
                episode = json.loads((directory / arm /
                    f"episode_{index+1:03}" / "trajectory.json").read_text())
                saved = json.loads((directory / arm /
                    f"episode_{index+1:03}" / "row.json").read_text())
                if (saved != row or row["reward"] != episode["reward"] or
                        row["success"] != episode["success"] or
                        row["initial_query_sha256"] != digest(episode["initial_public_query"])):
                    raise ValueError("Actor episode or official outcome record changed")
                episodes[arm] = episode
            should_write = (entry["online"]["status"] == "complete" and
                            entry["online"]["success"] is True and
                            index + 1 < args.episodes and
                            bool(public_records(episodes["online"])))
            if (entry["write"]["written"] != should_write or
                    (should_write and (entry["write"]["source_records_sha256"] !=
                        digest(public_records(episodes["online"])) or
                        entry["write"]["source_tokens_retained"] > args.context_tokens)) or
                    (not should_write and entry["memory_sha256_after"] != state_hash)):
                raise ValueError("Write reward, source or parameter state changed")
            state_hash = entry["memory_sha256_after"]
            writes += int(should_write)
            if entry["base"]["status"] == entry["online"]["status"] == "complete":
                paired.append(entry)
            else:
                failures.append({"domain": domain, "index": index,
                                 "base_status": entry["base"]["status"],
                                 "online_status": entry["online"]["status"]})
        summary[domain] = {"paired": len(paired),
            "base_mean": statistics.mean(x["base"]["reward"] for x in paired) if paired else None,
            "online_mean": statistics.mean(x["online"]["reward"] for x in paired) if paired else None,
            "base_successes": sum(x["base"]["success"] for x in paired),
            "online_successes": sum(x["online"]["success"] for x in paired),
            "writes": sum(x["write"]["written"] for x in paired)}
    if report["summary"] != summary:
        raise ValueError("CLBench summary changed")
    value = {"protocol": "Read-only official CLBench target/JSON action record and online write chronology audit; no reward replay",
        "raw_report_sha256": file_hash(args.output),
        "review_sha256": file_hash(args.review),
        "summary": summary, "incomplete_pairs": failures}
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"summary": summary, "incomplete_pairs": failures}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT /
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--checkpoint", type=Path, default=ROOT /
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt")
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-tokens", type=int, default=512)
    parser.add_argument("--context-limit", type=int, default=8192)
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=.7)
    parser.add_argument("--top-p", type=float, default=.9)
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--read-temperature", type=float, default=.05)
    parser.add_argument("--attention-mix", type=float, default=.75)
    parser.add_argument("--history-turns", type=int)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
