"""Full CLBench frozen-actor baseline: every instance starts without memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.clbench_general_evidence_v22 import targets
from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v2 import (
    Actor, DOMAINS, ROOT, run_episode,
)


def binding(args):
    return {"method": "independent_frozen_base_v22",
        "runner_sha256": file_hash(Path(__file__)),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "domain": args.domain, "seed": args.seed,
        "action_tokens": args.action_tokens,
        "context_limit": args.context_limit,
        "context_tokens": args.context_tokens,
        "history_turns": args.history_turns,
        "temperature": args.temperature, "top_p": args.top_p,
        "max_turns_per_instance": 64, "action_retries": 2}


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    value = {"protocol": "Every official instance independently starts with the "
        "same frozen actor, no active LoRA, no evidence, no history writes",
        "binding": binding(args), "targets": targets(args)}
    value["input_content_sha256"] = digest({"binding": value["binding"],
                                            "targets": value["targets"]})
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"domain": args.domain, "targets": len(value["targets"])}),
          flush=True)


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    review = json.loads(args.review.read_text())
    if (review["binding"] != binding(args) or
            review["input_content_sha256"] != digest({
                "binding": review["binding"], "targets": review["targets"]}) or
            review["targets"] != targets(args)):
        raise ValueError("Changed reviewed targets or frozen actor settings")
    args.episodes = len(review["targets"])
    actor = Actor(args)
    actor.arm = "base"
    report = {"protocol": review["protocol"],
        "review_sha256": file_hash(args.review), "binding": review["binding"],
        "rows": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for target in review["targets"]:
        index = target["index"]
        folder = args.output.parent / (args.output.stem + "_episodes") / f"episode_{index+1:03}"
        row, _ = run_episode(args, actor, args.domain, index, "base", folder)
        if (row["instance_id"] != target["instance_id"] or
                row["initial_query_sha256"] != target["initial_query_sha256"] or
                row["adapter_enabled"] or row["read"]["entries"]):
            raise ValueError("Changed target or activated memory")
        report["rows"].append({"index": index, "target": target, "base": row})
        if row["status"] != "complete":
            report["failures"].append({"index": index, "row": row})
        good = [x["base"]["reward"] for x in report["rows"]
                if x["base"]["status"] == "complete"]
        report["summary"] = {"completed_targets": len(report["rows"]),
                             "scored": len(good),
                             "base_mean": statistics.mean(good) if good else None}
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"domain": args.domain, "index": index,
                          "base": row["reward"],
                          "failures": len(report["failures"])}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--model", type=Path, default=ROOT /
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--checkpoint", type=Path, default=ROOT /
        "results/trajectory_hyperlora/clbench_hyperlora_v2_20261008.pt")
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.80)
    parser.add_argument("--action-tokens", type=int, default=512)
    parser.add_argument("--context-limit", type=int, default=16384)
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--history-turns", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=.7)
    parser.add_argument("--top-p", type=float, default=.9)
    args = parser.parse_args()
    args.event_only = False
    args.disable_events = False
    for name in ("model", "checkpoint", "review", "output"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.command == "evaluate" and args.output is None:
        parser.error("evaluate requires --output")
    if args.domain == "cohort_studies" and \
            (args.action_tokens < 8192 or args.context_limit < 32768):
        parser.error("Cohort requires its reviewed long-output budget")
    if args.command == "prepare":
        prepare(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
