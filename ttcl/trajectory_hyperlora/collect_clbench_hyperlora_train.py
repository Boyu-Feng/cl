"""Collect independent official CLBench training episodes for hyper-LoRA.

This collector records only public trajectories and official outcomes. Target
indices and all budgets are fixed before the actor is loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v2 import (
    Actor, BENCH, DOMAINS, ROOT, benchmark, run_episode,
)
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS[:3]))
    parser.add_argument("--start", type=int, default=4)
    parser.add_argument("--stop", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=Path, default=ROOT / "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.85)
    parser.add_argument("--context-limit", type=int, default=16384)
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--action-tokens", type=int, default=512)
    parser.add_argument("--history-turns", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=.7)
    parser.add_argument("--top-p", type=float, default=.9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.episodes = args.stop
    if args.start < 0 or args.stop <= args.start or args.output.exists():
        parser.error("Use fresh output and an increasing nonnegative index range")
    import os
    os.chdir(BENCH)
    base = benchmark()
    targets = []
    for domain in args.domains:
        for index in range(args.start, args.stop):
            task = base.make_task(domain, args.seed, independent=True)
            query = task.reset_baseline_instance(index)
            targets.append({"domain": domain, "index": index,
                            "instance_id": query.instance_id,
                            "initial_query_sha256": digest(query.prompt)})
            connection = getattr(task, "_conn", None)
            if connection is not None:
                connection.close()
    binding = {"domains": args.domains, "start": args.start, "stop": args.stop,
               "seed": args.seed, "model_config_sha256": file_hash(args.model / "config.json"),
               "checkpoint_sha256": file_hash(args.checkpoint),
               "context_limit": args.context_limit, "context_tokens": args.context_tokens,
               "action_tokens": args.action_tokens, "history_turns": args.history_turns,
               "temperature": args.temperature, "top_p": args.top_p,
               "max_turns_per_instance": 64, "action_retries": 2}
    report = {"protocol": "Independent official CLBench training collection; no LoRA; fixed ordered indices",
              "binding": binding, "targets": targets, "input_content_sha256": digest({"binding": binding, "targets": targets}),
              "rows": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    actor = Actor(args)
    actor.arm = "base"
    for target in targets:
        domain, index = target["domain"], target["index"]
        output = args.output.parent / (args.output.stem + "_episodes") / domain / f"episode_{index+1:03}"
        row, _ = run_episode(args, actor, domain, index, "base", output)
        if row["instance_id"] != target["instance_id"] or row["initial_query_sha256"] != target["initial_query_sha256"]:
            raise ValueError("CLBench training target changed")
        report["rows"].append(row)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"domain": domain, "index": index,
                          "status": row["status"], "reward": row["reward"]}), flush=True)


if __name__ == "__main__":
    main()
