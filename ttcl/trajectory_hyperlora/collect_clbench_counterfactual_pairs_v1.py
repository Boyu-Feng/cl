"""Train-split official reward deltas for single-source trajectory LoRA.

For each fixed target, compare a previously collected independent baseline
with a new rollout using only the immediately preceding own public episode.
No held-out episode is used to form training labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v2 import (
    BENCH, ROOT, benchmark, run_episode,
)
from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v3 import TrainedActor
from ttcl.trajectory_hyperlora.online_parameter_memory_v2 import ParameterMemory
from ttcl.trajectory_hyperlora.train_clbench_hyperlora_v1 import load_episode

DOMAINS = ("blind_spectrum_monitoring", "exploitable_poker")


def targets(args):
    result = []
    base = benchmark()
    for domain in args.domains:
        for index in range(1, 8):
            prior_row, prior, _, prior_dir = load_episode(domain, index - 1)
            base_row, _, _, target_dir = load_episode(domain, index)
            task = base.make_task(domain, args.seed, independent=True)
            query = task.reset_baseline_instance(index)
            if (prior_row["status"] != "complete" or
                    base_row["status"] != "complete" or
                    base_row["instance_id"] != query.instance_id or
                    base_row["initial_query_sha256"] != digest(query.prompt)):
                raise ValueError("Train pair identity or baseline changed")
            result.append({"domain": domain, "index": index,
                "instance_id": query.instance_id,
                "initial_query_sha256": digest(query.prompt),
                "source_trajectory_sha256": file_hash(prior_dir / "trajectory.json"),
                "base_trajectory_sha256": file_hash(target_dir / "trajectory.json"),
                "base_row_sha256": file_hash(target_dir / "row.json"),
                "base_reward": base_row["reward"],
                "split": "train" if index <= 5 else "dev"})
            conn = getattr(task, "_conn", None)
            if conn is not None:
                conn.close()
    return result


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    import os
    os.chdir(BENCH)
    binding = {"checkpoint_sha256": file_hash(args.checkpoint),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "domains": args.domains, "seed": args.seed,
        "context_limit": args.context_limit, "context_tokens": args.context_tokens,
        "action_tokens": args.action_tokens, "history_turns": args.history_turns,
        "temperature": args.temperature, "top_p": args.top_p,
        "max_turns_per_instance": 64, "action_retries": 2}
    rows = targets(args)
    value = {"protocol": "Official train/development single-prior-source CLBench counterfactuals; indices 1-5 train, 6-7 dev, 8+ held out",
             "binding": binding, "targets": rows,
             "input_content_sha256": digest({"binding": binding, "targets": rows})}
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"targets": len(rows)}), flush=True)


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    import os
    os.chdir(BENCH)
    reviewed = json.loads(args.review.read_text())
    if (reviewed["binding"]["checkpoint_sha256"] != file_hash(args.checkpoint) or
            reviewed["input_content_sha256"] != digest({
                "binding": reviewed["binding"], "targets": reviewed["targets"]}) or
            reviewed["targets"] != targets(args)):
        raise ValueError("Counterfactual training input binding changed")
    actor = TrainedActor(args)
    actor.arm = "online"
    report = {"protocol": reviewed["protocol"],
              "review_sha256": file_hash(args.review),
              "checkpoint_sha256": file_hash(args.checkpoint), "rows": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for target in reviewed["targets"]:
        domain, index = target["domain"], target["index"]
        _, source, _, _ = load_episode(domain, index - 1)
        actor.memory = ParameterMemory(capacity=1)
        write = actor.encode_episode(source)
        directory = args.output.parent / (args.output.stem + "_episodes") / domain / f"episode_{index+1:03}"
        row, episode = run_episode(args, actor, domain, index, "online", directory)
        if row["instance_id"] != target["instance_id"] or row["initial_query_sha256"] != target["initial_query_sha256"]:
            raise ValueError("Counterfactual target changed")
        report["rows"].append({"target": target, "write": write,
            "online": row, "online_trajectory_sha256": file_hash(directory / "trajectory.json"),
            "reward_delta": (float(row["reward"]) - float(target["base_reward"])
                             if row["status"] == "complete" else None)})
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"domain": domain, "index": index,
            "base": target["base_reward"], "online": row["reward"],
            "delta": report["rows"][-1]["reward_delta"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=Path, default=ROOT / "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results/trajectory_hyperlora/clbench_hyperlora_v2_20261008.pt")
    parser.add_argument("--review", type=Path, default=ROOT / "data/annotations/clbench_hyperlora_v2_counterfactual_reviewed_20261008.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/trajectory_hyperlora/clbench_hyperlora_v2_counterfactual_20261008.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.85)
    parser.add_argument("--action-tokens", type=int, default=512)
    parser.add_argument("--context-limit", type=int, default=16384)
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--history-turns", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=.7)
    parser.add_argument("--top-p", type=float, default=.9)
    args = parser.parse_args()
    args.episodes = 8
    (prepare if args.command == "prepare" else evaluate)(args)


if __name__ == "__main__":
    main()
