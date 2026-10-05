"""Conditional live diagnostic: supply an independent reviewed success history.

This tests the hypernetwork after a positive trajectory has been obtained. It
does not measure empty-history exploration or train on held-out task labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.evaluate_xland_official_live import (
    Worker, episode, load_actor, sha256,
)
from ttcl.trajectory_hyperlora.train_xland_official_history_hyperlora import source_tensor
from ttcl.trajectory_hyperlora.train_xland_raw_hyperlora import checked_query


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    annotations = json.loads(args.annotations.read_text())
    items = annotations["split"][args.split][:args.limit]
    if args.positive_source_only:
        items = [item for item in items
                 if item.get("online_source_positive", 0) > 0]
    if len(items) < 2:
        raise ValueError("At least two distinct reviewed tasks required")
    torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                               device=args.device)
    agent, tokenizer, choices = load_actor(args)
    worker = Worker(args.xland_python, args.data_dir, args.benchmark,
                    args.environment, args.benchmark_path)
    result = {"protocol": "Conditional official live test with independently reviewed positive source history; own versus wrong-task versus no LoRA; no empty-history exploration",
              "annotations_sha256": sha256(args.annotations),
              "checkpoint_sha256": sha256(args.checkpoint),
              "benchmark_sha256": sha256(args.benchmark_path),
              "split": args.split, "seed": args.seed,
              "target_sampling_seed": args.target_sampling_seed,
              "target_epsilon": args.target_epsilon,
              "positive_source_only": args.positive_source_only,
              "steps": args.steps,
              "target_feedback_window": args.target_feedback_window,
              "rows": [], "failures": []}
    try:
        with torch.no_grad():
            factors = []
            source_positive = []
            for item in items:
                content, _ = checked_query(item["queries"][0])
                source_positive.append(sum(step["reward"] > 0
                    for episode_data in content["source_episodes"]
                    for step in episode_data["steps"]))
                factors.append(agent.compile_adapters(
                    source_tensor(content, args.device,
                                  agent.encoder.max_source_length)))
            for index, item in enumerate(items):
                try:
                    rid = item["ruleset_id"]
                    seed = args.seed + rid
                    rollouts = {}
                    for arm, adapter in (("own", factors[index]),
                                         ("wrong", factors[(index + 1) % len(items)]),
                                         ("none", None)):
                        rollouts[arm] = episode(worker, agent, tokenizer,
                            choices, rid, seed, args.steps, adapter,
                            args.device,
                            sample_seed=args.target_sampling_seed + rid,
                            epsilon=args.target_epsilon,
                            feedback_window=args.target_feedback_window)
                    result["rows"].append({"task_id": item["task_id"],
                        "ruleset_id": rid,
                        "source_positive": source_positive[index],
                        "wrong_ruleset_id": items[(index + 1) % len(items)]["ruleset_id"],
                        "rollouts": rollouts})
                except Exception as exc:
                    result["failures"].append({"task_id": item["task_id"],
                        "error": f"{type(exc).__name__}: {exc}"})
                rows = result["rows"]
                result["summary"] = {"n": len(rows),
                    "source_positive": sum(row["source_positive"] > 0
                        for row in rows),
                    "own_vs_wrong_first_action_changes": sum(
                        row["rollouts"]["own"]["steps"][0]["action"] !=
                        row["rollouts"]["wrong"]["steps"][0]["action"]
                        for row in rows),
                    "own_vs_none_first_action_changes": sum(
                        row["rollouts"]["own"]["steps"][0]["action"] !=
                        row["rollouts"]["none"]["steps"][0]["action"]
                        for row in rows),
                    **{arm + "_positive": sum(row["rollouts"][arm][
                        "positive_rewards"] > 0 for row in rows)
                        for arm in ("own", "wrong", "none")}}
                args.output.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.output.with_suffix(args.output.suffix + ".tmp")
                temporary.write_text(json.dumps(result, indent=2) + "\n")
                temporary.replace(args.output)
                print(json.dumps(result["summary"]), flush=True)
    finally:
        worker.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--xland-python", type=Path, default=Path(
        "ttcl/.runtime/xland_env/bin/python"))
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/xland_minigrid"))
    parser.add_argument("--benchmark", default="trivial-1m")
    parser.add_argument("--benchmark-path", type=Path, required=True)
    parser.add_argument("--environment", default="XLand-MiniGrid-R1-9x9")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.7)
    parser.add_argument("--seed", type=int, default=71005)
    parser.add_argument("--target-sampling-seed", type=int, default=91005)
    parser.add_argument("--target-epsilon", type=float, default=0.)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--target-feedback-window", type=int, default=0)
    parser.add_argument("--limit", type=int, default=13)
    parser.add_argument("--positive-source-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
