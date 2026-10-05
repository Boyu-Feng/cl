"""Collect model-chosen, empty-history XLand source episodes by frozen split.

The generated sources are candidates. Re-review their input bindings with
review_xland_online_sources before using official expert targets for training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.evaluate_xland_official_live import (
    Worker, episode, load_actor, selected_source_steps, sha256, digest,
)


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    original = json.loads(args.annotations.read_text())
    torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                               device=args.device)
    agent, tokenizer, choices = load_actor(args)
    worker = Worker(args.xland_python, args.data_dir, args.benchmark,
                    args.environment, args.benchmark_path)
    output = {"protocol": "Model-selected official XLand episodes from empty history; frozen actor/hypernetwork; source-only collection with no target expert actions",
              "annotations_sha256": sha256(args.annotations),
              "checkpoint_sha256": sha256(args.checkpoint),
              "training_result_sha256": sha256(args.training_result),
              "benchmark_sha256": sha256(args.benchmark_path),
              "source_seed": args.source_seed,
              "source_sampling_seed": args.source_sampling_seed,
              "source_steps": args.source_steps,
              "source_epsilon": args.source_epsilon,
              "source_feedback_window": args.source_feedback_window,
              "stream_source_updates": args.stream_source_updates,
              "split": {name: [] for name in ("train", "dev", "test")},
              "failures": []}
    try:
        for split in ("train", "dev", "test"):
            limit = getattr(args, "limit_" + split)
            for item in original["split"][split][:limit]:
                rid = item["ruleset_id"]
                positive = None
                try:
                    source = episode(worker, agent, tokenizer, choices,
                        rid, args.source_seed + rid, args.source_steps, None,
                        args.device, sample_seed=args.source_sampling_seed + rid,
                        stream_updates=args.stream_source_updates,
                        epsilon=args.source_epsilon,
                        feedback_window=args.source_feedback_window)
                    public, window = selected_source_steps(source["steps"])
                    if not public:
                        raise ValueError("Empty source episode")
                    positive = source["positive_rewards"]
                    output["split"][split].append({"task_id": item["task_id"],
                        "ruleset_id": rid, "source": source,
                        "source_sha256": digest(source),
                        "selected_window": list(window),
                        "selected_public_sha256": digest(public)})
                except Exception as exc:
                    output["failures"].append({"split": split,
                        "task_id": item["task_id"], "ruleset_id": rid,
                        "error": f"{type(exc).__name__}: {exc}"})
                output["summary"] = {name: {
                    "n": len(output["split"][name]),
                    "positive": sum(row["source"]["positive_rewards"] > 0
                        for row in output["split"][name])}
                    for name in output["split"]}
                args.output.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.output.with_suffix(args.output.suffix + ".tmp")
                temporary.write_text(json.dumps(output, indent=2) + "\n")
                temporary.replace(args.output)
                print(json.dumps({"split": split, "task_id": item["task_id"],
                    "positive": positive,
                    "summary": output["summary"]}), flush=True)
    finally:
        worker.close()
    return output


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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.45)
    parser.add_argument("--source-seed", type=int, default=61005)
    parser.add_argument("--source-sampling-seed", type=int, default=81005)
    parser.add_argument("--source-steps", type=int, default=200)
    parser.add_argument("--source-epsilon", type=float, default=.2)
    parser.add_argument("--source-feedback-window", type=int, default=4)
    parser.add_argument("--stream-source-updates", action="store_true")
    parser.add_argument("--limit-train", type=int, default=60)
    parser.add_argument("--limit-dev", type=int, default=13)
    parser.add_argument("--limit-test", type=int, default=13)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
