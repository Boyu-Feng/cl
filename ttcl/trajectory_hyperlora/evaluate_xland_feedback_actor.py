"""Paired live test of a self-distilled static LoRA exploration policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.evaluate_xland_official_live import (
    Worker, episode, sha256,
)
from ttcl.trajectory_hyperlora.train_xland_official_history_hyperlora import QwenRawHyperLoRA


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    result = json.loads(args.training_result.read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu",
                            weights_only=True)
    if (result["sources_sha256"] != checkpoint["sources_sha256"] or
            result["sources_sha256"] != sha256(args.sources)):
        raise ValueError("Self-distilled actor lineage mismatch")
    torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                               device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    choice_ids = [tokenizer(str(i), add_special_tokens=False).input_ids[0]
                  for i in range(result["choice_count"])]
    base = AutoModelForCausalLM.from_pretrained(args.model,
        torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    base.config.use_cache = False
    agent = QwenRawHyperLoRA(base, result["rank"], result["layers"],
                             64, False).to(args.device)
    factors = []
    with torch.no_grad():
        for adapter, matrix_a, matrix_b in zip(agent.adapters,
                checkpoint["lora_a"], checkpoint["static_b"], strict=True):
            adapter.a.copy_(matrix_a.to(adapter.a.device))
            factors.append(matrix_b.to(args.device).unsqueeze(0))
    agent.eval()
    annotations = json.loads(args.annotations.read_text())
    items = annotations["split"][args.split][:args.limit]
    worker = Worker(args.xland_python, args.data_dir, args.benchmark,
                    args.environment, args.benchmark_path)
    output = {"protocol": "Paired held-out live exploration with self-distilled static LoRA versus frozen Qwen; both see same within-episode feedback, seeds and action budget",
              "training_result_sha256": sha256(args.training_result),
              "checkpoint_sha256": sha256(args.checkpoint),
              "annotations_sha256": sha256(args.annotations),
              "benchmark_sha256": sha256(args.benchmark_path),
              "split": args.split, "source_seed": args.source_seed,
              "source_sampling_seed": args.source_sampling_seed,
              "steps": args.steps, "epsilon": args.epsilon,
              "feedback_window": args.feedback_window,
              "rows": [], "failures": []}
    try:
        for item in items:
            rid = item["ruleset_id"]
            try:
                arms = {}
                for name, adapter in (("distilled", factors), ("base", None)):
                    arms[name] = episode(worker, agent, tokenizer, choice_ids,
                        rid, args.source_seed + rid, args.steps, adapter,
                        args.device, sample_seed=args.source_sampling_seed + rid,
                        epsilon=args.epsilon,
                        feedback_window=args.feedback_window)
                output["rows"].append({"task_id": item["task_id"],
                    "ruleset_id": rid, "arms": arms})
            except Exception as exc:
                output["failures"].append({"task_id": item["task_id"],
                    "error": f"{type(exc).__name__}: {exc}"})
            output["summary"] = {"n": len(output["rows"]),
                **{name + "_positive": sum(row["arms"][name][
                    "positive_rewards"] > 0 for row in output["rows"])
                    for name in ("distilled", "base")}}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(json.dumps(output, indent=2) + "\n")
            temporary.replace(args.output)
            print(json.dumps(output["summary"]), flush=True)
    finally:
        worker.close()
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
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
    parser.add_argument("--source-seed", type=int, default=61005)
    parser.add_argument("--source-sampling-seed", type=int, default=81005)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--epsilon", type=float, default=.2)
    parser.add_argument("--feedback-window", type=int, default=4)
    parser.add_argument("--limit", type=int, default=13)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
