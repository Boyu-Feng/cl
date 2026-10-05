"""Measure frozen Qwen exploration with feedback on official XLand rulesets.

This is a model-capacity/data-acquisition diagnostic, not a hypernetwork test.
The existing Qwen4B baseline uses the same episode helper and task seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.evaluate_xland_official_live import (
    Worker, episode, sha256,
)
from ttcl.trajectory_hyperlora.train_xland_qwen_hyperlora import QwenRawHyperLoRA


class NoThinkingTokenizer:
    """Make Qwen3 base models expose a direct action token, not a think prefix."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(
            *args, enable_thinking=False, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.model.joinpath("config.json").is_file():
        raise FileNotFoundError("Local model weights/config are required")
    annotations = json.loads(args.annotations.read_text())
    items = annotations["split"][args.split][:args.limit]
    if not items:
        raise ValueError("Empty reviewed split")
    torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                               device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              local_files_only=True)
    choices = [tokenizer(str(i), add_special_tokens=False).input_ids
               for i in range(args.choice_count)]
    if any(len(tokens) != 1 for tokens in choices):
        raise ValueError("Action choice is not one token")
    choice_ids = [tokens[0] for tokens in choices]
    prompt_tokenizer = (NoThinkingTokenizer(tokenizer) if args.no_thinking
                        else tokenizer)
    base = AutoModelForCausalLM.from_pretrained(args.model,
        torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    base.config.use_cache = False
    actor = QwenRawHyperLoRA(base, rank=8, layers=2, width=64,
                            order_invariant_source=False).to(args.device)
    actor.eval()
    worker = Worker(args.xland_python, args.data_dir, args.benchmark,
                    args.environment, args.benchmark_path)
    output = {"protocol": "Frozen model feedback-guided exploration from empty history; no generated LoRA, no expert actions; version-bound official XLand reward",
              "model_config_sha256": sha256(args.model / "config.json"),
              "annotations_sha256": sha256(args.annotations),
              "benchmark_sha256": sha256(args.benchmark_path),
              "model": str(args.model), "split": args.split,
              "seed": args.seed, "sampling_seed": args.sampling_seed,
              "steps": args.steps, "epsilon": args.epsilon,
              "feedback_window": args.feedback_window,
              "choice_count": args.choice_count,
              "no_thinking": args.no_thinking,
              "rows": [], "failures": []}
    try:
        for item in items:
            rid = item["ruleset_id"]
            try:
                outcome = episode(worker, actor, prompt_tokenizer,
                    choice_ids, rid, args.seed + rid, args.steps, None,
                    args.device, sample_seed=args.sampling_seed + rid,
                    epsilon=args.epsilon,
                    feedback_window=args.feedback_window)
                output["rows"].append({"task_id": item["task_id"],
                    "ruleset_id": rid, "episode": outcome})
            except Exception as exc:
                output["failures"].append({"task_id": item["task_id"],
                    "error": f"{type(exc).__name__}: {exc}"})
            output["summary"] = {"n": len(output["rows"]),
                "positive": sum(row["episode"]["positive_rewards"] > 0
                                for row in output["rows"]),
                "failures": len(output["failures"])}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temp = args.output.with_suffix(args.output.suffix + ".tmp")
            temp.write_text(json.dumps(output, indent=2) + "\n")
            temp.replace(args.output)
            print(json.dumps(output["summary"]), flush=True)
    finally:
        worker.close()
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--benchmark-path", type=Path, required=True)
    parser.add_argument("--xland-python", type=Path, default=Path(
        "ttcl/.runtime/xland_env/bin/python"))
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/xland_minigrid"))
    parser.add_argument("--benchmark", default="trivial-1m")
    parser.add_argument("--environment", default="XLand-MiniGrid-R1-9x9")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--limit", type=int, default=13)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.7)
    parser.add_argument("--seed", type=int, default=61005)
    parser.add_argument("--sampling-seed", type=int, default=81005)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--epsilon", type=float, default=.2)
    parser.add_argument("--feedback-window", type=int, default=4)
    parser.add_argument("--choice-count", type=int, default=5)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.steps < 1 or args.limit < 1 or
            args.choice_count not in (5, 6) or
            not 0 <= args.epsilon <= 1 or
            not 0 < args.gpu_fraction <= 1):
        parser.error("Invalid experiment budget")
    run(args)


if __name__ == "__main__":
    main()
