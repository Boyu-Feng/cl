"""Evaluate a newly reward-distilled LoRA on reviewed, untouched ALFWorld dev."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_same_game_retry import file_hash
from ttcl.trajectory_hyperlora.alfworld_reward_router_data_v1 import checked_pairs
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_source_fields


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    original = args.checkpoint
    pairs = [row for row in checked_pairs(args) if row["split"] == "dev"]
    if len(pairs) != 24:
        raise ValueError("Expected 24 untouched development pairs")
    pairs = pairs[args.offset:args.offset + args.limit]
    if not pairs:
        raise ValueError("Empty development slice")
    selected_checkpoint = original if args.use_parent else args.new_checkpoint
    if not args.use_parent:
        new_checkpoint = torch.load(selected_checkpoint, map_location="cpu",
                                    weights_only=True)
        if (new_checkpoint.get("source_checkpoint_sha256") != file_hash(original) or
                new_checkpoint.get("reward_selected_labels_sha256") !=
                    file_hash(args.labels) or
                new_checkpoint.get("reward_selected_review_sha256") !=
                    file_hash(args.label_review)):
            raise ValueError("Reward-distilled checkpoint lineage changed")
    agent, tokenizer = load_agent(args.model, selected_checkpoint,
                                  args.device, args.gpu_fraction)
    if agent.encoder_kind != "contextual" or not agent.task_conditioned:
        raise ValueError("Expected task-conditioned contextual hypernetwork")
    report = {"protocol": "Fresh ALFWorld train-domain development evaluation of reward-distilled LoRA; 24 previously reviewed pairs excluded from distillation; 50 steps/64 tokens, 2-turn compact history, official won; old checkpoint base and LoRA results in separate frozen reward collection",
        "reviewed_pairs_sha256": file_hash(args.reviewed_pairs),
        "source_checkpoint_sha256": file_hash(original),
        "new_checkpoint_sha256": file_hash(selected_checkpoint),
        "offset": args.offset, "limit": args.limit,
        "base": args.base, "use_parent": args.use_parent,
        "loop_guard_max": args.loop_guard_max,
        "games": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for pair in pairs:
        try:
            fields = contextual_source_fields(agent, tokenizer,
                pair["source_records"], args.device, 2048, pooling="both")
            result = run_episode(agent, tokenizer,
                args.data_root / pair["target_game"], fields,
                adapter=not args.base,
                device=args.device, max_steps=50, max_new_tokens=64,
                constrain_actions=True, actor_history_turns=2,
                loop_guard_max=args.loop_guard_max)
            if result["status"] != "complete":
                raise RuntimeError(str(result))
            report["games"].append({"game": pair["target_game"],
                "input_content_sha256": pair["input_content_sha256"],
                "source_records_sha256": pair["source_records_sha256"],
                "new": result})
        except Exception as exc:
            report["failures"].append({"game": pair["target_game"],
                "error": f"{type(exc).__name__}: {exc}"})
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(args.output)
        print(json.dumps({"completed": len(report["games"]),
            "failures": len(report["failures"]),
            "successes": sum(x["new"]["reward"] for x in report["games"])}),
            flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed-pairs", type=Path, default=Path(
        "data/annotations/alf_reward_router_train72_reviewed_20261007.json"))
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train600_reviewed_20261006.json"))
    parser.add_argument("--retry-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_same_game_retry_candidates_20261006.json"))
    parser.add_argument("--retry-review", type=Path, default=Path(
        "data/annotations/alf_same_game_retry_reviewed_20261006.json"))
    parser.add_argument("--all-source-review", type=Path, default=Path(
        "data/annotations/alf_expert_first_attempts_reviewed_20261006.json"))
    parser.add_argument("--labels", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_reward_selected_train48_v1_20261007.json"))
    parser.add_argument("--label-review", type=Path, default=Path(
        "data/annotations/alf_reward_selected_train48_v1_reviewed_20261007.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--new-checkpoint", type=Path)
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--data-root", type=Path, default=Path("ttcl/data/alfworld_delta"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.65)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--loop-guard-max", type=int, default=None)
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--use-parent", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.offset < 0 or args.limit < 1:
        parser.error("Invalid development slice")
    if args.loop_guard_max is not None and args.loop_guard_max < 1:
        parser.error("Loop guard threshold must be positive")
    if not args.use_parent and args.new_checkpoint is None:
        parser.error("Reward-distilled evaluation needs --new-checkpoint")
    run(args)


if __name__ == "__main__":
    main()
