"""Frozen 134-game, LoRA-only evaluation for a checkpoint ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_memrl134_transfer import checked_targets, _save
from ttcl.trajectory_hyperlora.alfworld_same_game_retry import file_hash
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_source_fields


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    targets = checked_targets(args)
    chosen = targets[args.offset:args.offset + args.limit]
    if not chosen:
        raise ValueError("No selected valid_unseen games")
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if agent.encoder_kind != "contextual" or not agent.task_conditioned:
        raise ValueError("Expected contextual trajectory hypernetwork")
    report = {"protocol": "Frozen 134 valid_unseen source/target bindings; original checkpoint LoRA-only with two-repeat observation-action guard; 50 steps, 64 tokens, greedy constrained commands, two-turn history, official won; matched guarded-base result stored separately",
        "target_review_sha256": file_hash(args.target_review),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "offset": args.offset, "limit": args.limit,
        "loop_guard_max": 2, "max_steps": 50, "max_new_tokens": 64,
        "actor_history_turns": 2,
        "games": [], "failures": []}
    for row in chosen:
        try:
            fields = contextual_source_fields(
                agent, tokenizer, row["source_records"], args.device,
                2048, pooling="both")
            result = run_episode(agent, tokenizer,
                args.data_root / row["game"], fields, adapter=True,
                device=args.device, max_steps=50, max_new_tokens=64,
                constrain_actions=True, actor_history_turns=2,
                loop_guard_max=2)
            if result["status"] != "complete":
                raise RuntimeError(str(result))
            report["games"].append({"game": row["game"],
                "family": row["family"],
                "target_input_content_sha256": row["input_content_sha256"],
                "source_records_sha256": row["source_records_sha256"],
                "lora": result})
        except Exception as exc:
            report["failures"].append({"game": row["game"],
                "error": f"{type(exc).__name__}: {exc}"})
        _save(args.output, report)
        print(json.dumps({"n": len(report["games"]),
            "failures": len(report["failures"]),
            "wins": sum(r["lora"]["reward"] for r in report["games"])}),
            flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memrl-plan", type=Path, default=Path(
        "results/memrl_comparison/20260928_budgeted/plan.json"))
    parser.add_argument("--source-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train600_reviewed_20261006.json"))
    parser.add_argument("--retry-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_same_game_retry_candidates_20261006.json"))
    parser.add_argument("--retry-review", type=Path, default=Path(
        "data/annotations/alf_same_game_retry_reviewed_20261006.json"))
    parser.add_argument("--all-source-review", type=Path, default=Path(
        "data/annotations/alf_expert_first_attempts_reviewed_20261006.json"))
    parser.add_argument("--target-review", type=Path, default=Path(
        "data/annotations/alf_memrl134_hyperlora_train_sources_reviewed_20261006.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.55)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=134)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.offset < 0 or args.limit < 1:
        parser.error("Invalid target slice")
    evaluate(args)


if __name__ == "__main__":
    main()
