"""Online ALFWorld: add each own trajectory's generated LoRA to persistent factors.

The actor and hypernetwork stay frozen. Only earlier completed episodes can
change the factor state used by a later game; no walkthrough is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode, select_games,
)
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.relational_router_pilot import tokenize_records


def factors_hash(factors: list[torch.Tensor] | None) -> str:
    state = hashlib.sha256()
    if factors is None:
        state.update(b"empty")
    else:
        for factor in factors:
            value = factor.detach().cpu().contiguous().float()
            state.update(str(tuple(value.shape)).encode())
            state.update(value.numpy().tobytes())
    return state.hexdigest()


def add_factors(previous: list[torch.Tensor] | None,
                increment: list[torch.Tensor]) -> list[torch.Tensor]:
    if not increment or not all(torch.isfinite(x).all() for x in increment):
        raise ValueError("Generated LoRA increment is empty or nonfinite")
    if previous is None:
        return [x.detach().cpu().float().clone() for x in increment]
    if len(previous) != len(increment):
        raise ValueError("LoRA layer count changed")
    result = []
    for old, delta in zip(previous, increment, strict=True):
        if old.shape != delta.shape:
            raise ValueError("LoRA factor shape changed")
        updated = old + delta.detach().cpu().float()
        if not torch.isfinite(updated).all():
            raise ValueError("Accumulated LoRA factor became nonfinite")
        result.append(updated)
    return result


def mean_factors(previous: list[torch.Tensor] | None,
                 increment: list[torch.Tensor], count: int) -> list[torch.Tensor]:
    """Keep a running mean of trajectory-generated LoRA factors in-place."""
    if count < 0:
        raise ValueError("Increment count must be nonnegative")
    if previous is None:
        if count != 0:
            raise ValueError("Missing mean factors after previous updates")
        return add_factors(None, increment)
    if count == 0 or len(previous) != len(increment):
        raise ValueError("Invalid running mean state")
    if not all(torch.isfinite(x).all() for x in increment):
        raise ValueError("Generated LoRA increment is nonfinite")
    result = []
    for old, delta in zip(previous, increment, strict=True):
        if old.shape != delta.shape:
            raise ValueError("LoRA factor shape changed")
        updated = old + (delta.detach().cpu().float() - old) / (count + 1)
        if not torch.isfinite(updated).all():
            raise ValueError("Running mean LoRA factor became nonfinite")
        result.append(updated)
    return result


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    parent = json.loads(args.parent_review.read_text())
    plan = json.loads(args.plan.read_text())
    games = select_games(plan, parent["first_sequence"], parent["sequences"])
    if (parent["plan_sha256"] != file_hash(args.plan) or
            [row["game"] for row in parent["targets"]] != games):
        raise ValueError("Historical online target sequence changed")
    targets = []
    for index, (game, old) in enumerate(zip(games, parent["targets"], strict=True)):
        game_hash = file_hash(args.data_root / game)
        if game_hash != old["game_sha256"]:
            raise ValueError("Online game content changed")
        content = {"index": index, "game": game,
                   "game_sha256": game_hash,
                   "parent_review_sha256": file_hash(args.parent_review)}
        if args.mode == "mean":
            content["update_mode"] = "mean"
        targets.append({**content, "input_content_sha256": digest(content),
            "reviewed_target": True,
            "review_basis": "Fresh frozen train-plan target and game-content review; action targets come only from own live transitions"})
    review = {"protocol": "Fresh content-bound target review for additive online LoRA; no initial memory or target walkthrough",
        "plan_sha256": file_hash(args.plan),
        "parent_review_sha256": file_hash(args.parent_review),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "targets": targets}
    if args.mode == "mean":
        review["update_mode"] = "mean"
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"reviewed_targets": len(targets)}), flush=True)


def checked_targets(args):
    review = json.loads(args.review.read_text())
    parent = json.loads(args.parent_review.read_text())
    plan = json.loads(args.plan.read_text())
    games = select_games(plan, parent["first_sequence"], parent["sequences"])
    if (review["plan_sha256"] != file_hash(args.plan) or
            review["parent_review_sha256"] != file_hash(args.parent_review) or
            review["checkpoint_sha256"] != file_hash(args.checkpoint) or
            [r["game"] for r in review["targets"]] != games):
        raise ValueError("Additive online target review changed")
    if args.mode == "mean" and review.get("update_mode") != "mean":
        raise ValueError("Running-mean target review changed")
    if args.mode == "sum" and "update_mode" in review:
        raise ValueError("Sum target review changed")
    for index, row in enumerate(review["targets"]):
        content = {"index": index, "game": games[index],
                   "game_sha256": file_hash(args.data_root / games[index]),
                   "parent_review_sha256": file_hash(args.parent_review)}
        if args.mode == "mean":
            content["update_mode"] = "mean"
        if (not row["reviewed_target"] or
                row["input_content_sha256"] != digest(content) or
                any(row[key] != value for key, value in content.items())):
            raise ValueError("Unreviewed or changed additive target")
    return review["targets"]


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    targets = checked_targets(args)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if agent.encoder_kind not in ("covariance", "factorized") or agent.task_conditioned:
        raise ValueError("Expected non-task-conditioned relational LoRA checkpoint")
    factors = None
    increment_count = 0
    prior_episodes = []
    report = {"protocol": "Prequential ALFWorld from empty memory: after each complete own episode generate LoRA from that episode alone and add factors B; frozen actor, frozen hypernetwork and fixed A; compare separately against same-sequence all-history regeneration",
        "review_sha256": file_hash(args.review),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "max_steps": args.max_steps, "max_new_tokens": args.max_new_tokens,
        "source_field_token_limit": args.field_tokens,
        "update_rule": ("B_next = B_previous + B(one_new_own_trajectory); no clipping or normalization"
            if args.mode == "sum" else
            "B_next = B_previous + (B(one_new_own_trajectory) - B_previous) / (increment_count + 1)"),
        "update_mode": args.mode,
        "games": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for row in targets:
        before = factors_hash(factors)
        source_hash = digest(prior_episodes)
        episode = run_episode(agent, tokenizer, args.data_root / row["game"],
            {}, adapter=False, fixed_adapter=factors, device=args.device,
            max_steps=args.max_steps, max_new_tokens=args.max_new_tokens,
            constrain_actions=True)
        entry = {"game": row["game"],
            "input_content_sha256": row["input_content_sha256"],
            "prior_episodes_sha256": source_hash,
            "prior_episode_count": len(prior_episodes),
            "factor_sha256_before": before,
            "factor_l2_before": [float(x.norm()) for x in factors] if factors else [],
            "episode": episode}
        if episode["status"] == "complete":
            try:
                records = records_from_episode(episode)
                entry["new_records_sha256"] = digest(records)
                if len(records) >= 2:
                    fields = tokenize_records(tokenizer, records, args.device,
                                              max_tokens=args.field_tokens)
                    with torch.no_grad():
                        agent.set_source(fields)
                        increment = [layer.b.detach().cpu().float().clone()
                                     for layer in agent.adapters]
                        agent.set_source(None)
                    factors = (add_factors(factors, increment)
                        if args.mode == "sum" else
                        mean_factors(factors, increment, increment_count))
                    increment_count += 1
                    entry["increment_l2"] = [float(x.norm()) for x in increment]
                    entry["increment_applied"] = True
                else:
                    entry["increment_l2"] = []
                    entry["increment_applied"] = False
                prior_episodes.append({"game": row["game"],
                    "reward": episode["reward"], "records": records})
            except Exception as exc:
                report["failures"].append({"game": row["game"],
                    "error": f"{type(exc).__name__}: {exc}"})
        else:
            report["failures"].append({"game": row["game"],
                "error": episode.get("error", "incomplete rollout")})
        entry["factor_sha256_after"] = factors_hash(factors)
        entry["increment_count_after"] = increment_count
        entry["factor_l2_after"] = [float(x.norm()) for x in factors] if factors else []
        report["games"].append(entry)
        report["summary"] = {"completed": sum(x["episode"]["status"] == "complete"
                                         for x in report["games"]),
            "success": sum(x["episode"].get("reward", 0) for x in report["games"]),
            "failures": len(report["failures"])}
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(args.output)
        print(json.dumps({"n": len(report["games"]),
            "reward": episode.get("reward"),
            "summary": report["summary"],
            "factor_l2": entry["factor_l2_after"]}), flush=True)
        if report["failures"]:
            raise RuntimeError("Additive online failure recorded")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_warmstart_20261005/seed42_v2.pt"))
    parser.add_argument("--plan", type=Path, default=Path(
        "ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--parent-review", type=Path, default=Path(
        "data/annotations/alfworld_online_from_empty_train_seq0_6_reviewed_20261005.json"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/alfworld_online_lora_additive_train_seq0_6_reviewed_20261007.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_lora_additive_train_seq0_6_20261007.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.55)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--field-tokens", type=int, default=40)
    parser.add_argument("--mode", choices=("sum", "mean"), default="sum")
    args = parser.parse_args()
    if args.max_steps < 1 or args.max_new_tokens < 1 or args.field_tokens < 1:
        parser.error("Invalid online budget")
    (prepare if args.command == "prepare" else evaluate)(args)


if __name__ == "__main__":
    main()
