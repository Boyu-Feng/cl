"""Matched-norm perturbation control for old-HyperLoRA failed retries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_failure_retry_placebo_v31c import (
    matched_random_residual,
)
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_success_gated_factors_v27 import episode_factor
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_source_fields
from ttcl.trajectory_hyperlora.evaluate_alfworld_failure_retry_v31b import (
    checked_input, retry, source_factors,
)


def inputs(args):
    pairs, first, _, _ = checked_input(args)
    comparison = json.loads(args.comparison.read_text())
    if (len(comparison["games"]) != 24 or comparison["failures"] or
            comparison["summary"]["first_failures"] != 13 or
            comparison["first_report_sha256"] != file_hash(args.first_report) or
            comparison["checkpoint_sha256"] != file_hash(args.checkpoint) or
            comparison["residual_sha256"] != file_hash(
                Path(__file__).with_name(
                    "alfworld_failure_retry_residual_v31b.py"))):
        raise ValueError("Changed or incomplete v31b comparison")
    fixed = {"method": "我的在线更新 v31c matched-norm random control",
        "reviewed_pairs_sha256": file_hash(args.reviewed_pairs),
        "first_report_sha256": file_hash(args.first_report),
        "comparison_sha256": file_hash(args.comparison),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "runner_sha256": file_hash(Path(__file__)),
        "actor_sha256": file_hash(Path(__file__).with_name(
            "alfworld_zero_shot_probe.py")),
        "placebo_sha256": file_hash(Path(__file__).with_name(
            "alfworld_failure_retry_placebo_v31c.py")),
        "dev_runner_sha256": file_hash(Path(__file__).with_name(
            "evaluate_alfworld_failure_retry_v31b.py")),
        "max_steps": 50, "max_new_tokens": 64,
        "history_turns": 2, "repeat_limit": 2}
    targets = []
    for i, (pair, old, tested) in enumerate(zip(
            pairs, first["games"], comparison["games"], strict=True)):
        if (tested["index"] != i or tested["game"] != pair["target_game"] or
                tested["first_episode_sha256"] != digest(old["new"])):
            raise ValueError("Changed first or comparison target")
        if bool(old["new"]["reward"]):
            continue
        content = {"index": i, "game": pair["target_game"],
            "game_sha256": pair["target_game_sha256"],
            "pair_input_content_sha256": pair["input_content_sha256"],
            "first_episode_sha256": digest(old["new"]),
            "true_residual_sha256": tested["residual_factor_sha256"],
            "true_residual_effective_norm": tested["residual_info"][
                "residual_effective_norm"],
            "comparison_sha256": fixed["comparison_sha256"],
            "runner_sha256": fixed["runner_sha256"]}
        if file_hash(args.data_root / pair["target_game"]) != content["game_sha256"]:
            raise ValueError("Changed official development task")
        targets.append({**content, "input_content_sha256": digest(content),
            "reviewed_target": True})
    if len(targets) != 13:
        raise ValueError("Changed failed development set")
    return pairs, first, comparison, fixed, targets


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    *_, fixed, targets = inputs(args)
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps({"protocol":
        "New content-bound placebo retries for 13 exposed ALFWorld train-development failures; no future/test outcomes for candidate construction",
        **fixed, "targets": targets, "targets_sha256": digest(targets)},
        ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"reviewed_failures": len(targets)}), flush=True)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    temp.replace(path)


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    pairs, first, comparison, fixed, targets = inputs(args)
    review = json.loads(args.review.read_text())
    if (review["targets"] != targets or
            review["targets_sha256"] != digest(targets) or
            any(review[k] != v for k, v in fixed.items())):
        raise ValueError("Changed placebo review")
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    report = {"protocol":
        "Exposed ALFWorld train-development failures: one matched-effective-norm random B correction retry versus frozen v31b true-failure and static retries; same actor and action guard",
        "review_sha256": file_hash(args.review), **fixed,
        "games": [], "failures": []}
    for binding in targets:
        i = binding["index"]
        pair = pairs[i]
        first_episode = first["games"][i]["new"]
        source = contextual_source_fields(agent, tokenizer,
            pair["source_records"], args.device, 2048, pooling="both")
        expert = source_factors(agent, tokenizer, source,
            first_episode["initial_observation"], args.device)
        failed, _, _, _ = episode_factor(agent, tokenizer, first_episode,
            argparse.Namespace(device=args.device, context_tokens=2048))
        if (factors_hash(expert) != comparison["games"][i][
                "expert_factor_sha256"] or
                factors_hash(failed) != comparison["games"][i][
                    "failed_factor_sha256"]):
            raise ValueError("Changed source encoding for placebo control")
        seed = int(digest(["matched_random_v31c", binding["game"]])[:16], 16)
        random_residual = matched_random_residual(agent, expert,
            binding["true_residual_effective_norm"], seed)
        episode = retry(agent, tokenizer, args.data_root / pair["target_game"],
                        source, args.device, random_residual)
        row = {"index": i, "game": binding["game"],
            "input_content_sha256": binding["input_content_sha256"],
            "seed": seed,
            "random_residual_sha256": factors_hash(random_residual),
            "retry": episode}
        report["games"].append(row)
        if (episode["status"] != "complete" or
                episode["initial_observation"] !=
                first_episode["initial_observation"]):
            report["failures"].append({"index": i, "episode": episode})
        report["summary"] = {"failed_targets": len(report["games"]),
            "random_rescues": sum(row["retry"].get("reward", 0)
                for row in report["games"]),
            "true_failure_rescues": sum(comparison["games"][row["index"]]
                ["retries"]["away_from_failure"]["reward"]
                for row in report["games"]),
            "failures": len(report["failures"])}
        save(args.output, report)
        print(json.dumps(report["summary"]), flush=True)
        if report["failures"]:
            raise RuntimeError("Official placebo retry failure recorded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate"))
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
    parser.add_argument("--first-report", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_parent_loop2_dev24_20261007.json"))
    parser.add_argument("--comparison", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_failure_retry_v31b_dev24_20261009.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/alfworld_failure_retry_placebo_v31c_dev24_20261009.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_failure_retry_placebo_v31c_dev24_20261009.json"))
    parser.add_argument("--max-relative", type=float, default=.25)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--gpu-fraction", type=float, default=.65)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else evaluate)(args)
