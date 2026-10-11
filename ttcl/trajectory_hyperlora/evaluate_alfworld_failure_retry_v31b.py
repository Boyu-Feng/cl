"""Development retry of the frozen 68/134 LoRA and action-guard method.

The exact historical first attempts are bound as input. On each failure, the
same expert source and actor run one retry per arm. No target walkthrough or
hidden reward is used to construct either candidate residual.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_failure_retry_residual_v31b import (
    bounded_residual, inject_dynamic_residual,
)
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_reward_router_data_v1 import checked_pairs
from ttcl.trajectory_hyperlora.alfworld_success_gated_factors_v27 import episode_factor
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import (
    contextual_source_fields, contextual_text_fields, task_context_text,
)


ARMS = ("static_retry", "toward_failure", "away_from_failure")


def checked_input(args):
    pairs = [row for row in checked_pairs(args) if row["split"] == "dev"]
    first_report = json.loads(args.first_report.read_text())
    if (len(pairs) != 24 or len(first_report["games"]) != 24 or
            first_report["failures"] or
            first_report["reviewed_pairs_sha256"] !=
                file_hash(args.reviewed_pairs) or
            first_report["source_checkpoint_sha256"] !=
                file_hash(args.checkpoint) or
            first_report["new_checkpoint_sha256"] !=
                file_hash(args.checkpoint) or
            first_report["use_parent"] is not True or
            first_report["base"] is not False or
            first_report["loop_guard_max"] != 2):
        raise ValueError("Changed frozen old-LoRA development first attempts")
    fixed = {"method": "我的在线更新 v31b failure retry on old 68/134 checkpoint",
        "reviewed_pairs_sha256": file_hash(args.reviewed_pairs),
        "first_report_sha256": file_hash(args.first_report),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "runner_sha256": file_hash(Path(__file__)),
        "actor_sha256": file_hash(Path(__file__).with_name(
            "alfworld_zero_shot_probe.py")),
        "residual_sha256": file_hash(Path(__file__).with_name(
            "alfworld_failure_retry_residual_v31b.py")),
        "failed_encoder_sha256": file_hash(Path(__file__).with_name(
            "alfworld_success_gated_factors_v27.py")),
        "max_relative": args.max_relative,
        "max_steps_per_attempt": 50,
        "max_new_tokens": 64, "history_turns": 2,
        "repeat_limit": 2, "failure_source_tokens": 2048}
    targets = []
    for index, (pair, old) in enumerate(zip(pairs, first_report["games"],
                                           strict=True)):
        first = old["new"]
        if (old["game"] != pair["target_game"] or
                old["input_content_sha256"] !=
                    pair["input_content_sha256"] or
                old["source_records_sha256"] !=
                    pair["source_records_sha256"] or
                first["status"] != "complete" or
                first["steps"] > 50 or
                first["reward"] != float(first["termination"] == "success") or
                file_hash(args.data_root / pair["target_game"]) !=
                    pair["target_game_sha256"]):
            raise ValueError("Changed reviewed source or official first attempt")
        content = {"index": index, "game": pair["target_game"],
            "game_sha256": pair["target_game_sha256"],
            "pair_input_content_sha256": pair["input_content_sha256"],
            "source_records_sha256": pair["source_records_sha256"],
            "first_episode_sha256": digest(first),
            "first_report_sha256": fixed["first_report_sha256"],
            "runner_sha256": fixed["runner_sha256"],
            "checkpoint_sha256": fixed["checkpoint_sha256"]}
        targets.append({**content, "input_content_sha256": digest(content),
            "reviewed_target": True})
    return pairs, first_report, fixed, targets


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    _, first_report, fixed, targets = checked_input(args)
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps({"protocol":
        "Fresh content-bound failure-retry inputs on previously exposed 24 ALFWorld training-domain development tasks; no independent-test claim",
        **fixed, "targets": targets, "targets_sha256": digest(targets)},
        ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"reviewed_targets": len(targets),
        "first_failures": sum(not bool(row["new"]["reward"])
            for row in first_report["games"])}), flush=True)


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def source_factors(agent, tokenizer, fields, initial, device):
    target = contextual_text_fields(agent, tokenizer,
        task_context_text(initial, initial), device, 2048, pooling="both")
    with torch.no_grad():
        agent.set_source(fields, target_fields=target)
        result = [layer.b.detach().cpu().float().clone()
                  for layer in agent.adapters]
        agent.set_source(None)
    return result


def retry(agent, tokenizer, game, fields, device, residual=None):
    kwargs = dict(adapter=True, device=device, max_steps=50,
        max_new_tokens=64, constrain_actions=True,
        actor_history_turns=2, loop_guard_max=2)
    if residual is None:
        return run_episode(agent, tokenizer, game, fields, **kwargs)
    with inject_dynamic_residual(agent, residual):
        return run_episode(agent, tokenizer, game, fields, **kwargs)


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    pairs, first_report, fixed, targets = checked_input(args)
    review = json.loads(args.review.read_text())
    if (review["targets"] != targets or
            review["targets_sha256"] != digest(targets) or
            any(review[key] != val for key, val in fixed.items())):
        raise ValueError("Changed v31b input review")
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != "contextual" or not agent.task_conditioned or
            agent.task_context_scope != "current" or
            agent.task_pair_pooling != "mean"):
        raise ValueError("Changed old contextual hypernetwork")
    report = {"protocol":
        "Frozen 68/134 actor and expert-source LoRA with two-repeat action guard; historical first attempt; same-budget static versus opposite bounded failure-B residual retries; exposed train-domain dev only",
        "review_sha256": file_hash(args.review), **fixed,
        "arms": ARMS, "games": [], "failures": []}
    for pair, old, binding in zip(pairs, first_report["games"], targets,
                                  strict=True):
        first = old["new"]
        row = {"index": binding["index"], "game": pair["target_game"],
            "input_content_sha256": binding["input_content_sha256"],
            "first_episode_sha256": binding["first_episode_sha256"],
            "first_reward": first["reward"], "retries": {}}
        if not bool(first["reward"]):
            fields = contextual_source_fields(agent, tokenizer,
                pair["source_records"], args.device, 2048, pooling="both")
            expert = source_factors(agent, tokenizer, fields,
                                    first["initial_observation"], args.device)
            failed, _, original, retained = episode_factor(
                agent, tokenizer, first, argparse.Namespace(
                    device=args.device, context_tokens=2048))
            residual, info = bounded_residual(agent, expert, failed,
                                              max_relative=args.max_relative)
            row.update(expert_factor_sha256=factors_hash(expert),
                failed_factor_sha256=factors_hash(failed),
                residual_factor_sha256=factors_hash(residual),
                failed_source_tokens_original=original,
                failed_source_tokens_retained=retained,
                residual_info=info)
            game = args.data_root / pair["target_game"]
            for arm in ARMS:
                adjustment = (None if arm == "static_retry" else
                    residual if arm == "toward_failure" else
                    [-x for x in residual])
                episode = retry(agent, tokenizer, game, fields,
                                args.device, adjustment)
                row["retries"][arm] = episode
                if (episode["status"] != "complete" or
                        episode["initial_observation"] !=
                        first["initial_observation"]):
                    report["failures"].append({"index": binding["index"],
                        "arm": arm, "episode": episode})
                    break
        report["games"].append(row)
        report["summary"] = {"n": len(report["games"]),
            "first_wins": sum(x["first_reward"] for x in report["games"]),
            "first_failures": sum(not x["first_reward"]
                                  for x in report["games"]),
            "two_attempt_wins": {arm: sum(
                x["first_reward"] or x["retries"].get(arm, {}).get(
                    "reward", 0) for x in report["games"]) for arm in ARMS},
            "failures": len(report["failures"])}
        save(args.output, report)
        print(json.dumps(report["summary"]), flush=True)
        if report["failures"]:
            raise RuntimeError("Official retry episode failure recorded")


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
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/alfworld_failure_retry_v31b_dev24_20261009.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_failure_retry_v31b_dev24_20261009.json"))
    parser.add_argument("--max-relative", type=float, default=.25)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--gpu-fraction", type=float, default=.65)
    args = parser.parse_args()
    if not 0 < args.max_relative <= 1:
        parser.error("Invalid frozen residual bound")
    (prepare if args.command == "prepare" else evaluate)(args)
