"""Train-domain pilot: learn from own success and paired failed trajectories.

Three policies run from empty persistent B on the same official task order.
Every policy uses its own prior outcomes. A shared no-LoRA episode provides
the same-task successful contrast when a policy fails; no walkthrough is read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import (
    factors_hash, mean_factors,
)
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)
from ttcl.trajectory_hyperlora.alfworld_success_gated_factors_v27 import (
    episode_factor,
)
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import (
    load_agent, run_episode,
)
from ttcl.trajectory_hyperlora.contrastive_failure_factors_v30 import (
    contrastive_failure_update, factor_norm,
)


ARMS = ("success_only", "base_success", "failure_contrast")


def fixed_protocol(args):
    return {"method": "我的在线更新 v30 train-domain failure contrast pilot",
        "source_review_sha256": file_hash(args.source_review),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "runner_sha256": file_hash(Path(__file__)),
        "actor_sha256": file_hash(Path(__file__).with_name(
            "alfworld_zero_shot_probe.py")),
        "factor_helper_sha256": file_hash(Path(__file__).with_name(
            "alfworld_success_gated_factors_v27.py")),
        "update_sha256": file_hash(Path(__file__).with_name(
            "contrastive_failure_factors_v30.py")),
        "step_size": args.step_size, "relative_cap": args.relative_cap,
        "sequence": "train_seq6_6", "max_steps": 50,
        "max_new_tokens": 64, "history_turns": 2,
        "repeat_limit": 2, "context_tokens": args.context_tokens}


def target_bindings(args, fixed):
    source = json.loads(args.source_review.read_text())
    if (len(source["targets"]) != 18 or
            source["checkpoint_sha256"] != fixed["checkpoint_sha256"]):
        raise ValueError("Changed train sequence or checkpoint")
    rows = []
    for index, target in enumerate(source["targets"]):
        game = target["game"]
        if "/train/" not in "/" + game:
            raise ValueError("Nontraining target")
        content = {"index": index, "game": game,
            "game_sha256": file_hash(args.data_root / game),
            "source_review_sha256": fixed["source_review_sha256"],
            "runner_sha256": fixed["runner_sha256"],
            "checkpoint_sha256": fixed["checkpoint_sha256"]}
        if content["game_sha256"] != target["game_sha256"]:
            raise ValueError("Changed target content")
        rows.append({**content, "input_content_sha256": digest(content),
                     "reviewed_target": True})
    return rows


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    fixed = fixed_protocol(args)
    targets = target_bindings(args, fixed)
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps({"protocol":
        "New content-bound targets for own-history failure contrast on exposed official training order; no independent test claim",
        **fixed, "targets": targets, "targets_sha256": digest(targets)},
        ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"reviewed_targets": len(targets)}), flush=True)


def episode(agent, tokenizer, game, factors, args):
    return run_episode(agent, tokenizer, game, {}, adapter=False,
        fixed_adapter=factors, device=args.device, max_steps=50,
        max_new_tokens=64, constrain_actions=True,
        actor_history_turns=2, loop_guard_max=2)


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    fixed = fixed_protocol(args)
    targets = target_bindings(args, fixed)
    review = json.loads(args.review.read_text())
    if (review["targets"] != targets or
            review["targets_sha256"] != digest(targets) or
            any(review[k] != v for k, v in fixed.items())):
        raise ValueError("Changed v30 review")
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != "contextual" or
            agent.task_conditioned is not True or
            agent.task_context_scope != "current" or
            agent.task_pair_pooling != "mean"):
        raise ValueError("Changed frozen contextual hypernetwork")
    state = {arm: None for arm in ARMS}
    state_count = {arm: 0 for arm in ARMS}
    success_count = {arm: 0 for arm in ARMS}
    base_positive_count = {arm: 0 for arm in ARMS}
    contrast_count = {arm: 0 for arm in ARMS}
    history = {arm: [] for arm in ARMS}
    report = {"protocol":
        "我的在线更新 v30: own-history persistent B; successes use running mean; failed episodes only update when same-task no-LoRA replay wins, comparing successful and failed generated factors; three equal-budget policy arms",
        "review_sha256": file_hash(args.review), **fixed,
        "arms": ARMS, "games": [], "failures": []}
    for binding in targets:
        game = args.data_root / binding["game"]
        base = episode(agent, tokenizer, game, None, args)
        row = {"index": binding["index"], "game": binding["game"],
            "input_content_sha256": binding["input_content_sha256"],
            "base": base, "arms": {}}
        base_factor = None
        for arm in ARMS:
            before = factors_hash(state[arm])
            current = episode(agent, tokenizer, game, state[arm], args)
            entry = {"episode": current,
                "prior_episode_count": len(history[arm]),
                "prior_episodes_sha256": digest(history[arm]),
                "factor_sha256_before": before,
                "factor_l2_before": factor_norm(state[arm]) if state[arm] else 0.,
                "update_kind": "none"}
            if (base["status"] != "complete" or
                    current["status"] != "complete" or
                    current["initial_observation"] !=
                    base["initial_observation"]):
                report["failures"].append({"index": binding["index"],
                    "arm": arm, "error": "Incomplete or inconsistent task reset"})
                row["arms"][arm] = entry
                break
            records = records_from_episode(current)
            entry["records_sha256"] = digest(records)
            if bool(current["reward"]):
                generated, _, original, retained = episode_factor(
                    agent, tokenizer, current, args)
                state[arm] = mean_factors(state[arm], generated,
                                          state_count[arm])
                state_count[arm] += 1
                success_count[arm] += 1
                entry.update(update_kind="own_success",
                    source_tokens_original=original,
                    source_tokens_retained=retained,
                    generated_factor_l2=factor_norm(generated))
            elif bool(base["reward"]) and arm != "success_only":
                if base_factor is None:
                    base_factor, _, original, retained = episode_factor(
                        agent, tokenizer, base, args)
                    row["base_source_tokens_original"] = original
                    row["base_source_tokens_retained"] = retained
                    row["base_factor_l2"] = factor_norm(base_factor)
                if arm == "base_success":
                    state[arm] = mean_factors(state[arm], base_factor,
                        state_count[arm])
                    state_count[arm] += 1
                    base_positive_count[arm] += 1
                    entry["update_kind"] = "paired_base_success"
                else:
                    failed_factor, _, original, retained = episode_factor(
                        agent, tokenizer, current, args)
                    candidate, info = contrastive_failure_update(
                        state[arm], base_factor, failed_factor,
                        step_size=args.step_size,
                        relative_cap=args.relative_cap)
                    applied = info["applied_norm"] > 0
                    if applied:
                        state[arm] = candidate
                        state_count[arm] += 1
                        contrast_count[arm] += 1
                    entry.update(update_kind=("paired_failure_contrast"
                        if applied else "zero_contrast_abstain"),
                        source_tokens_original=original,
                        source_tokens_retained=retained,
                        failed_factor_l2=factor_norm(failed_factor),
                        contrast=info)
            history[arm].append({"game": binding["game"],
                "reward": current["reward"], "records": records})
            entry.update(factor_sha256_after=factors_hash(state[arm]),
                factor_l2_after=factor_norm(state[arm]) if state[arm] else 0.,
                success_count_after=success_count[arm],
                state_count_after=state_count[arm],
                base_positive_count_after=base_positive_count[arm],
                contrast_count_after=contrast_count[arm])
            row["arms"][arm] = entry
        report["games"].append(row)
        report["summary"] = {"n": len(report["games"]),
            "base": sum(r["base"].get("reward", 0) for r in report["games"]),
            **{arm: sum(r["arms"].get(arm, {}).get("episode", {}).get(
                "reward", 0) for r in report["games"]) for arm in ARMS},
            "own_success_updates": dict(success_count),
            "state_updates": dict(state_count),
            "paired_base_success_updates": dict(base_positive_count),
            "paired_failure_contrast_updates": dict(contrast_count),
            "failures": len(report["failures"])}
        save(args.output, report)
        print(json.dumps(report["summary"]), flush=True)
        if report["failures"]:
            raise RuntimeError("Official failure-contrast episode failed")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alfworld_online_context_persistent_train_seq6_6_reviewed_20261007.json"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/alfworld_failure_contrast_online_v30_train_seq6_6_20261009.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_failure_contrast_online_v30_train_seq6_6_20261009.json"))
    parser.add_argument("--step-size", type=float, default=.25)
    parser.add_argument("--relative-cap", type=float, default=.25)
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--gpu-fraction", type=float, default=.65)
    args = parser.parse_args()
    if args.context_tokens < 2 or not 0 < args.step_size <= 1 or not 0 < args.relative_cap <= 1:
        parser.error("Invalid frozen contrast protocol")
    (prepare if args.command == "prepare" else evaluate)(args)
