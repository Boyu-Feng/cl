"""Equal-budget failed-retry test on the old 68/134 ALFWorld method.

First attempts are the exact previously audited LoRA-plus-action-guard run.
Only failed targets are rerun, with static, own-failure, and matched-random
LoRA residuals. This reused 134-task set is descriptive, not independent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_failure_retry_residual_v31b import (
    bounded_residual,
)
from ttcl.trajectory_hyperlora.alfworld_failure_retry_placebo_v31c import (
    matched_random_residual,
)
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_memrl134_transfer import checked_targets
from ttcl.trajectory_hyperlora.alfworld_success_gated_factors_v27 import episode_factor
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_source_fields
from ttcl.trajectory_hyperlora.evaluate_alfworld_failure_retry_v31b import (
    retry, source_factors,
)


ARMS = ("static_retry", "away_from_failure", "matched_random")


def checked_input(args):
    targets = checked_targets(args)
    if len(targets) != 134:
        raise ValueError("Changed official ALFWorld target count")
    old_audit = json.loads(args.old_audit.read_text())
    dev_audit = json.loads(args.dev_audit.read_text())
    placebo_audit = json.loads(args.placebo_audit.read_text())
    dev = json.loads(args.dev_report.read_text())
    placebo = json.loads(args.placebo_report.read_text())
    if (old_audit["summary"]["overall"]["old_lora"] != 68 or
            old_audit["target_review_sha256"] != file_hash(args.target_review) or
            old_audit["old_checkpoint_sha256"] != file_hash(args.checkpoint) or
            dev_audit["audit"] != "passed" or
            dev_audit["report_sha256"] != file_hash(args.dev_report) or
            placebo_audit["audit"] != "passed" or
            placebo_audit["report_sha256"] != file_hash(args.placebo_report) or
            dev["summary"]["two_attempt_wins"] != {
                "static_retry": 11.0,
                "toward_failure": 11.0,
                "away_from_failure": 13.0} or
            placebo["summary"]["failed_targets"] != 13):
        raise ValueError("Changed development selection or old 134 audit")
    first = []
    for path in args.first_parts:
        if (str(path) not in old_audit["part_sha256"] or
                file_hash(path) != old_audit["part_sha256"][str(path)]):
            raise ValueError("Changed audited old 134 part")
        part = json.loads(path.read_text())
        if (part["target_review_sha256"] != file_hash(args.target_review) or
                part["checkpoint_sha256"] != file_hash(args.checkpoint) or
                part["offset"] != len(first) or
                part["loop_guard_max"] != 2 or
                (part["max_steps"], part["max_new_tokens"],
                 part["actor_history_turns"]) != (50, 64, 2) or
                part["failures"]):
            raise ValueError("Changed old 134 first-attempt protocol")
        first.extend(part["games"])
    if len(first) != 134 or sum(x["lora"]["reward"] for x in first) != 68:
        raise ValueError("Incomplete audited first attempts")
    fixed = {"method": "我的在线更新 v32 old 68/134 failure retry",
        "target_review_sha256": file_hash(args.target_review),
        "old_audit_sha256": file_hash(args.old_audit),
        "first_part_sha256": {str(path): file_hash(path)
                              for path in args.first_parts},
        "dev_audit_sha256": file_hash(args.dev_audit),
        "placebo_audit_sha256": file_hash(args.placebo_audit),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "runner_sha256": file_hash(Path(__file__)),
        "actor_sha256": file_hash(Path(__file__).with_name(
            "alfworld_zero_shot_probe.py")),
        "residual_sha256": file_hash(Path(__file__).with_name(
            "alfworld_failure_retry_residual_v31b.py")),
        "placebo_sha256": file_hash(Path(__file__).with_name(
            "alfworld_failure_retry_placebo_v31c.py")),
        "dev_runner_sha256": file_hash(Path(__file__).with_name(
            "evaluate_alfworld_failure_retry_v31b.py")),
        "max_relative": .25, "max_steps_per_attempt": 50,
        "max_new_tokens": 64, "history_turns": 2,
        "repeat_limit": 2, "failure_source_tokens": 2048}
    bindings = []
    for i, (target, row) in enumerate(zip(targets, first, strict=True)):
        if (row["game"] != target["game"] or
                row["target_input_content_sha256"] !=
                    target["input_content_sha256"] or
                row["source_records_sha256"] !=
                    target["source_records_sha256"] or
                row["lora"]["status"] != "complete" or
                row["lora"]["steps"] > 50 or
                file_hash(args.data_root / target["game"]) !=
                    target["game_sha256"]):
            raise ValueError("Changed official target or old first episode")
        content = {"index": i, "game": target["game"],
            "game_sha256": target["game_sha256"],
            "expert_source_records_sha256":
                target["source_records_sha256"],
            "target_input_content_sha256":
                target["input_content_sha256"],
            "first_episode_sha256": digest(row["lora"]),
            "old_audit_sha256": fixed["old_audit_sha256"],
            "runner_sha256": fixed["runner_sha256"],
            "checkpoint_sha256": fixed["checkpoint_sha256"]}
        bindings.append({**content,
            "input_content_sha256": digest(content),
            "reviewed_target": True})
    return targets, first, fixed, bindings


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    temp.replace(path)


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    _, first, fixed, bindings = checked_input(args)
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps({"protocol":
        "Fresh content-bound retries on already exposed 134 official valid_unseen ALFWorld tasks; first 68/134 old checkpoint and action guard unchanged; equal second-attempt budget; not independent test",
        **fixed, "targets": bindings,
        "targets_sha256": digest(bindings)}, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"reviewed_targets": len(bindings),
        "first_failures": sum(not bool(x["lora"]["reward"]) for x in first)}),
        flush=True)


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    targets, first, fixed, bindings = checked_input(args)
    review = json.loads(args.review.read_text())
    if (review["targets"] != bindings or
            review["targets_sha256"] != digest(bindings) or
            any(review[k] != v for k, v in fixed.items())):
        raise ValueError("Changed official retry review")
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != "contextual" or not agent.task_conditioned or
            agent.task_context_scope != "current" or
            agent.task_pair_pooling != "mean"):
        raise ValueError("Changed old contextual hypernetwork")
    report = {"protocol":
        "Old 68/134 one-attempt LoRA and two-repeat action guard unchanged; after failure one equal-budget retry using static, away-from-own-failure, or matched-random B residual; exposed valid_unseen, descriptive only",
        "review_sha256": file_hash(args.review), **fixed,
        "arms": ARMS, "games": [], "failures": []}
    for target, first_row, binding in zip(targets, first, bindings, strict=True):
        old = first_row["lora"]
        row = {"index": binding["index"], "game": target["game"],
            "input_content_sha256": binding["input_content_sha256"],
            "first_episode_sha256": binding["first_episode_sha256"],
            "first_reward": old["reward"], "retries": {}}
        if not bool(old["reward"]):
            fields = contextual_source_fields(agent, tokenizer,
                target["source_records"], args.device, 2048, pooling="both")
            expert = source_factors(agent, tokenizer, fields,
                                    old["initial_observation"], args.device)
            failed, _, original, retained = episode_factor(
                agent, tokenizer, old, argparse.Namespace(
                    device=args.device, context_tokens=2048))
            residual, info = bounded_residual(agent, expert, failed,
                                              max_relative=.25)
            seed = int(digest(["matched_random_v32", target["game"]])[:16], 16)
            random_residual = matched_random_residual(agent, expert,
                info["residual_effective_norm"], seed)
            row.update(expert_factor_sha256=factors_hash(expert),
                failed_factor_sha256=factors_hash(failed),
                residual_factor_sha256=factors_hash(residual),
                random_residual_sha256=factors_hash(random_residual),
                random_seed=seed, residual_info=info,
                failed_source_tokens_original=original,
                failed_source_tokens_retained=retained)
            adjustments = {"static_retry": None,
                "away_from_failure": [-x for x in residual],
                "matched_random": random_residual}
            for arm in ARMS:
                episode = retry(agent, tokenizer,
                    args.data_root / target["game"], fields, args.device,
                    adjustments[arm])
                row["retries"][arm] = episode
                if (episode["status"] != "complete" or
                        episode["initial_observation"] !=
                        old["initial_observation"]):
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
    parser.add_argument("--first-parts", type=Path, nargs=4, default=[Path(
        "results/trajectory_hyperlora/alf_parent_loop2_official_0_34_20261007.json"), Path(
        "results/trajectory_hyperlora/alf_parent_loop2_official_34_34_20261007.json"), Path(
        "results/trajectory_hyperlora/alf_parent_loop2_official_68_33_20261007.json"), Path(
        "results/trajectory_hyperlora/alf_parent_loop2_official_101_33_20261007.json")])
    parser.add_argument("--old-audit", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_parent_loop2_official_audited_20261007.json"))
    parser.add_argument("--dev-report", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_failure_retry_v31b_dev24_20261009.json"))
    parser.add_argument("--dev-audit", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_failure_retry_v31b_dev24_audit_20261009.json"))
    parser.add_argument("--placebo-report", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_failure_retry_placebo_v31c_dev24_20261009.json"))
    parser.add_argument("--placebo-audit", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_failure_retry_placebo_v31c_dev24_audit_20261009.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/alfworld_failure_retry_v32_134_20261009.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_failure_retry_v32_134_20261009.json"))
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--gpu-fraction", type=float, default=.65)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else evaluate)(args)
