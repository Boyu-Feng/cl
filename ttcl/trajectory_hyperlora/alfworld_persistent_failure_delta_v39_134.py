"""Frozen running-mean own-failure residual on exposed official unseen 134."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_failure_retry_residual_v31b import bounded_residual
from ttcl.trajectory_hyperlora.alfworld_success_gated_factors_v27 import episode_factor
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_source_fields
from ttcl.trajectory_hyperlora.evaluate_alfworld_failure_retry_v31b import retry, source_factors
from ttcl.trajectory_hyperlora.evaluate_alfworld_failure_retry_v32_134 import checked_input
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import bounded_readout, update_state, save


def bound(args):
    targets, first, _, _ = checked_input(args)
    audit = json.loads(args.v32_audit.read_text())
    if (audit.get("audit") != "passed" or
            audit.get("report_sha256") != file_hash(args.v32_report)):
        raise ValueError("Changed audited same-task retry")
    fixed = {"method": "v39 frozen v36 running-mean failure residual on exposed unseen 134",
        "target_review_sha256": file_hash(args.target_review),
        "old_audit_sha256": file_hash(args.old_audit),
        "v32_report_sha256": file_hash(args.v32_report),
        "v32_audit_sha256": file_hash(args.v32_audit),
        "v36_rule_sha256": file_hash(Path(__file__).with_name("alfworld_persistent_failure_delta_v36.py")),
        "runner_sha256": file_hash(Path(__file__)),
        "actor_sha256": file_hash(Path(__file__).with_name("alfworld_zero_shot_probe.py")),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "selected_direction": "away_from_failure", "update": "running_mean",
        "max_relative": .25, "max_steps": 50, "max_new_tokens": 64,
        "actor_history_turns": 2, "loop_guard_max": 2}
    bindings = []
    for i, (target, old) in enumerate(zip(targets, first, strict=True)):
        if "/valid_unseen/" not in "/" + target["game"]:
            raise ValueError("Non-unseen target")
        content = {"index": i, "game": target["game"],
            "game_sha256": target["game_sha256"],
            "expert_source_records_sha256": target["source_records_sha256"],
            "target_input_content_sha256": target["input_content_sha256"],
            "old_first_episode_sha256": digest(old["lora"]),
            "runner_sha256": fixed["runner_sha256"],
            "checkpoint_sha256": fixed["checkpoint_sha256"]}
        bindings.append({**content, "input_content_sha256": digest(content),
                         "reviewed_target": True})
    return targets, first, fixed, bindings


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    _, _, fixed, bindings = bound(args)
    save(args.review, {"protocol": "Exposed official valid_unseen 134 in audited old order; one attempt per task; frozen actor and hypernetwork; after each own completed failure write bounded away-from-failure residual into running mean; read with 25-percent effective-matrix cap on later tasks; descriptive, not fresh held-out test",
        **fixed, "targets": bindings, "targets_sha256": digest(bindings)})
    print(json.dumps({"reviewed_targets": len(bindings)}), flush=True)


def evaluate(args):
    targets, first, fixed, bindings = bound(args)
    review = json.loads(args.review.read_text())
    if (review.get("targets") != bindings or
            review.get("targets_sha256") != digest(bindings) or
            any(review.get(k) != v for k, v in fixed.items())):
        raise ValueError("Changed reviewed inputs")
    if args.output.exists():
        raise FileExistsError(args.output)
    agent, tokenizer = load_agent(args.model, args.checkpoint, args.device, args.gpu_fraction)
    if (agent.encoder_kind != "contextual" or not agent.task_conditioned or
            agent.task_context_scope != "current" or agent.task_pair_pooling != "mean"):
        raise ValueError("Changed contextual hypernetwork")
    state = None
    count = 0
    report = {"protocol": review["protocol"], "review_sha256": file_hash(args.review),
              **fixed, "rows": [], "failures": []}
    for i, (target, old, binding) in enumerate(zip(targets, first, bindings, strict=True)):
        fields = contextual_source_fields(agent, tokenizer, target["source_records"],
                                           args.device, 2048, pooling="both")
        initial = old["lora"]["initial_observation"]
        expert = source_factors(agent, tokenizer, fields, initial, args.device)
        applied, read_info = bounded_readout(agent, state, expert)
        episode = retry(agent, tokenizer, args.data_root / binding["game"],
                        fields, args.device, applied)
        row = {"index": i, "game": binding["game"],
            "input_content_sha256": binding["input_content_sha256"],
            "old_first_reward": old["lora"]["reward"],
            "old_first_episode_sha256": binding["old_first_episode_sha256"],
            "expert_factor_sha256": factors_hash(expert),
            "state_sha256_before": factors_hash(state),
            "applied_sha256": factors_hash(applied), "read_info": read_info,
            "failure_updates_before": count, "episode": episode,
            "write_kind": "none"}
        if episode["status"] != "complete" or episode["initial_observation"] != initial:
            report["failures"].append({"index": i, "status": episode["status"]})
        elif not bool(episode["reward"]):
            failed, _, original, retained = episode_factor(agent, tokenizer,
                episode, argparse.Namespace(device=args.device, context_tokens=2048))
            toward, info = bounded_residual(agent, expert, failed, max_relative=.25)
            delta = [-x for x in toward]
            state = update_state(state, delta, count, "failure_mean")
            count += 1
            row.update(write_kind="own_failure", failed_factor_sha256=factors_hash(failed),
                delta_sha256=factors_hash(delta), delta_info=info,
                failed_source_tokens_original=original,
                failed_source_tokens_retained=retained)
        row.update(state_sha256_after=factors_hash(state), failure_updates_after=count)
        report["rows"].append(row)
        report["summary"] = {"n": len(report["rows"]),
            "old_first_wins": sum(x["old_first_reward"] for x in report["rows"]),
            "failure_mean_wins": sum(x["episode"]["reward"] for x in report["rows"]),
            "failure_updates": count, "failures": len(report["failures"])}
        save(args.output, report)
        print(json.dumps(report["summary"]), flush=True)
        if report["failures"]:
            raise RuntimeError("Incomplete official episode recorded")


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "evaluate"))
    defaults = {
        "memrl_plan": "results/memrl_comparison/20260928_budgeted/plan.json",
        "source_candidates": "results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json",
        "source_review": "data/annotations/alf_sibling_train600_reviewed_20261006.json",
        "retry_candidates": "results/trajectory_hyperlora/alf_same_game_retry_candidates_20261006.json",
        "retry_review": "data/annotations/alf_same_game_retry_reviewed_20261006.json",
        "all_source_review": "data/annotations/alf_expert_first_attempts_reviewed_20261006.json",
        "target_review": "data/annotations/alf_memrl134_hyperlora_train_sources_reviewed_20261006.json",
        "old_audit": "results/trajectory_hyperlora/alf_parent_loop2_official_audited_20261007.json",
        "dev_report": "results/trajectory_hyperlora/alfworld_failure_retry_v31b_dev24_20261009.json",
        "dev_audit": "results/trajectory_hyperlora/alfworld_failure_retry_v31b_dev24_audit_20261009.json",
        "placebo_report": "results/trajectory_hyperlora/alfworld_failure_retry_placebo_v31c_dev24_20261009.json",
        "placebo_audit": "results/trajectory_hyperlora/alfworld_failure_retry_placebo_v31c_dev24_audit_20261009.json",
        "v32_report": "results/trajectory_hyperlora/alfworld_failure_retry_v32_134_20261009.json",
        "v32_audit": "results/trajectory_hyperlora/alfworld_failure_retry_v32_134_audit_20261009.json",
        "checkpoint": "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt",
        "model": "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507",
        "data_root": "ttcl/data/alfworld_delta",
        "review": "data/annotations/alfworld_persistent_failure_delta_v39_134_20261010.json",
        "output": "results/trajectory_hyperlora/alfworld_persistent_failure_delta_v39_134_20261010.json"}
    for name, value in defaults.items():
        p.add_argument("--" + name.replace("_", "-"), type=Path, default=Path(value))
    p.add_argument("--first-parts", type=Path, nargs=4, default=[Path(
        f"results/trajectory_hyperlora/alf_parent_loop2_official_{s}_{n}_20261007.json")
        for s, n in ((0, 34), (34, 34), (68, 33), (101, 33))])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gpu-fraction", type=float, default=.65)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    (prepare if args.command == "prepare" else evaluate)(args)
