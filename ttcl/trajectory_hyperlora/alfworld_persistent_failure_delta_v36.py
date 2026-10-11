"""Online persistent failure residuals on an exposed ALFWorld train cohort.

Both arms run their own single-attempt sequence. On a failure, the frozen
hypernetwork encodes that arm's own failed trajectory and the same task's
reviewed expert success source. The selected away-from-failure residual is
stored as either a running mean or only the latest residual. The state is
read on later tasks; both writes and reads are norm bounded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_failure_retry_residual_v31b import (
    bounded_residual, effective_norm,
)
from ttcl.trajectory_hyperlora.alfworld_success_gated_factors_v27 import episode_factor
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_source_fields
from ttcl.trajectory_hyperlora.evaluate_alfworld_failure_retry_v31b import (
    retry, source_factors,
)
from ttcl.trajectory_hyperlora.alfworld_retry_delta_next_v35 import read_inputs


ARMS = ("failure_mean", "failure_latest")


def bounded_readout(agent, state, expert, cap=.25):
    if state is None:
        return None, {"stored_effective_norm": 0., "applied_effective_norm": 0.,
                      "reference_effective_norm": effective_norm(agent, expert),
                      "read_scale": 0.}
    reference = effective_norm(agent, expert)
    stored = effective_norm(agent, state)
    scale = min(1., cap * reference / stored) if stored else 0.
    result = [scale * x for x in state]
    return result, {"stored_effective_norm": stored,
        "applied_effective_norm": effective_norm(agent, result),
        "reference_effective_norm": reference, "read_scale": scale}


def update_state(old, delta, count, arm):
    if arm == "failure_latest" or old is None:
        return [x.detach().cpu().float().clone() for x in delta]
    if arm != "failure_mean" or count < 1 or len(old) != len(delta):
        raise ValueError("Invalid accumulated failure state")
    return [(count * a + b) / (count + 1)
            for a, b in zip(old, delta, strict=True)]


def bound_targets(args, pairs, first, prior):
    fixed = {"method": "v36 own-failure persistent residual mean versus latest",
        "pairs_sha256": file_hash(args.pairs),
        "first_report_sha256": file_hash(args.first_report),
        "retry_report_sha256": file_hash(args.retry_report),
        "retry_audit_sha256": file_hash(args.retry_audit),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "runner_sha256": file_hash(Path(__file__)),
        "actor_sha256": prior["actor_sha256"],
        "residual_sha256": prior["residual_sha256"],
        "selected_direction": "away_from_failure",
        "max_relative": .25, "max_steps": 50, "max_new_tokens": 64,
        "actor_history_turns": 2, "loop_guard_max": 2,
        "arms": list(ARMS)}
    targets = []
    for i, (pair, row) in enumerate(zip(pairs, first["games"], strict=True)):
        if "/train/" not in "/" + row["game"]:
            raise ValueError("Nontraining game")
        content = {"index": i, "game": row["game"],
            "game_sha256": file_hash(args.data_root / row["game"]),
            "expert_source_sha256": pair["source_records_sha256"],
            "baseline_episode_sha256": digest(row["new"]),
            "initial_observation_sha256": digest(row["new"]["initial_observation"]),
            "runner_sha256": fixed["runner_sha256"],
            "checkpoint_sha256": fixed["checkpoint_sha256"]}
        if content["game_sha256"] != pair["target_game_sha256"]:
            raise ValueError("Changed task content")
        targets.append({**content, "input_content_sha256": digest(content),
                        "reviewed_target": True})
    return fixed, targets


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    pairs, first, prior = read_inputs(args)
    fixed, targets = bound_targets(args, pairs, first, prior)
    save(args.review, {"protocol": "Exposed official train development cohort, single attempt per arm and task; only own completed failure changes persistent residual; next-task use with 25-percent effective-matrix cap; baseline old first episodes audited by v31b",
        **fixed, "targets": targets, "targets_sha256": digest(targets)})
    print(json.dumps({"reviewed_targets": len(targets)}))


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    pairs, first, prior = read_inputs(args)
    fixed, targets = bound_targets(args, pairs, first, prior)
    review = json.loads(args.review.read_text())
    if (review.get("targets") != targets or
            review.get("targets_sha256") != digest(targets) or
            any(review.get(k) != v for k, v in fixed.items())):
        raise ValueError("Changed v36 review")
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    state = {arm: None for arm in ARMS}
    counts = {arm: 0 for arm in ARMS}
    report = {"protocol": review["protocol"], "review_sha256": file_hash(args.review),
        **fixed, "rows": [], "failures": []}
    for i, (pair, binding, old) in enumerate(zip(pairs, targets,
                                                  first["games"], strict=True)):
        fields = contextual_source_fields(agent, tokenizer, pair["source_records"],
                                           args.device, 2048, pooling="both")
        initial = old["new"]["initial_observation"]
        expert = source_factors(agent, tokenizer, fields, initial, args.device)
        row = {"index": i, "game": binding["game"],
            "input_content_sha256": binding["input_content_sha256"],
            "expert_factor_sha256": factors_hash(expert),
            "baseline_reward": old["new"]["reward"],
            "baseline_episode_sha256": digest(old["new"]), "arms": {}}
        for arm in ARMS:
            before = factors_hash(state[arm])
            applied, read_info = bounded_readout(agent, state[arm], expert)
            episode = retry(agent, tokenizer, args.data_root / binding["game"],
                            fields, args.device, applied)
            entry = {"episode": episode, "state_sha256_before": before,
                "applied_sha256": factors_hash(applied),
                "read_info": read_info, "write_kind": "none",
                "failure_updates_before": counts[arm]}
            if (episode["status"] != "complete" or
                    episode["initial_observation"] != initial):
                report["failures"].append({"index": i, "arm": arm,
                    "status": episode["status"]})
            elif not bool(episode["reward"]):
                failed, _, original, retained = episode_factor(
                    agent, tokenizer, episode,
                    argparse.Namespace(device=args.device, context_tokens=2048))
                toward, info = bounded_residual(agent, expert, failed,
                                                 max_relative=.25)
                delta = [-x for x in toward]
                state[arm] = update_state(state[arm], delta, counts[arm], arm)
                counts[arm] += 1
                entry.update(write_kind="own_failure", failed_factor_sha256=factors_hash(failed),
                    delta_sha256=factors_hash(delta), delta_info=info,
                    failed_source_tokens_original=original,
                    failed_source_tokens_retained=retained)
            entry.update(state_sha256_after=factors_hash(state[arm]),
                         failure_updates_after=counts[arm])
            row["arms"][arm] = entry
        report["rows"].append(row)
        report["summary"] = {"n": len(report["rows"]),
            "baseline": sum(x["baseline_reward"] for x in report["rows"]),
            **{arm: sum(x["arms"][arm]["episode"]["reward"]
                   for x in report["rows"]) for arm in ARMS},
            "failure_updates": dict(counts), "failures": len(report["failures"])}
        save(args.output, report)
        print(json.dumps(report["summary"]), flush=True)
        if report["failures"]:
            raise RuntimeError("Incomplete official online failure-memory episode")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--pairs", type=Path, default=Path("data/annotations/alf_reward_router_train72_reviewed_20261007.json"))
    parser.add_argument("--first-report", type=Path, default=Path("results/trajectory_hyperlora/alf_parent_loop2_dev24_20261007.json"))
    parser.add_argument("--retry-report", type=Path, default=Path("results/trajectory_hyperlora/alfworld_failure_retry_v31b_dev24_20261009.json"))
    parser.add_argument("--retry-audit", type=Path, default=Path("results/trajectory_hyperlora/alfworld_failure_retry_v31b_dev24_audit_20261009.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--model", type=Path, default=Path("current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--data-root", type=Path, default=Path("ttcl/data/alfworld_delta"))
    parser.add_argument("--review", type=Path, default=Path("data/annotations/alfworld_persistent_failure_delta_v36_dev24_20261009.json"))
    parser.add_argument("--output", type=Path, default=Path("results/trajectory_hyperlora/alfworld_persistent_failure_delta_v36_dev24_20261009.json"))
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--gpu-fraction", type=float, default=.65)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else evaluate)(args)


if __name__ == "__main__":
    main()
