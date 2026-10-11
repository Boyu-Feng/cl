"""Frozen v36 failure-memory rule on 48 different reviewed train tasks.

Two predetermined interleaved orders contain the same six families x eight
tasks. Baseline, running-mean failure residual, and latest failure residual
each use one official attempt per task. The actor and hypernetwork are frozen.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_failure_retry_residual_v31b import bounded_residual
from ttcl.trajectory_hyperlora.alfworld_success_gated_factors_v27 import episode_factor
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_source_fields
from ttcl.trajectory_hyperlora.evaluate_alfworld_failure_retry_v31b import retry, source_factors
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import (
    ARMS, bounded_readout, update_state, save,
)


def ordered_pairs(args):
    source = json.loads(args.pairs.read_text())
    original = [x for x in source["pairs"] if x["split"] == "train"]
    dev = [x for x in source["pairs"] if x["split"] == "dev"]
    if len(original) != 48 or len(dev) != 24:
        raise ValueError("Changed reviewed 48/24 partition")
    if set(x["target_game"] for x in original) & set(x["target_game"] for x in dev):
        raise ValueError("Train/development target overlap")
    families = defaultdict(list)
    for pair in original:
        families[pair["family"]].append(pair)
    if len(families) != 6 or any(len(x) != 8 for x in families.values()):
        raise ValueError("Changed six-family training pool")
    forward = [families[f][j] for j in range(8) for f in sorted(families)]
    return forward if args.order == "interleave" else list(reversed(forward))


def bindings(args, pairs):
    previous_audit = json.loads(args.v36_audit.read_text())
    if (previous_audit.get("audit") != "passed" or
            previous_audit.get("report_sha256") != file_hash(args.v36_report) or
            previous_audit["summary"]["n"] != 24):
        raise ValueError("Changed v36 development rule source")
    fixed = {"method": "v37 frozen v36 failure-memory rule on other reviewed train tasks",
        "order": args.order, "pairs_sha256": file_hash(args.pairs),
        "v36_report_sha256": file_hash(args.v36_report),
        "v36_audit_sha256": file_hash(args.v36_audit),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "actor_sha256": file_hash(Path(__file__).with_name("alfworld_zero_shot_probe.py")),
        "v36_rule_sha256": file_hash(Path(__file__).with_name("alfworld_persistent_failure_delta_v36.py")),
        "runner_sha256": file_hash(Path(__file__)),
        "selected_direction": "away_from_failure",
        "max_relative": .25, "max_steps": 50, "max_new_tokens": 64,
        "actor_history_turns": 2, "loop_guard_max": 2,
        "arms": ["baseline", *ARMS]}
    targets = []
    for i, pair in enumerate(pairs):
        game = pair["target_game"]
        if "/train/" not in "/" + game:
            raise ValueError("Nontraining target")
        content = {"index": i, "game": game,
            "game_sha256": file_hash(args.data_root / game),
            "pair_input_content_sha256": pair["input_content_sha256"],
            "expert_source_sha256": pair["source_records_sha256"],
            "checkpoint_sha256": fixed["checkpoint_sha256"],
            "runner_sha256": fixed["runner_sha256"],
            "order": args.order}
        if content["game_sha256"] != pair["target_game_sha256"]:
            raise ValueError("Changed game content")
        targets.append({**content, "input_content_sha256": digest(content),
                        "reviewed_target": True})
    return fixed, targets


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    pairs = ordered_pairs(args)
    fixed, targets = bindings(args, pairs)
    save(args.review, {"protocol": "Two fixed orders before any v37 reward: 48 nonoverlapping reviewed official train tasks, six families x eight; same v36 update and readout, independent own-history arms, one attempt per task, no order-specific tuning",
        **fixed, "targets": targets, "targets_sha256": digest(targets)})
    print(json.dumps({"order": args.order, "reviewed_targets": len(targets)}))


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    pairs = ordered_pairs(args)
    fixed, targets = bindings(args, pairs)
    review = json.loads(args.review.read_text())
    if (review.get("targets") != targets or
            review.get("targets_sha256") != digest(targets) or
            any(review.get(k) != v for k, v in fixed.items())):
        raise ValueError("Changed v37 reviewed targets")
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    state = {arm: None for arm in ARMS}
    counts = {arm: 0 for arm in ARMS}
    report = {"protocol": review["protocol"], "review_sha256": file_hash(args.review),
        **fixed, "rows": [], "failures": []}
    for i, (pair, binding) in enumerate(zip(pairs, targets, strict=True)):
        fields = contextual_source_fields(agent, tokenizer,
            pair["source_records"], args.device, 2048, pooling="both")
        game = args.data_root / binding["game"]
        baseline = retry(agent, tokenizer, game, fields, args.device)
        row = {"index": i, "game": binding["game"],
            "input_content_sha256": binding["input_content_sha256"],
            "baseline": baseline, "arms": {}}
        if baseline["status"] != "complete":
            report["failures"].append({"index": i, "arm": "baseline",
                "status": baseline["status"]})
        else:
            initial = baseline["initial_observation"]
            expert = source_factors(agent, tokenizer, fields, initial, args.device)
            row["expert_factor_sha256"] = factors_hash(expert)
            for arm in ARMS:
                before = factors_hash(state[arm])
                applied, read_info = bounded_readout(agent, state[arm], expert)
                episode = retry(agent, tokenizer, game, fields, args.device, applied)
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
                    entry.update(write_kind="own_failure",
                        failed_factor_sha256=factors_hash(failed),
                        delta_sha256=factors_hash(delta), delta_info=info,
                        failed_source_tokens_original=original,
                        failed_source_tokens_retained=retained)
                entry.update(state_sha256_after=factors_hash(state[arm]),
                             failure_updates_after=counts[arm])
                row["arms"][arm] = entry
        report["rows"].append(row)
        report["summary"] = {"n": len(report["rows"]),
            "baseline": sum(x["baseline"]["reward"] for x in report["rows"]),
            **{arm: sum(x["arms"].get(arm, {}).get("episode", {}).get("reward", 0)
                   for x in report["rows"]) for arm in ARMS},
            "failure_updates": dict(counts), "failures": len(report["failures"])}
        save(args.output, report)
        print(json.dumps({"order": args.order, **report["summary"]}), flush=True)
        if report["failures"]:
            raise RuntimeError("Incomplete official v37 episode")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--order", required=True, choices=("interleave", "reverse"))
    parser.add_argument("--pairs", type=Path, default=Path("data/annotations/alf_reward_router_train72_reviewed_20261007.json"))
    parser.add_argument("--v36-report", type=Path, default=Path("results/trajectory_hyperlora/alfworld_persistent_failure_delta_v36_dev24_20261009.json"))
    parser.add_argument("--v36-audit", type=Path, default=Path("results/trajectory_hyperlora/alfworld_persistent_failure_delta_v36_dev24_audit_20261009.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--model", type=Path, default=Path("current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--data-root", type=Path, default=Path("ttcl/data/alfworld_delta"))
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.65)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else evaluate)(args)


if __name__ == "__main__":
    main()
