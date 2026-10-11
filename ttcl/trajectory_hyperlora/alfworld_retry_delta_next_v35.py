"""Carry the exact v31b/v32 failure residual to the next ALFWorld task.

The old dynamic expert-source hypernetwork remains frozen. For each failed
first attempt in the ordered 24-task train-domain development cohort, compute
the already selected away-from-failure residual and apply it to the next task's
own expert-conditioned dynamic LoRA. Compare with that task's audited old first
attempt. This is a local transfer test, not an on-policy sequence score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_failure_retry_residual_v31b import bounded_residual
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_success_gated_factors_v27 import episode_factor
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_source_fields
from ttcl.trajectory_hyperlora.evaluate_alfworld_failure_retry_v31b import source_factors, retry


def read_inputs(args):
    pairs = [x for x in json.loads(args.pairs.read_text())["pairs"] if x["split"] == "dev"]
    first = json.loads(args.first_report.read_text())
    prior = json.loads(args.retry_report.read_text())
    audit = json.loads(args.retry_audit.read_text())
    if (len(pairs) != 24 or len(first["games"]) != 24 or
            len(prior["games"]) != 24):
        raise ValueError("Changed dev cohort size")
    if (first["failures"] or prior["failures"] or
            first["reviewed_pairs_sha256"] != file_hash(args.pairs) or
            first["source_checkpoint_sha256"] != file_hash(args.checkpoint) or
            prior["checkpoint_sha256"] != file_hash(args.checkpoint) or
            prior["first_report_sha256"] != file_hash(args.first_report) or
            prior["actor_sha256"] != file_hash(Path(__file__).with_name("alfworld_zero_shot_probe.py")) or
            prior["residual_sha256"] != file_hash(Path(__file__).with_name("alfworld_failure_retry_residual_v31b.py")) or
            prior["max_relative"] != .25 or audit.get("audit") != "passed"):
        raise ValueError("Changed or unaudited retry lineage")
    for p, a, b in zip(pairs, first["games"], prior["games"], strict=True):
        if (p["target_game"] != a["game"] or a["game"] != b["game"] or
                p["source_records_sha256"] != a["source_records_sha256"] or
                digest(a["new"]) != b["first_episode_sha256"] or
                a["new"]["reward"] != b["first_reward"] or
                file_hash(args.data_root / a["game"]) != p["target_game_sha256"]):
            raise ValueError("Changed source pair or first episode")
    return pairs, first, prior


def bound_targets(args, pairs, first, prior):
    fixed = {"method": "exact selected v31b/v32 residual transferred to next train-domain task",
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
        "actor_history_turns": 2, "loop_guard_max": 2}
    targets = []
    for i in range(23):
        if bool(first["games"][i]["new"]["reward"]):
            continue
        old, nxt = first["games"][i:i+2]
        content = {"source_index": i, "target_index": i+1,
            "source_game": old["game"], "target_game": nxt["game"],
            "source_game_sha256": pairs[i]["target_game_sha256"],
            "target_game_sha256": pairs[i+1]["target_game_sha256"],
            "source_first_episode_sha256": digest(old["new"]),
            "target_baseline_episode_sha256": digest(nxt["new"]),
            "source_expert_records_sha256": pairs[i]["source_records_sha256"],
            "target_expert_records_sha256": pairs[i+1]["source_records_sha256"],
            "runner_sha256": fixed["runner_sha256"],
            "checkpoint_sha256": fixed["checkpoint_sha256"]}
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
    save(args.review, {"protocol": "Fresh content-bound next-task targets from exposed official train development tasks; exact prior retry-selected residual; local paired transfer only",
        **fixed, "targets": targets, "targets_sha256": digest(targets)})
    print(json.dumps({"reviewed_targets": len(targets)}))


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    pairs, first, prior = read_inputs(args)
    fixed, targets = bound_targets(args, pairs, first, prior)
    review = json.loads(args.review.read_text())
    if any(review.get(k) != v for k, v in fixed.items()) or review.get("targets") != targets:
        raise ValueError("Changed v35 target bindings")
    agent, tokenizer = load_agent(args.model, args.checkpoint, args.device, args.gpu_fraction)
    report = {"protocol": review["protocol"], "review_sha256": file_hash(args.review),
        **fixed, "rows": [], "failures": []}
    for binding in targets:
        i = binding["source_index"]
        current, nxt = first["games"][i:i+2]
        source_fields = contextual_source_fields(agent, tokenizer,
            pairs[i]["source_records"], args.device, 2048, pooling="both")
        expert = source_factors(agent, tokenizer, source_fields,
            current["new"]["initial_observation"], args.device)
        failed, _, _, _ = episode_factor(agent, tokenizer, current["new"],
                                         argparse.Namespace(device=args.device,
                                                            context_tokens=2048))
        toward, info = bounded_residual(agent, expert, failed, max_relative=.25)
        away = [-x for x in toward]
        if (factors_hash(expert) != prior["games"][i]["expert_factor_sha256"] or
                factors_hash(failed) != prior["games"][i]["failed_factor_sha256"] or
                factors_hash(toward) != prior["games"][i]["residual_factor_sha256"]):
            raise ValueError(f"Source residual changed at {i}")
        target_fields = contextual_source_fields(agent, tokenizer,
            pairs[i+1]["source_records"], args.device, 2048, pooling="both")
        outcome = retry(agent, tokenizer, args.data_root / nxt["game"],
                        target_fields, args.device, away)
        row = {"source_index": i, "target_index": i+1,
            "input_content_sha256": binding["input_content_sha256"],
            "residual_sha256": factors_hash(away), "residual_info": info,
            "baseline_reward": nxt["new"]["reward"],
            "baseline_episode_sha256": digest(nxt["new"]),
            "transfer_episode": outcome}
        if (outcome["status"] != "complete" or outcome["initial_observation"] !=
                nxt["new"]["initial_observation"]):
            report["failures"].append({"source_index": i,
                "status": outcome["status"]})
        report["rows"].append(row)
        save(args.output, report)
        print(json.dumps({"source_index": i, "rows": len(report["rows"]),
                          "failures": len(report["failures"])}), flush=True)
        if report["failures"]:
            raise RuntimeError("Incomplete transferred residual episode")
    report["summary"] = {"pairs": len(report["rows"]),
        "baseline": sum(x["baseline_reward"] for x in report["rows"]),
        "transfer": sum(x["transfer_episode"]["reward"] for x in report["rows"]),
        "failures": len(report["failures"])}
    save(args.output, report)
    print(json.dumps(report["summary"]), flush=True)


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
    parser.add_argument("--review", type=Path, default=Path("data/annotations/alfworld_retry_delta_next_v35_dev24_20261009.json"))
    parser.add_argument("--output", type=Path, default=Path("results/trajectory_hyperlora/alfworld_retry_delta_next_v35_dev24_20261009.json"))
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--gpu-fraction", type=float, default=.65)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else evaluate)(args)


if __name__ == "__main__":
    main()
