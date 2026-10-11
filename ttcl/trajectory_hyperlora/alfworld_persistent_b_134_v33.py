"""From-empty, single-attempt persistent LoRA-B comparison on 134 games."""

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
from ttcl.trajectory_hyperlora.alfworld_success_gated_factors_v27 import episode_factor
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode


def targets(args):
    source = json.loads(args.target_review.read_text())
    if len(source["targets"]) != 134:
        raise ValueError("Changed 134 target count")
    result = []
    for index, old in enumerate(source["targets"]):
        game = old["game"]
        if ("/valid_unseen/" not in "/" + game or
                file_hash(args.data_root / game) != old["game_sha256"]):
            raise ValueError("Changed target game content or split")
        content = {"index": index, "game": game, "family": old["family"],
            "game_sha256": old["game_sha256"],
            "target_review_sha256": file_hash(args.target_review),
            "checkpoint_sha256": file_hash(args.checkpoint),
            "runner_sha256": file_hash(Path(__file__)),
            "actor_sha256": file_hash(Path(__file__).with_name(
                "alfworld_zero_shot_probe.py")),
            "factor_update_sha256": file_hash(Path(__file__).with_name(
                "alfworld_online_accumulate_lora_v1.py")),
            "model_config_sha256": file_hash(args.model / "config.json"),
            "mode": args.mode, "context_tokens": 2048,
            "budget": {"max_steps": 50, "max_new_tokens": 64,
                       "actor_history_turns": 2, "loop_guard_max": 2}}
        result.append({**content, "input_content_sha256": digest(content),
                       "reviewed_target": True})
    return result


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    rows = targets(args)
    save(args.review, {"protocol": "From-empty online 134 valid_unseen; own completed trajectories only; same target order and single-attempt budget; no future feedback at write time",
        "mode": args.mode, "targets": rows, "targets_sha256": digest(rows)})
    print(json.dumps({"reviewed_targets": len(rows), "mode": args.mode}))


def evaluate(args):
    rows = targets(args)
    review = json.loads(args.review.read_text())
    if (review["mode"] != args.mode or review["targets"] != rows or
            review["targets_sha256"] != digest(rows)):
        raise ValueError("Changed reviewed targets")
    if args.output.exists():
        raise FileExistsError(args.output)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != "contextual" or not agent.task_conditioned or
            agent.task_context_scope != "current" or
            agent.task_pair_pooling != "mean"):
        raise ValueError("Changed frozen hypernetwork")
    factors = None
    updates = 0
    history = []
    report = {"protocol": review["protocol"], "mode": args.mode,
        "review_sha256": file_hash(args.review),
        "target_review_sha256": file_hash(args.target_review),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "runner_sha256": file_hash(Path(__file__)),
        "factor_update_sha256": rows[0]["factor_update_sha256"],
        "games": [], "failures": []}
    for row in rows:
        before = factors_hash(factors)
        episode = run_episode(agent, tokenizer, args.data_root / row["game"],
            {}, adapter=False, fixed_adapter=factors, device=args.device,
            max_steps=50, max_new_tokens=64, constrain_actions=True,
            actor_history_turns=2, loop_guard_max=2)
        item = {"index": row["index"], "game": row["game"],
            "family": row["family"],
            "input_content_sha256": row["input_content_sha256"],
            "prior_episodes_sha256": digest(history),
            "factor_sha256_before": before,
            "episode": episode, "memory_updated": False}
        if episode["status"] != "complete":
            report["failures"].append({"index": row["index"],
                                       "status": episode["status"]})
        else:
            records = records_from_episode(episode)
            item["new_records_sha256"] = digest(records)
            if args.mode == "all_mean" or bool(episode["reward"]):
                increment, _, original, retained = episode_factor(
                    agent, tokenizer, episode, argparse.Namespace(
                        device=args.device, context_tokens=2048))
                factors = mean_factors(factors, increment, updates)
                updates += 1
                item.update(memory_updated=True,
                    source_tokens_original=original,
                    source_tokens_retained=retained,
                    increment_sha256=factors_hash(increment))
            history.append({"game": row["game"], "reward": episode["reward"],
                            "records_sha256": digest(records)})
        item["factor_sha256_after"] = factors_hash(factors)
        item["update_count_after"] = updates
        report["games"].append(item)
        report["summary"] = {"n": len(report["games"]),
            "wins": sum(x["episode"]["reward"] for x in report["games"]),
            "updates": updates, "failures": len(report["failures"])}
        save(args.output, report)
        print(json.dumps(report["summary"]), flush=True)
        if report["failures"]:
            raise RuntimeError("Online episode failure recorded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--mode", required=True,
                        choices=("all_mean", "success_mean"))
    parser.add_argument("--target-review", type=Path, default=Path(
        "data/annotations/alf_memrl134_hyperlora_train_sources_reviewed_20261006.json"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.65)
    options = parser.parse_args()
    (prepare if options.command == "prepare" else evaluate)(options)
