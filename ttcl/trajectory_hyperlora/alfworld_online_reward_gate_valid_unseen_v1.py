"""From-empty reward-gated online LoRA on frozen official valid_unseen games."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)
from ttcl.trajectory_hyperlora.alfworld_online_context_vector_mean_v1 import (
    update_fields, vector_hash,
)
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import (
    bounded_source_text,
)
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields


def selected_targets(args):
    parent = json.loads(args.parent_review.read_text())
    selected = []
    counts = {}
    for row in parent["targets"]:
        family = row["family"]
        if counts.get(family, 0) == args.per_family:
            continue
        if "/valid_unseen/" not in "/" + row["game"]:
            raise ValueError("Target is not in official valid_unseen")
        selected.append({"game": row["game"], "family": family,
            "game_sha256": row["game_sha256"]})
        counts[family] = counts.get(family, 0) + 1
    if len(counts) != 6 or any(n != args.per_family for n in counts.values()):
        raise ValueError("Expected six complete families")
    return selected


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    targets = []
    for index, row in enumerate(selected_targets(args)):
        if file_hash(args.data_root / row["game"]) != row["game_sha256"]:
            raise ValueError("Target content changed")
        content = {"index": index, **row,
            "parent_review_sha256": file_hash(args.parent_review),
            "checkpoint_sha256": file_hash(args.checkpoint),
            "method": "from_empty_reward_gated_source_vector"}
        targets.append({**content,
            "input_content_sha256": digest(content),
            "reviewed_target": True,
            "review_basis": "Fresh binding to frozen valid_unseen game content and source-free online reward-gated protocol"})
    value = {"protocol": "Fresh target-only content review for official valid_unseen online reward-gated memory; prior expert source fields are not read by actor or hypernetwork",
        "parent_review_sha256": file_hash(args.parent_review),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "per_family": args.per_family, "targets": targets}
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"reviewed_targets": len(targets),
                      "per_family": args.per_family}), flush=True)


def checked_targets(args):
    value = json.loads(args.review.read_text())
    selected = selected_targets(args)
    if (value["parent_review_sha256"] != file_hash(args.parent_review) or
            value["checkpoint_sha256"] != file_hash(args.checkpoint) or
            value["per_family"] != args.per_family or
            len(value["targets"]) != len(selected)):
        raise ValueError("Online valid_unseen review lineage changed")
    for index, row in enumerate(selected):
        if file_hash(args.data_root / row["game"]) != row["game_sha256"]:
            raise ValueError("Valid_unseen game content changed")
        content = {"index": index, **row,
            "parent_review_sha256": file_hash(args.parent_review),
            "checkpoint_sha256": file_hash(args.checkpoint),
            "method": "from_empty_reward_gated_source_vector"}
        target = value["targets"][index]
        if (not target["reviewed_target"] or
                target["input_content_sha256"] != digest(content) or
                any(target[key] != val for key, val in content.items())):
            raise ValueError("Unreviewed or changed target")
    return value["targets"]


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    targets = checked_targets(args)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != "contextual" or
            not agent.task_conditioned or
            agent.task_context_scope != "current" or
            agent.task_pair_pooling != "mean"):
        raise ValueError("Expected frozen current-context ALFWorld hypernetwork")
    fields = None
    updates = 0
    prior = []
    report = {"protocol": "Official valid_unseen exploratory from-empty online reward-gated source-vector memory; per-family frozen target order, global memory accumulation; only own official-won trajectories update; paired same-actor base; frozen Qwen/hypernetwork; 50 steps/64 tokens/two history turns/constrained greedy/two-repeat loop guard",
        "review_sha256": file_hash(args.review),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "per_family": args.per_family,
        "max_steps": 50, "max_new_tokens": 64,
        "actor_history_turns": 2, "loop_guard_max": 2,
        "context_tokens": args.context_tokens,
        "games": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for row in targets:
        game = args.data_root / row["game"]
        before = vector_hash(fields)
        base = run_episode(agent, tokenizer, game, {}, adapter=False,
            device=args.device, max_steps=50, max_new_tokens=64,
            constrain_actions=True, actor_history_turns=2, loop_guard_max=2)
        online = run_episode(agent, tokenizer, game, fields or {},
            adapter=fields is not None, device=args.device,
            max_steps=50, max_new_tokens=64,
            constrain_actions=True, actor_history_turns=2, loop_guard_max=2)
        entry = {"game": row["game"], "family": row["family"],
            "input_content_sha256": row["input_content_sha256"],
            "prior_episode_count": len(prior),
            "prior_episodes_sha256": digest(prior),
            "source_vector_sha256_before": before,
            "base": base, "online": online}
        if (base["status"] == online["status"] == "complete" and
                base["initial_observation"] == online["initial_observation"]):
            try:
                records = records_from_episode(online)
                entry["new_records_sha256"] = digest(records)
                entry["memory_updated"] = bool(online["reward"])
                if online["reward"]:
                    text, original, retained = bounded_source_text(
                        tokenizer, records, args.context_tokens)
                    fresh = contextual_text_fields(agent, tokenizer, text,
                        args.device, args.context_tokens, pooling="both")
                    fields = update_fields(fields, fresh, updates)
                    updates += 1
                    entry.update({"source_tokens_original": original,
                                  "source_tokens_retained": retained})
                prior.append({"game": row["game"],
                    "reward": online["reward"], "records": records})
            except Exception as exc:
                report["failures"].append({"game": row["game"],
                    "error": f"{type(exc).__name__}: {exc}"})
        else:
            report["failures"].append({"game": row["game"],
                "error": "Incomplete paired episode or inconsistent reset"})
        entry["source_vector_sha256_after"] = vector_hash(fields)
        entry["update_count_after"] = updates
        report["games"].append(entry)
        report["summary"] = {"n": len(report["games"]),
            "base": sum(x["base"].get("reward", 0) for x in report["games"]),
            "online": sum(x["online"].get("reward", 0) for x in report["games"]),
            "updates": updates, "failures": len(report["failures"])}
        temp = args.output.with_suffix(args.output.suffix + ".tmp")
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temp.replace(args.output)
        print(json.dumps(report["summary"]), flush=True)
        if report["failures"]:
            raise RuntimeError("Official valid_unseen online run failed; record retained")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--parent-review", type=Path, default=Path(
        "data/annotations/alf_memrl134_hyperlora_train_sources_reviewed_20261006.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/alf_online_reward_gate_valid_unseen_36_reviewed_20261007.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_reward_gate_valid_unseen_36_20261007.json"))
    parser.add_argument("--per-family", type=int, default=6)
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.65)
    args = parser.parse_args()
    if args.per_family < 1 or args.context_tokens < 2:
        parser.error("Invalid family count or source token budget")
    (prepare if args.command == "prepare" else evaluate)(args)


if __name__ == "__main__":
    main()
