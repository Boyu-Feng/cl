"""Cross-evaluate own successful train trajectories on disjoint train tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import (
    bounded_source_text,
)
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields


SOURCE_INDICES = (3, 7, 11, 15)
TARGET_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17)


def family(game):
    return next(part.split("-")[0] for part in Path(game).parts
                if part.startswith(("pick_", "look_")))


def expected(args):
    source_report = json.loads(args.source_report.read_text())
    target_report = json.loads(args.target_report.read_text())
    if (source_report["checkpoint_sha256"] != file_hash(args.checkpoint) or
            target_report["checkpoint_sha256"] != file_hash(args.checkpoint) or
            source_report["failures"] or target_report["failures"] or
            len(source_report["games"]) != 18 or
            len(target_report["games"]) != 18):
        raise ValueError("Changed train-domain source or target runs")
    pairs = []
    for source_index in SOURCE_INDICES:
        source = source_report["games"][source_index]
        episode = source["episode"]
        if episode["status"] != "complete" or episode["reward"] != 1:
            raise ValueError("Selected own source was not successful")
        records = records_from_episode(episode)
        if source["new_records_sha256"] != digest(records):
            raise ValueError("Own source records changed")
        for target_index in TARGET_INDICES:
            target = target_report["games"][target_index]
            if target["game"] == source["game"]:
                raise ValueError("Source and target task overlap")
            content = {"source_index": source_index,
                "source_game": source["game"],
                "source_game_sha256": file_hash(args.data_root / source["game"]),
                "source_records_sha256": digest(records),
                "source_report_sha256": file_hash(args.source_report),
                "target_index": target_index,
                "target_game": target["game"],
                "target_game_sha256": file_hash(args.data_root / target["game"]),
                "target_report_sha256": file_hash(args.target_report),
                "checkpoint_sha256": file_hash(args.checkpoint),
                "method": "own_success_source_cross_target_lora"}
            pairs.append({**content, "input_content_sha256": digest(content),
                "source_family": family(source["game"]),
                "target_family": family(target["game"]),
                "source_records": records,
                "target_base_reward": target["base"]["reward"]})
    if len(pairs) != 48 or len({x["source_family"] for x in pairs}) != 4 or \
            len({x["target_family"] for x in pairs}) != 4:
        raise ValueError("Expected a four-by-twelve cross-task matrix")
    return pairs


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    pairs = expected(args)
    targets = [{key: val for key, val in row.items()
                if key != "source_records"} for row in pairs]
    value = {"protocol": "Fresh content-bound own-success source by disjoint train target matrix; no old history-id labels reused",
        "source_report_sha256": file_hash(args.source_report),
        "target_report_sha256": file_hash(args.target_report),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "targets": targets}
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"reviewed_pairs": len(targets)}), flush=True)


def checked_pairs(args):
    pairs = expected(args)
    review = json.loads(args.review.read_text())
    without_records = [{key: val for key, val in row.items()
                        if key != "source_records"} for row in pairs]
    if (review["source_report_sha256"] != file_hash(args.source_report) or
            review["target_report_sha256"] != file_hash(args.target_report) or
            review["checkpoint_sha256"] != file_hash(args.checkpoint) or
            review["targets"] != without_records):
        raise ValueError("Unreviewed or changed cross-task input")
    return pairs


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    pairs = checked_pairs(args)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != "contextual" or
            agent.task_context_scope != "current" or
            agent.task_pair_pooling != "mean"):
        raise ValueError("Expected frozen contextual hypernetwork")
    source_cache = {}
    result = {"protocol": "Four reviewed successful own train trajectories crossed with twelve disjoint reviewed train tasks; 48 frozen LoRA single attempts, 50 steps/64 tokens/two-turn history/constrained greedy/two-repeat loop guard; source family only for analysis",
        "review_sha256": file_hash(args.review),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "max_steps": 50, "max_new_tokens": 64,
        "actor_history_turns": 2, "loop_guard_max": 2,
        "context_tokens": args.context_tokens,
        "pairs": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for pair in pairs:
        source_index = pair["source_index"]
        if source_index not in source_cache:
            text, original, retained = bounded_source_text(
                tokenizer, pair["source_records"], args.context_tokens)
            fields = contextual_text_fields(agent, tokenizer, text,
                args.device, args.context_tokens, pooling="both")
            source_cache[source_index] = (fields, original, retained)
        fields, original, retained = source_cache[source_index]
        episode = run_episode(agent, tokenizer,
            args.data_root / pair["target_game"], fields,
            adapter=True, device=args.device, max_steps=50, max_new_tokens=64,
            constrain_actions=True, actor_history_turns=2, loop_guard_max=2)
        entry = {"source_index": source_index,
            "target_index": pair["target_index"],
            "source_game": pair["source_game"],
            "target_game": pair["target_game"],
            "source_family": pair["source_family"],
            "target_family": pair["target_family"],
            "input_content_sha256": pair["input_content_sha256"],
            "source_tokens_original": original,
            "source_tokens_retained": retained,
            "target_base_reward": pair["target_base_reward"],
            "episode": episode}
        if episode["status"] != "complete":
            result["failures"].append({"source_index": source_index,
                "target_index": pair["target_index"],
                "error": episode.get("error", "incomplete rollout")})
        result["pairs"].append(entry)
        result["summary"] = {"n": len(result["pairs"]),
            "success": sum(x["episode"].get("reward", 0)
                           for x in result["pairs"]),
            "failures": len(result["failures"])}
        temp = args.output.with_suffix(args.output.suffix + ".tmp")
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        temp.replace(args.output)
        print(json.dumps(result["summary"]), flush=True)
        if result["failures"]:
            raise RuntimeError("Cross-task source utility failure recorded")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--source-report", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq0_6_20261007.json"))
    parser.add_argument("--target-report", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_context_persistent_train_seq6_6_20261007.json"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/alf_own_success_source_utility_matrix48_reviewed_20261007.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_own_success_source_utility_matrix48_20261007.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.65)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else evaluate)(args)


if __name__ == "__main__":
    main()
