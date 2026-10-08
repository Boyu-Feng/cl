"""ALFWorld v2: success-gated persistent LoRA with task-conditioned readout.

The frozen hypernetwork converts each own successful trajectory to factors
once. A bounded parameter bank persists across games. The next game's initial
observation selects a convex mixture of those factors, without rereading old
trajectory text or using task-family labels. This is an exploratory train
split protocol, not a trained gate or an unseen-test result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import (
    checked_targets as checked_parent_targets, episode_factor,
)
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import (
    contextual_text_fields, task_context_text,
)
from ttcl.trajectory_hyperlora.online_parameter_memory_v2 import ParameterMemory


def parent_targets(args):
    original = args.review
    try:
        args.review = args.parent_persistent_review
        return checked_parent_targets(args)
    finally:
        args.review = original


def target_content(args, index, row):
    return {"index": index, "game": row["game"],
            "game_sha256": file_hash(args.data_root / row["game"]),
            "parent_target_sha256": row["input_content_sha256"],
            "parent_review_sha256": file_hash(args.parent_persistent_review),
            "checkpoint_sha256": file_hash(args.checkpoint),
            "method": "success_gated_task_keyed_persistent_lora_v2",
            "capacity": args.capacity, "temperature": args.temperature,
            "attention_mix": args.attention_mix,
            "context_tokens": args.context_tokens}


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    parent = parent_targets(args)
    targets = []
    for index, row in enumerate(parent):
        content = target_content(args, index, row)
        targets.append({**content, "input_content_sha256": digest(content),
                        "reviewed_target": True,
                        "review_basis": "Checked official train target; own completed success only; no future source"})
    review = {"protocol": "Content-bound v2 task-keyed persistent LoRA train targets",
              "parent_review_sha256": file_hash(args.parent_persistent_review),
              "checkpoint_sha256": file_hash(args.checkpoint), "targets": targets}
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"reviewed_targets": len(targets)}), flush=True)


def checked_targets(args):
    parent = parent_targets(args)
    review = json.loads(args.review.read_text())
    if (review["parent_review_sha256"] != file_hash(args.parent_persistent_review) or
            review["checkpoint_sha256"] != file_hash(args.checkpoint) or
            len(review["targets"]) != len(parent)):
        raise ValueError("v2 target review lineage changed")
    for index, row in enumerate(parent):
        content = target_content(args, index, row)
        approved = review["targets"][index]
        if (approved["reviewed_target"] is not True or
                approved["input_content_sha256"] != digest(content) or
                any(approved[key] != value for key, value in content.items())):
            raise ValueError("Unreviewed or changed v2 target")
    return review["targets"]


def initial_key(agent, tokenizer, observation: str, args):
    fields = contextual_text_fields(agent, tokenizer,
        task_context_text(observation, observation), args.device,
        args.context_tokens, pooling="both")
    return fields["pair_contextual"]


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    targets = checked_targets(args)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != "contextual" or
            agent.task_conditioned is not True or
            agent.task_context_scope != "current" or
            agent.task_pair_pooling != "mean"):
        raise ValueError("Expected frozen current-context hypernetwork")
    memory = ParameterMemory(capacity=args.capacity,
        temperature=args.temperature, attention_mix=args.attention_mix)
    prior = []
    report = {"protocol": "From-empty exploratory official ALFWorld train sequence; own official successes produce persistent LoRA factors once; current initial task retrieves convex factor mixture by frozen contextual key; no historical text reread; paired no-LoRA baseline",
        "review_sha256": file_hash(args.review),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "capacity": args.capacity, "temperature": args.temperature,
        "attention_mix": args.attention_mix,
        "context_tokens": args.context_tokens,
        "max_steps": 50, "max_new_tokens": 64,
        "actor_history_turns": 2, "loop_guard_max": 2,
        "games": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for row in targets:
        game = args.data_root / row["game"]
        before = memory.digest()
        retrieval = {}

        def select_adapter(observation):
            nonlocal retrieval
            if not memory.entries:
                retrieval = {"entries": 0, "weights": [], "cosine": []}
                return None
            factors, retrieval = memory.read(initial_key(
                agent, tokenizer, observation, args))
            return factors

        base = run_episode(agent, tokenizer, game, {}, adapter=False,
            device=args.device, max_steps=50, max_new_tokens=64,
            constrain_actions=True, actor_history_turns=2,
            loop_guard_max=2)
        online = run_episode(agent, tokenizer, game, {}, adapter=False,
            fixed_adapter_selector=select_adapter, device=args.device,
            max_steps=50, max_new_tokens=64, constrain_actions=True,
            actor_history_turns=2, loop_guard_max=2)
        entry = {"game": row["game"],
            "input_content_sha256": row["input_content_sha256"],
            "prior_episode_count": len(prior),
            "prior_episodes_sha256": digest(prior),
            "memory_sha256_before": before,
            "memory_entries_before": len(memory.entries),
            "retrieval": retrieval,
            "base": base, "online": online,
            "memory_updated": False}
        if (base["status"] == "complete" and online["status"] == "complete" and
                base["initial_observation"] == online["initial_observation"]):
            try:
                records = records_from_episode(online)
                entry["new_records_sha256"] = digest(records)
                if online["reward"]:
                    factors, _, original, retained = episode_factor(
                        agent, tokenizer, online, args)
                    key = initial_key(agent, tokenizer,
                        online["initial_observation"], args)
                    entry["write_slot"] = memory.write(key, factors)
                    entry["source_tokens_original"] = original
                    entry["source_tokens_retained"] = retained
                    entry["increment_l2"] = [float(value.norm())
                                             for value in factors]
                    entry["memory_updated"] = True
                prior.append({"game": row["game"],
                    "reward": online["reward"], "records": records})
            except Exception as exc:
                report["failures"].append({"game": row["game"],
                    "error": f"{type(exc).__name__}: {exc}"})
        else:
            report["failures"].append({"game": row["game"],
                "error": "Incomplete paired episode or inconsistent reset"})
        entry["memory_sha256_after"] = memory.digest()
        entry["memory_entries_after"] = len(memory.entries)
        report["games"].append(entry)
        complete = [item for item in report["games"] if
                    item["base"]["status"] == item["online"]["status"] == "complete"]
        report["summary"] = {"n": len(complete),
            "base": sum(item["base"]["reward"] for item in complete),
            "online": sum(item["online"]["reward"] for item in complete),
            "writes": sum(item["memory_updated"] for item in report["games"]),
            "failures": len(report["failures"])}
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(args.output)
        print(json.dumps(report["summary"]), flush=True)
        if report["failures"]:
            raise RuntimeError("v2 online parameter memory failure recorded")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--plan", type=Path, default=Path(
        "ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json"))
    parser.add_argument("--data-root", type=Path, default=Path("ttcl/data/alfworld_delta"))
    parser.add_argument("--parent-review", type=Path, default=Path(
        "data/annotations/alfworld_online_from_empty_train_seq6_6_reviewed_20261007.json"))
    parser.add_argument("--parent-persistent-review", type=Path, default=Path(
        "data/annotations/alfworld_online_context_persistent_train_seq6_6_reviewed_20261007.json"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/alfworld_online_parameter_memory_v2_train_seq6_6_reviewed_20261008.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_parameter_memory_v2_train_seq6_6_20261008.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.85)
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=.05)
    parser.add_argument("--attention-mix", type=float, default=.75)
    args = parser.parse_args()
    if args.context_tokens < 2:
        parser.error("Invalid source token budget")
    (prepare if args.command == "prepare" else evaluate)(args)


if __name__ == "__main__":
    main()
