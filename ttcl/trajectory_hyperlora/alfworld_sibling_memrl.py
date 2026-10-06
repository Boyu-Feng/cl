"""Native MemRL single-source comparison for reviewed ALFWorld sibling games.

The source is the same replay-approved successful sibling trajectory used by
the trajectory-to-LoRA arm. Each target has a fresh MemRL store, so this tests
one-source transfer rather than lifetime accumulation across target games.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.memrl_comparison.memory import Embedder, Memory
from ttcl.trajectory_hyperlora.alfworld_heldout_retry import replay_episode
from ttcl.trajectory_hyperlora.alfworld_same_game_retry import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_sibling_transfer import checked_review
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.evaluate_xland_memrl import LocalQwenClient, plan


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def evaluate(args: argparse.Namespace) -> dict:
    if args.output.exists() or args.directory.exists():
        raise FileExistsError("MemRL comparison requires fresh output paths")
    reviewed = checked_review(args)
    if args.limit:
        reviewed = reviewed[:args.limit]
    prior = json.loads(args.lora_result.read_text())
    if (prior["candidates_sha256"] != file_hash(args.candidates) or
            prior["source_review_sha256"] != file_hash(args.source_review) or
            prior["checkpoint_sha256"] != file_hash(args.checkpoint) or
            prior["actor_history_turns"] != args.actor_history_turns or
            len(prior["games"]) != len(reviewed) or prior["failures"]):
        raise ValueError("LoRA paired result or actor budget changed")
    paired = {item["game"]: item for item in prior["games"]}
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    client = LocalQwenClient(agent.model, tokenizer, args.device, args.seed)
    memory_plan = plan(args)
    embedder = Embedder(args.embedding)
    result = {
        "protocol": "Same 30 reviewed train_large ALFWorld sibling target/source pairs as reward-trained LoRA; one native MemRL memory per target from the replay-approved successful sibling history, retrieved for the target; frozen Qwen, constrained 30-step/64-token/2-turn actor; no cross-target accumulation",
        "candidates_sha256": file_hash(args.candidates),
        "source_review_sha256": file_hash(args.source_review),
        "lora_result_sha256": file_hash(args.lora_result),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "runner_sha256": file_hash(Path(__file__)),
        "memory_plan": memory_plan,
        "actor_history_turns": args.actor_history_turns,
        "max_steps": 30, "max_new_tokens": 64,
        "games": [], "failures": [],
    }
    args.directory.mkdir(parents=True)
    for index, (row, note) in enumerate(reviewed):
        try:
            previous = paired[row["target_game"]]
            if (previous["input_content_sha256"] != row["input_content_sha256"] or
                    previous["source_game"] != row["source_game"] or
                    previous["wrong_source_game"] != row["wrong_source_game"]):
                raise ValueError("Paired game or source changed")
            base = previous["arms"]["base"]
            game = args.data_root / row["target_game"]
            replay_episode(game, base)
            source = note["source_records"]
            if digest(source) != note["source_records_sha256"]:
                raise ValueError("Reviewed source content changed")
            source_query = source[0]["observation"]
            target_query = base["initial_observation"]
            public_trace = "\n".join(json.dumps(record, ensure_ascii=False)
                                      for record in source)
            memory = Memory(memory_plan, client,
                            args.directory / f"target_{index:03d}",
                            {"threshold": args.retrieval_threshold,
                             "mean": 0., "std": 1.}, embedder=embedder)
            before = memory.retrieve(source_query)
            update = memory.update(source_query, public_trace, 1., True,
                                   before, digest({
                                       "target": row["input_content_sha256"],
                                       "source_records": source}))
            retrieved = memory.retrieve(target_query)
            target = run_episode(agent, tokenizer, game, {}, adapter=False,
                                 device=args.device, max_steps=30,
                                 max_new_tokens=64, constrain_actions=True,
                                 actor_history_turns=args.actor_history_turns,
                                 memory_text=retrieved["context"])
            if (target["status"] != "complete" or
                    target["initial_observation"] != target_query):
                raise ValueError("MemRL target rollout or reset failed")
            memory.snapshot(args.directory / f"target_{index:03d}" / "memory.json")
            result["games"].append({
                "game": row["target_game"], "family": row["family"],
                "source_game": row["source_game"],
                "input_content_sha256": row["input_content_sha256"],
                "source_records_sha256": note["source_records_sha256"],
                "source_reward": 1.,
                "new_memory_id": update["new_memory_id"],
                "retrieved_ids": retrieved["ids"],
                "retrieved_tokens": retrieved["tokens"],
                "writer_calls": memory.calls,
                "base": base, "memrl": target,
            })
        except Exception as exc:
            result["failures"].append({"game": row["target_game"],
                                       "error": f"{type(exc).__name__}: {exc}"})
        result["summary"] = {
            "n": len(result["games"]), "failures": len(result["failures"]),
            "base": sum(item["base"]["reward"] for item in result["games"]),
            "memrl": sum(item["memrl"]["reward"] for item in result["games"]),
            "retrieved": sum(bool(item["retrieved_ids"])
                             for item in result["games"]),
        }
        save(args.output, result)
        print(json.dumps(result["summary"]), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--source-review", type=Path, required=True)
    parser.add_argument("--retry-candidates", type=Path, default=Path("results/trajectory_hyperlora/alf_same_game_retry_candidates_20261006.json"))
    parser.add_argument("--review", type=Path, default=Path("data/annotations/alf_same_game_retry_reviewed_20261006.json"))
    parser.add_argument("--all-source-review", type=Path, default=Path("data/annotations/alf_expert_first_attempts_reviewed_20261006.json"))
    parser.add_argument("--data-root", type=Path, default=Path("ttcl/data/alfworld_delta"))
    parser.add_argument("--split", default="train_large")
    parser.add_argument("--large-offset", type=int, default=140)
    parser.add_argument("--family", default="pick_and_place_simple")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--lora-result", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=Path("current_work/MemRL"))
    parser.add_argument("--embedding", type=Path, default=Path("models/embedding/bge-m3"))
    parser.add_argument("--memory-tokens", type=int, default=2048)
    parser.add_argument("--writer-tokens", type=int, default=256)
    parser.add_argument("--retrieval-threshold", type=float, default=.5)
    parser.add_argument("--actor-history-turns", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.55)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.split != "train_large" or args.limit < 1 or
            args.large_offset < 0 or args.actor_history_turns < 1 or
            not 0 < args.gpu_fraction <= 1):
        parser.error("Invalid paired comparison budget")
    evaluate(args)


if __name__ == "__main__":
    main()
