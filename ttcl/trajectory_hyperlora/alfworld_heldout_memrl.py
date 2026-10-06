"""One-history native MemRL comparison on reviewed ALFWorld retry targets.

Each target starts with an empty memory. The frozen actor's completed first
attempt is written with its official reward, then MemRL retrieves the resulting
experience for a second attempt. This isolates the memory representation under
the same actor, target games, action constraint, and two-attempt budget as the
trajectory-to-LoRA retry probe; it is not a full cross-task MemRL run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.memrl_comparison.memory import Embedder, Memory
from ttcl.trajectory_hyperlora.alfworld_heldout_retry import (
    checked_targets, replay_episode,
)
from ttcl.trajectory_hyperlora.alfworld_same_game_retry import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import records_from_episode
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.evaluate_xland_memrl import LocalQwenClient, plan


def _save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def evaluate(args: argparse.Namespace) -> dict:
    if args.output.exists() or args.directory.exists():
        raise FileExistsError("MemRL comparison needs fresh output paths")
    targets = checked_targets(args)
    if args.family:
        targets = [row for row in targets if row["family"] == args.family]
    targets = targets[args.offset:args.offset + args.limit]
    if not targets:
        raise ValueError("Empty target selection")
    prior = json.loads(args.base_result.read_text())
    prior_rows = {row["game"]: row for row in prior["games"]}
    if prior["review_sha256"] != file_hash(args.review) or prior["max_steps"] != 30 or prior["max_new_tokens"] != 64:
        raise ValueError("First-attempt result has a different target or budget")
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    client = LocalQwenClient(agent.model, tokenizer, args.device, args.seed)
    memory_plan = plan(args)
    embedder = Embedder(args.embedding)
    result = {
        "protocol": "Per-target empty native MemRL store; official first-attempt failure/success written with public trace and official reward, retrieved for one retry; same frozen Qwen actor and 30-step constrained compact-two-turn budget as LoRA; no cross-task accumulation",
        "review_sha256": file_hash(args.review),
        "base_result_sha256": file_hash(args.base_result),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "memory_plan": memory_plan,
        "family": args.family, "offset": args.offset, "limit": args.limit,
        "actor_history_turns": args.actor_history_turns,
        "max_steps": 30, "max_new_tokens": 64,
        "games": [], "failures": [],
    }
    args.directory.mkdir(parents=True)
    for index, row in enumerate(targets):
        try:
            original = prior_rows[row["game"]]
            if original["target_input_content_sha256"] != row["input_content_sha256"]:
                raise ValueError("First-attempt target content changed")
            base = original["arms"]["base"]
            game = args.data_root / row["game"]
            replay_episode(game, base)
            records = records_from_episode(base)
            query = base["initial_observation"]
            public_trace = "\n".join(json.dumps(record, ensure_ascii=False)
                                      for record in records)
            memory = Memory(memory_plan, client,
                            args.directory / f"target_{index:03d}",
                            {"threshold": args.retrieval_threshold,
                             "mean": 0., "std": 1.}, embedder=embedder)
            before = memory.retrieve(query)
            update = memory.update(query, public_trace, base["reward"],
                                   bool(base["reward"]), before,
                                   digest({"target": row["input_content_sha256"],
                                           "episode": base}))
            retrieved = memory.retrieve(query)
            retry = run_episode(agent, tokenizer, game, {}, adapter=False,
                                device=args.device, max_steps=30,
                                max_new_tokens=64, constrain_actions=True,
                                actor_history_turns=args.actor_history_turns,
                                memory_text=retrieved["context"])
            if retry["status"] != "complete" or retry["initial_observation"] != base["initial_observation"]:
                raise ValueError("MemRL retry failed or reset changed")
            memory.snapshot(args.directory / f"target_{index:03d}" / "memory.json")
            result["games"].append({
                "game": row["game"], "family": row["family"],
                "target_input_content_sha256": row["input_content_sha256"],
                "own_source_episode_sha256": digest(base),
                "own_source_records_sha256": digest(records),
                "new_memory_id": update["new_memory_id"],
                "retrieved_ids": retrieved["ids"],
                "retrieved_tokens": retrieved["tokens"],
                "writer_calls": memory.calls,
                "base": base, "retry": retry,
            })
        except Exception as exc:
            result["failures"].append({"game": row["game"],
                                       "error": f"{type(exc).__name__}: {exc}"})
        result["summary"] = {
            "n": len(result["games"]), "failures": len(result["failures"]),
            "base": sum(item["base"]["reward"] for item in result["games"]),
            "memrl_retry": sum(item["retry"]["reward"] for item in result["games"]),
            "within_two": sum(max(item["base"]["reward"], item["retry"]["reward"])
                              for item in result["games"]),
            "retrieved": sum(bool(item["retrieved_ids"]) for item in result["games"]),
        }
        _save(args.output, result)
        print(json.dumps(result["summary"]), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path("ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json"))
    parser.add_argument("--first-attempts", type=Path, default=Path("results/trajectory_hyperlora/alf_rl_20261005/train_rewards_complete.json"))
    parser.add_argument("--data-root", type=Path, default=Path("ttcl/data/alfworld_delta"))
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--base-result", type=Path, required=True)
    parser.add_argument("--family", default="pick_and_place_simple")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=6)
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
    if args.limit < 1 or args.offset < 0 or not 0 < args.gpu_fraction <= 1:
        parser.error("Invalid comparison budget")
    evaluate(args)


if __name__ == "__main__":
    main()
