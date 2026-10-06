"""Disjoint chunk of direct-history text comparison on audited MemRL-134."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_memrl134_transfer import checked_targets
from ttcl.trajectory_hyperlora.alfworld_replica30_direct_text import save
from ttcl.trajectory_hyperlora.alfworld_same_game_retry import file_hash
from ttcl.trajectory_hyperlora.alfworld_sibling_transfer import sibling_memory_text
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode


def evaluate(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    targets = checked_targets(args)
    prior = json.loads(args.lora_result.read_text())
    if (len(targets) != 134 or len(prior["games"]) != 134 or
            prior["summary"]["overall"]["base"] != 25 or
            prior["summary"]["overall"]["lora"] != 47 or
            prior["target_review_sha256"] != file_hash(args.target_review) or
            prior["checkpoint_sha256"] != file_hash(args.checkpoint) or
            prior["actor_history_turns"] != 2):
        raise ValueError("Frozen 134-game comparison lineage changed")
    paired = {item["game"]: item for item in prior["games"]}
    if set(paired) != {row["game"] for row in targets}:
        raise ValueError("Prior LoRA games differ from audited targets")
    if args.offset < 0 or args.limit < 1 or args.offset + args.limit > 134:
        raise ValueError("Invalid disjoint chunk bounds")
    targets = targets[args.offset:args.offset + args.limit]
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    output = {
        "protocol": "Disjoint slice of frozen reviewed official valid_unseen MemRL-134 target/source set; direct successful same-family train trajectory text in system prompt, no LoRA; one greedy constrained 50-command/64-token attempt, two-turn actor history, official won; same actor as paired LoRA result",
        "offset": args.offset, "limit": args.limit,
        "target_review_sha256": file_hash(args.target_review),
        "memrl_plan_sha256": file_hash(args.memrl_plan),
        "lora_result_sha256": file_hash(args.lora_result),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "runner_sha256": file_hash(Path(__file__)),
        "games": [], "failures": [],
    }
    for row in targets:
        game = row["game"]
        try:
            old = paired[game]
            if (old["target_input_content_sha256"] !=
                    row["input_content_sha256"] or
                    old["source_records_sha256"] != row["source_records_sha256"]):
                raise ValueError("Audited source or target changed")
            memory = sibling_memory_text(row["source_records"])
            episode = run_episode(
                agent, tokenizer, args.data_root / game, {}, adapter=False,
                device=args.device, max_steps=50, max_new_tokens=64,
                constrain_actions=True, actor_history_turns=2,
                memory_text=memory)
            if (episode["status"] != "complete" or
                    episode["initial_observation"] !=
                    old["arms"]["base"]["initial_observation"]):
                raise ValueError("Text rollout or reset failed")
            output["games"].append({
                "game": game, "family": row["family"],
                "target_input_content_sha256": row["input_content_sha256"],
                "source_records_sha256": row["source_records_sha256"],
                "text": episode})
        except Exception as error:
            output["failures"].append({"game": game,
                                       "error": f"{type(error).__name__}: {error}"})
        output["summary"] = {
            "n": len(output["games"]), "failures": len(output["failures"]),
            "text": sum(item["text"]["reward"] for item in output["games"]),
        }
        save(args.output, output)
        print(json.dumps(output["summary"]), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memrl-plan", type=Path, default=Path(
        "results/memrl_comparison/20260928_budgeted/plan.json"))
    parser.add_argument("--source-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train600_reviewed_20261006.json"))
    parser.add_argument("--retry-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_same_game_retry_candidates_20261006.json"))
    parser.add_argument("--retry-review", type=Path, default=Path(
        "data/annotations/alf_same_game_retry_reviewed_20261006.json"))
    parser.add_argument("--all-source-review", type=Path, default=Path(
        "data/annotations/alf_expert_first_attempts_reviewed_20261006.json"))
    parser.add_argument("--target-review", type=Path, default=Path(
        "data/annotations/alf_memrl134_hyperlora_train_sources_reviewed_20261006.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--lora-result", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_memrl134_simple40_reward_lora_audited_20261006.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.55)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.gpu_fraction <= 1:
        parser.error("Invalid GPU memory fraction")
    evaluate(args)


if __name__ == "__main__":
    main()
