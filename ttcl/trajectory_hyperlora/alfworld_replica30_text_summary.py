"""Frozen-Qwen generated text-experience control on reviewed replica30."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_replica30_direct_text import save
from ttcl.trajectory_hyperlora.alfworld_same_game_retry import file_hash
from ttcl.trajectory_hyperlora.alfworld_sibling_transfer import (
    checked_review, sibling_memory_text,
)
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import (
    generate, load_agent, run_episode,
)


def summarize(agent, tokenizer, records: list[dict], device: str) -> str:
    messages = [{"role": "system", "content":
        "You write reusable experience for an ALFWorld agent. Use only the "
        "completed source trajectory. Do not see or assume the next task."},
        {"role": "user", "content":
        "Extract a short transferable lesson from this successful trajectory. "
        "Describe the procedure and how to use observations. Avoid copying "
        "specific object names, receptacle names, room IDs, or numeric IDs. "
        "Keep it under 80 words.\n\n" + sibling_memory_text(records)}]
    return generate(agent, tokenizer, messages, device, 112)


def evaluate(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = checked_review(args)
    prior = json.loads(args.lora_result.read_text())
    if (len(rows) != 30 or prior["failures"] or len(prior["games"]) != 30 or
            prior["summary"] != {"n": 30, "failures": 0, "base": 14.0,
                                 "own": 25.0, "wrong": 26.0} or
            prior["candidates_sha256"] != file_hash(args.candidates) or
            prior["source_review_sha256"] != file_hash(args.source_review) or
            prior["checkpoint_sha256"] != file_hash(args.checkpoint) or
            prior["actor_history_turns"] != 2):
        raise ValueError("Frozen replica30 comparison lineage changed")
    paired = {item["game"]: item for item in prior["games"]}
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    agent.set_source(None)
    output = {
        "protocol": "Same reviewed replica30 sibling source and target as trajectory LoRA; frozen Qwen summarizes only source history into <=112 generated tokens using a generic transferable-lesson prompt, then same frozen actor runs with summary text only, no LoRA; greedy constrained 30-command 64-token two-turn official won",
        "candidates_sha256": file_hash(args.candidates),
        "source_review_sha256": file_hash(args.source_review),
        "lora_result_sha256": file_hash(args.lora_result),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "runner_sha256": file_hash(Path(__file__)),
        "games": [], "failures": [],
    }
    for row, note in rows:
        game = row["target_game"]
        try:
            old = paired[game]
            if (old["input_content_sha256"] != row["input_content_sha256"] or
                    old["source_game"] != row["source_game"]):
                raise ValueError("Paired source or target changed")
            lesson = summarize(agent, tokenizer, note["source_records"],
                               args.device)
            if not lesson:
                raise ValueError("Empty generated experience")
            episode = run_episode(
                agent, tokenizer, args.data_root / game, {}, adapter=False,
                device=args.device, max_steps=30, max_new_tokens=64,
                constrain_actions=True, actor_history_turns=2,
                memory_text=lesson)
            if (episode["status"] != "complete" or
                    episode["initial_observation"] !=
                    old["arms"]["base"]["initial_observation"]):
                raise ValueError("Summarized text rollout or reset failed")
            output["games"].append({"game": game,
                "input_content_sha256": row["input_content_sha256"],
                "source_records_sha256": note["source_records_sha256"],
                "lesson": lesson,
                "lesson_tokens": len(tokenizer(lesson,
                    add_special_tokens=False).input_ids),
                "text_summary": episode})
        except Exception as error:
            output["failures"].append({"game": game,
                                       "error": f"{type(error).__name__}: {error}"})
        output["summary"] = {
            "n": len(output["games"]), "failures": len(output["failures"]),
            "text_summary": sum(item["text_summary"]["reward"]
                                for item in output["games"]),
            "mean_lesson_tokens": (sum(item["lesson_tokens"]
                for item in output["games"]) / len(output["games"])
                if output["games"] else 0),
        }
        save(args.output, output)
        print(json.dumps(output["summary"]), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple_replica30_candidates_20261006.json"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alf_sibling_simple_replica30_reviewed_20261006.json"))
    parser.add_argument("--retry-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_same_game_retry_candidates_20261006.json"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/alf_same_game_retry_reviewed_20261006.json"))
    parser.add_argument("--all-source-review", type=Path, default=Path(
        "data/annotations/alf_expert_first_attempts_reviewed_20261006.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--lora-result", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple_replica30_onpolicy600_compact2_20261006.json"))
    parser.add_argument("--split", default="train_large")
    parser.add_argument("--large-offset", type=int, default=140)
    parser.add_argument("--family", default="pick_and_place_simple")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.55)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.split != "train_large" or args.large_offset != 140 or
            args.family != "pick_and_place_simple" or
            not 0 < args.gpu_fraction <= 1):
        parser.error("Frozen replica30 protocol changed")
    evaluate(args)


if __name__ == "__main__":
    main()
