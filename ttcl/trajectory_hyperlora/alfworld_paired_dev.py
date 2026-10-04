"""Paired full-episode ALFWorld train-game train/development rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.prepare_alf_next_task import validate_review
from ttcl.trajectory_hyperlora.relational_router_pilot import tokenize_records


def summarize(games: list[dict]) -> dict:
    complete = [row for row in games
                if row["base"]["status"] == row["generated"]["status"] == "complete"]
    return {"completed_pairs": len(complete),
            "base_successes": sum(row["base"]["reward"] == 1 for row in complete),
            "generated_successes": sum(row["generated"]["reward"] == 1
                                       for row in complete),
            "base_invalid_commands": sum(row["base"]["invalid_commands"]
                                         for row in complete),
            "generated_invalid_commands": sum(row["generated"]["invalid_commands"]
                                              for row in complete)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_candidates_20261005.json"))
    parser.add_argument("--reviewed-annotations", type=Path, default=Path(
        "data/annotations/trajectory_hyperlora_alf_20261005_reviewed.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.40)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--split", choices=("train", "dev"), default="dev")
    parser.add_argument("--per-family", type=int, default=0)
    parser.add_argument("--game", action="append")
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--constrain-actions", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; do not overwrite a frozen rollout")
    manifest = json.loads(args.candidates.read_text())
    review = json.loads(args.reviewed_annotations.read_text())
    approved = validate_review(manifest, review)
    rows_by_game = {}
    for row in approved:
        if row["split"] == args.split:
            previous = rows_by_game.get(row["target_game"])
            if previous is None or row["turn"] > previous["turn"]:
                rows_by_game[row["target_game"]] = row
    selected = [rows_by_game[game] for game in sorted(rows_by_game)]
    if args.game:
        requested = set(args.game)
        selected = [row for row in selected if row["target_game"] in requested]
        if {row["target_game"] for row in selected} != requested:
            parser.error("A requested game is absent from the reviewed split")
    if args.per_family:
        counts = {}
        filtered = []
        for row in selected:
            family = row["family"]
            if counts.get(family, 0) < args.per_family:
                filtered.append(row)
                counts[family] = counts.get(family, 0) + 1
        selected = filtered
    if args.limit:
        selected = selected[:args.limit]
    if not selected or args.max_steps < 1:
        parser.error("Need selected games and positive budget")
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    agent.set_adapter_scale(args.adapter_scale)
    report = {"protocol": "Paired frozen ALFWorld train-game episodes; same target reset and declared step cap; no official test games; greedy actor",
              "split": args.split,
              "adapter_scale": args.adapter_scale,
              "constrain_actions": args.constrain_actions,
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "candidate_manifest_sha256": hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
              "reviewed_annotations_sha256": hashlib.sha256(
                  args.reviewed_annotations.read_bytes()).hexdigest(),
              "max_steps": args.max_steps,
              "max_new_tokens": args.max_new_tokens,
              "games": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for row in selected:
        game = args.data_root / row["target_game"]
        if hashlib.sha256(game.read_bytes()).hexdigest() != row["target_game_sha256"]:
            raise ValueError("Frozen target game hash mismatch")
        fields = tokenize_records(tokenizer, row["source_records"], args.device)
        pair = {"game": row["target_game"],
                "game_sha256": row["target_game_sha256"],
                "source_game": row["source_game"],
                "input_content_sha256": row["input_content_sha256"]}
        for arm, enabled in (("base", False), ("generated", True)):
            pair[arm] = run_episode(agent, tokenizer, game, fields,
                                    adapter=enabled, device=args.device,
                                    max_steps=args.max_steps,
                                    max_new_tokens=args.max_new_tokens,
                                    constrain_actions=args.constrain_actions)
        report["games"].append(pair)
        report["summary"] = summarize(report["games"])
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"game": row["target_game"],
                          "base": pair["base"].get("reward"),
                          "generated": pair["generated"].get("reward"),
                          "summary": report["summary"]}), flush=True)


if __name__ == "__main__":
    main()
