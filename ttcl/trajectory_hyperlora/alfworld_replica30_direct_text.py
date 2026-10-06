"""Direct-history text control for the frozen 30-game trajectory-LoRA test.

Uses exactly the reviewed histories and target games of the existing paired
LoRA run. The target actor receives the source only in its system prompt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_same_game_retry import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_sibling_transfer import (
    checked_review, sibling_memory_text,
)
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


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
    output = {
        "protocol": "Audited replica30 direct source trajectory text in system prompt; same frozen Qwen actor, greedy constrained commands, two-turn history, 30 commands, 64 new tokens, official won; no LoRA; existing 30-game exploratory split",
        "candidates_sha256": file_hash(args.candidates),
        "source_review_sha256": file_hash(args.source_review),
        "lora_result_sha256": file_hash(args.lora_result),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "runner_sha256": file_hash(Path(__file__)),
        "actor_history_turns": 2, "max_steps": 30,
        "games": [], "failures": [],
    }
    for row, note in rows:
        game = row["target_game"]
        try:
            old = paired[game]
            if (old["input_content_sha256"] != row["input_content_sha256"] or
                    old["source_game"] != row["source_game"] or
                    old["wrong_source_game"] != row["wrong_source_game"] or
                    digest(note["source_records"]) != note["source_records_sha256"] or
                    digest(note["wrong_records"]) != note["wrong_records_sha256"]):
                raise ValueError("Reviewed game or source changed")
            arms = {}
            for name, records in (("own_text", note["source_records"]),
                                  ("wrong_text", note["wrong_records"])):
                memory = sibling_memory_text(records)
                arms[name] = run_episode(
                    agent, tokenizer, args.data_root / game, {}, adapter=False,
                    device=args.device, max_steps=30, max_new_tokens=64,
                    constrain_actions=True, actor_history_turns=2,
                    memory_text=memory)
                if (arms[name]["status"] != "complete" or
                        arms[name]["initial_observation"] !=
                        old["arms"]["base"]["initial_observation"]):
                    raise ValueError(f"{name} rollout or reset failed")
            output["games"].append({
                "game": game, "input_content_sha256": row["input_content_sha256"],
                "source_records_sha256": note["source_records_sha256"],
                "wrong_records_sha256": note["wrong_records_sha256"],
                "arms": arms})
        except Exception as error:
            output["failures"].append({"game": game,
                                       "error": f"{type(error).__name__}: {error}"})
        output["summary"] = {
            "n": len(output["games"]), "failures": len(output["failures"]),
            **{name: sum(item["arms"][name]["reward"]
                         for item in output["games"])
               for name in ("own_text", "wrong_text")},
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
