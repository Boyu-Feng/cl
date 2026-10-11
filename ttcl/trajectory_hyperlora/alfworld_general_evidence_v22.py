"""Exploratory, paired ALFWorld LoRA plus non-JSON public evidence.

Both arms mount the same frozen trajectory LoRA and use the same repeated
observation/command filter. The hybrid arm additionally retrieves past
observation/action/feedback records at each decision. No reward is used to
select evidence from the target episode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_replica30_direct_text import save
from ttcl.trajectory_hyperlora.alfworld_same_game_retry import file_hash
from ttcl.trajectory_hyperlora.alfworld_sibling_transfer import checked_review
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_source_fields
from ttcl.trajectory_hyperlora.general_evidence_v22 import (
    digest, extract_records, text_provider, verify,
)
from ttcl.trajectory_hyperlora.relational_router_pilot import tokenize_records


def bindings(args):
    rows = checked_review(args)
    prior = json.loads(args.lora_result.read_text())
    if (len(rows) != 30 or prior["failures"] or len(prior["games"]) != 30 or
            prior["summary"] != {"n": 30, "failures": 0, "base": 14.0,
                                 "own": 25.0, "wrong": 26.0} or
            prior["candidates_sha256"] != file_hash(args.candidates) or
            prior["source_review_sha256"] != file_hash(args.source_review) or
            prior["checkpoint_sha256"] != file_hash(args.checkpoint)):
        raise ValueError("Changed reviewed replica30 lineage")
    if not 1 <= args.limit <= 30:
        raise ValueError("Limit must be 1..30")
    selected = rows[:args.limit]
    bound = []
    for index, (row, note) in enumerate(selected):
        if digest(note["source_records"]) != note["source_records_sha256"]:
            raise ValueError("Source records changed")
        content = {"index": index, "target_game": row["target_game"],
            "target_game_sha256": file_hash(args.data_root / row["target_game"]),
            "source_game": row["source_game"],
            "source_records_sha256": note["source_records_sha256"],
            "parent_input_content_sha256": row["input_content_sha256"],
            "checkpoint_sha256": file_hash(args.checkpoint),
            "method": "lora_plus_public_text_evidence_v22_loop_guard2"}
        bound.append({**content, "input_content_sha256": digest(content),
                      "reviewed_target": True})
    return selected, prior, bound


def prepare(args):
    if args.review_v22.exists():
        raise FileExistsError(args.review_v22)
    _, _, targets = bindings(args)
    review = {"protocol": "New method-specific target and source-content bindings; "
               "each source is already independently approved in the replica30 review",
        "parent_source_review_sha256": file_hash(args.source_review),
        "parent_lora_result_sha256": file_hash(args.lora_result),
        "runner_sha256": file_hash(Path(__file__)),
        "targets": targets}
    save(args.review_v22, review)
    return review


def checked(args):
    rows, prior, targets = bindings(args)
    review = json.loads(args.review_v22.read_text())
    if (review["parent_source_review_sha256"] != file_hash(args.source_review) or
            review["parent_lora_result_sha256"] != file_hash(args.lora_result) or
            review["runner_sha256"] != file_hash(Path(__file__)) or
            review["targets"] != targets):
        raise ValueError("Changed v22 reviewed input binding")
    return rows, prior, targets


def source_fields(agent, tokenizer, records, args, prior):
    if agent.encoder_kind == "contextual":
        return contextual_source_fields(agent, tokenizer, records, args.device,
            agent.contextual_source_max_tokens,
            pooling="both" if agent.task_conditioned and
            agent.task_pair_pooling == "mean" else "last")
    return tokenize_records(tokenizer, records, args.device,
        max_tokens=prior["source_max_tokens"],
        truncation_mode=prior["source_truncation"])


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    rows, prior, targets = checked(args)
    old = {item["game"]: item for item in prior["games"]}
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    report = {"protocol": "Exploratory reviewed replica30 prefix; two matched "
        "single-attempt frozen-LoRA arms, 30 commands, 64 new tokens, constrained "
        "greedy actor, two-turn history and identical same-observation repeated-"
        "command guard (threshold 2); additive per-step non-JSON public evidence "
        "only in hybrid arm; previously exposed train-domain targets",
        "review_sha256": file_hash(args.review_v22),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "runner_sha256": file_hash(Path(__file__)),
        "limit": args.limit, "games": [], "failures": []}
    save(args.output, report)
    for (row, note), target in zip(rows, targets, strict=True):
        game = row["target_game"]
        entry = {"game": game,
                 "input_content_sha256": target["input_content_sha256"],
                 "source_records_sha256": note["source_records_sha256"]}
        try:
            records = note["source_records"]
            evidence = extract_records(records, source_index=0,
                source_sha256=note["source_records_sha256"], reward=1.)
            if not all(verify(item, records, note["source_records_sha256"])
                       for item in evidence):
                raise ValueError("Evidence provenance mismatch")
            fields = source_fields(agent, tokenizer, records, args, prior)
            provider = text_provider(evidence, target_index=1,
                                     max_chars=args.evidence_chars)
            with torch.no_grad():
                for arm in ("guarded_lora", "guarded_lora_evidence"):
                    episode = run_episode(agent, tokenizer,
                        args.data_root / game, fields, adapter=True,
                        device=args.device, max_steps=30, max_new_tokens=64,
                        constrain_actions=True, actor_history_turns=2,
                        loop_guard_max=2,
                        evidence_provider=provider if arm.endswith("evidence") else None)
                    if (episode["status"] != "complete" or
                            episode["initial_observation"] !=
                            old[game]["arms"]["base"]["initial_observation"]):
                        raise ValueError(f"Incomplete or changed {arm} rollout")
                    entry[arm] = episode
            report["games"].append(entry)
        except Exception as error:
            report["failures"].append({"game": game,
                                       "error": f"{type(error).__name__}: {error}"})
        report["summary"] = {"n": len(report["games"]),
            "failures": len(report["failures"]),
            "guarded_lora": sum(x["guarded_lora"]["reward"]
                                for x in report["games"]),
            "guarded_lora_evidence": sum(x["guarded_lora_evidence"]["reward"]
                                         for x in report["games"])}
        save(args.output, report)
        print(json.dumps(report["summary"]), flush=True)
        if report["failures"]:
            raise RuntimeError("Failure preserved in new report")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate"))
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
    parser.add_argument("--data-root", type=Path, default=Path("ttcl/data/alfworld_delta"))
    parser.add_argument("--lora-result", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple_replica30_onpolicy600_compact2_20261006.json"))
    parser.add_argument("--split", default="train_large")
    parser.add_argument("--large-offset", type=int, default=140)
    parser.add_argument("--family", default="pick_and_place_simple")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--evidence-chars", type=int, default=1400)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.55)
    parser.add_argument("--review-v22", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "evaluate" and args.output is None:
        parser.error("evaluate requires --output")
    if (args.split != "train_large" or args.large_offset != 140 or
            args.family != "pick_and_place_simple" or
            args.evidence_chars < 200):
        parser.error("Changed frozen exploratory protocol")
    (prepare if args.command == "prepare" else evaluate)(args)


if __name__ == "__main__":
    main()
