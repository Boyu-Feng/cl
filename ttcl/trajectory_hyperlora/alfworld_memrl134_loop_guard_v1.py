"""Paired trajectory-LoRA first attempts on MemRL's frozen ALFWorld games.

Sources are successful, replay-reviewed training games from each task family.
The 134 valid_unseen targets and order are taken from the prior MemRL plan.
No target walkthrough or historical target outcome is read for source choice.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_same_game_retry import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_sibling_transfer import checked_review
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_source_fields
from ttcl.trajectory_hyperlora.relational_router_pilot import tokenize_records


def _save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _source_pool(args: argparse.Namespace) -> dict[str, list[tuple[dict, dict]]]:
    reviewed = checked_review(argparse.Namespace(
        candidates=args.source_candidates, source_review=args.source_review,
        retry_candidates=args.retry_candidates, review=args.retry_review,
        all_source_review=args.all_source_review, data_root=args.data_root,
        split="train_large", large_offset=0, family=None))
    grouped: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for row, note in reviewed:
        if "/train/" not in "/" + row["source_game"]:
            raise ValueError("Source left the official training split")
        grouped[row["family"]].append((row, note))
    if len(grouped) != 6 or any(len(items) != 100 for items in grouped.values()):
        raise ValueError("Expected 100 reviewed train sources per ALFWorld family")
    return grouped


def _expected(args: argparse.Namespace) -> list[dict]:
    plan = json.loads(args.memrl_plan.read_text())
    alf = plan["alf"]
    if (alf["max_steps"] != 50 or alf["max_attempts"] != 3 or
            alf["test_tasks"] != 134 or 92601 not in alf["eval_seeds"]):
        raise ValueError("Historical MemRL ALFWorld budget changed")
    pool = _source_pool(args)
    training = {row["source_game"] for items in pool.values() for row, _ in items}
    rows = []
    for sequence in alf["sequences"]:
        if sequence["repeat"] != 92601:
            continue
        for item in sequence["tasks"]:
            game = item["path"]
            if (item["split"] != "valid_unseen" or game in training or
                    item["family"] != sequence["family"] or
                    file_hash(args.data_root / game) != item["sha256"]):
                raise ValueError("MemRL target split or content changed")
            source, note = min(pool[item["family"]], key=lambda pair: digest([
                "memrl134_category_source_v1", game, pair[0]["source_game"]]))
            content = {
                "game": game, "family": item["family"],
                "game_sha256": item["sha256"],
                "source_game": source["source_game"],
                "source_game_sha256": source["source_game_sha256"],
                "source_review_input_sha256": source["input_content_sha256"],
                "source_records_sha256": note["source_records_sha256"],
                "source_records": note["source_records"],
            }
            rows.append({**content, "input_content_sha256": digest(content)})
    if len(rows) != 134 or len({row["game"] for row in rows}) != 134:
        raise ValueError("Expected 134 distinct valid_unseen target games")
    return rows


def prepare(args: argparse.Namespace) -> None:
    if args.target_review.exists():
        raise FileExistsError(args.target_review)
    rows = _expected(args)
    _save(args.target_review, {
        "protocol": "Fresh content-bound MemRL 134 valid_unseen targets and deterministic same-family reviewed train expert sources; no target walkthrough or outcome used",
        "memrl_plan_sha256": file_hash(args.memrl_plan),
        "source_candidates_sha256": file_hash(args.source_candidates),
        "source_review_sha256": file_hash(args.source_review),
        "targets": rows,
    })
    print(json.dumps({"reviewed_targets": len(rows),
                      "families": dict((family, sum(row["family"] == family
                        for row in rows)) for family in sorted({r["family"]
                                                          for r in rows}))}), flush=True)


def checked_targets(args: argparse.Namespace) -> list[dict]:
    review = json.loads(args.target_review.read_text())
    if (review["memrl_plan_sha256"] != file_hash(args.memrl_plan) or
            review["source_candidates_sha256"] !=
                file_hash(args.source_candidates) or
            review["source_review_sha256"] != file_hash(args.source_review) or
            review["targets"] != _expected(args)):
        raise ValueError("Reviewed MemRL 134 target or source content changed")
    return review["targets"]


def evaluate(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    targets = checked_targets(args)
    if args.family:
        targets = [row for row in targets if row["family"] == args.family]
    targets = targets[args.offset:args.offset + args.limit]
    if not targets:
        raise ValueError("No selected valid_unseen games")
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    result = {
        "protocol": "MemRL frozen 134 valid_unseen game set, seed-order 92601; one deterministic reviewed same-family train expert history per game; paired no-LoRA versus dynamic LoRA; 50 commands, 64 new tokens, constrained greedy actor with 2-turn compact history; official won; single attempt per arm (historical MemRL used 3 attempts, sampled actor, different prompting)",
        "memrl_plan_sha256": file_hash(args.memrl_plan),
        "target_review_sha256": file_hash(args.target_review),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "runner_sha256": file_hash(Path(__file__)),
        "source_encoder": agent.encoder_kind,
        "contextual_source_max_tokens": args.contextual_source_max_tokens,
        "family": args.family, "offset": args.offset, "limit": args.limit,
        "max_steps": 50, "max_new_tokens": 64,
        "actor_history_turns": 2,
        "loop_guard_max": args.loop_guard_max,
        "games": [], "failures": [],
    }
    for row in targets:
        try:
            records = row["source_records"]
            if agent.encoder_kind == "contextual":
                fields = contextual_source_fields(
                    agent, tokenizer, records, args.device,
                    args.contextual_source_max_tokens,
                    pooling="both" if agent.task_conditioned and
                        agent.task_pair_pooling == "mean" else "last")
            else:
                fields = tokenize_records(tokenizer, records, args.device,
                                          max_tokens=args.source_max_tokens,
                                          truncation_mode="head_tail")
            game = args.data_root / row["game"]
            arms = {}
            for name, use_adapter in (("base", False), ("lora", True)):
                arms[name] = run_episode(
                    agent, tokenizer, game, fields if use_adapter else {},
                    adapter=use_adapter, device=args.device, max_steps=50,
                    max_new_tokens=64, constrain_actions=True,
                    actor_history_turns=2,
                    loop_guard_max=args.loop_guard_max)
                if arms[name]["status"] != "complete":
                    raise ValueError(f"{name} rollout failed: {arms[name]}")
            if arms["base"]["initial_observation"] != arms["lora"]["initial_observation"]:
                raise ValueError("Paired target reset changed")
            result["games"].append({
                "game": row["game"], "family": row["family"],
                "target_input_content_sha256": row["input_content_sha256"],
                "source_records_sha256": row["source_records_sha256"],
                "arms": arms,
            })
        except Exception as exc:
            result["failures"].append({"game": row["game"],
                                       "error": f"{type(exc).__name__}: {exc}"})
        complete = result["games"]
        result["summary"] = {
            "n": len(complete), "failures": len(result["failures"]),
            "base": sum(item["arms"]["base"]["reward"] for item in complete),
            "lora": sum(item["arms"]["lora"]["reward"] for item in complete),
            "by_family": {
                family: {
                    "n": sum(item["family"] == family for item in complete),
                    "base": sum(item["arms"]["base"]["reward"]
                                for item in complete if item["family"] == family),
                    "lora": sum(item["arms"]["lora"]["reward"]
                                for item in complete if item["family"] == family),
                } for family in sorted({item["family"] for item in complete})
            },
        }
        _save(args.output, result)
        print(json.dumps({"n": result["summary"]["n"],
                          "failures": result["summary"]["failures"],
                          "base": result["summary"]["base"],
                          "lora": result["summary"]["lora"]}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--memrl-plan", type=Path, default=Path("results/memrl_comparison/20260928_budgeted/plan.json"))
    parser.add_argument("--source-candidates", type=Path, default=Path("results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json"))
    parser.add_argument("--source-review", type=Path, default=Path("data/annotations/alf_sibling_train600_reviewed_20261006.json"))
    parser.add_argument("--retry-candidates", type=Path, default=Path("results/trajectory_hyperlora/alf_same_game_retry_candidates_20261006.json"))
    parser.add_argument("--retry-review", type=Path, default=Path("data/annotations/alf_same_game_retry_reviewed_20261006.json"))
    parser.add_argument("--all-source-review", type=Path, default=Path("data/annotations/alf_expert_first_attempts_reviewed_20261006.json"))
    parser.add_argument("--target-review", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("ttcl/data/alfworld_delta"))
    parser.add_argument("--model", type=Path, default=Path("current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--family")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=134)
    parser.add_argument("--contextual-source-max-tokens", type=int,
                        default=2048)
    parser.add_argument("--source-max-tokens", type=int, default=40)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.55)
    parser.add_argument("--loop-guard-max", type=int, default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.offset < 0 or args.limit < 1 or
            args.contextual_source_max_tokens < 2 or
            args.source_max_tokens < 2 or not 0 < args.gpu_fraction <= 1):
        parser.error("Invalid ALFWorld budget")
    if args.loop_guard_max is not None and args.loop_guard_max < 1:
        parser.error("Invalid loop guard threshold")
    if args.command == "prepare":
        prepare(args)
    else:
        if args.output is None:
            parser.error("evaluate needs --output")
        evaluate(args)


if __name__ == "__main__":
    main()
