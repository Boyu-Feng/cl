"""Collect same-protocol ALFWorld pairs to train a text-only LoRA-use router.

The router's inputs are task/source text. ALFWorld family names only balance
train/dev sampling; they are never router features. Reviewed source and target
contents are rebound whenever a new pair is created.
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


def reviewed_pool(args):
    return checked_review(argparse.Namespace(
        candidates=args.candidates, source_review=args.source_review,
        retry_candidates=args.retry_candidates, review=args.retry_review,
        all_source_review=args.all_source_review, data_root=args.data_root,
        split="train_large", large_offset=0, family=None))


def prepare(args):
    if args.reviewed_pairs.exists():
        raise FileExistsError(args.reviewed_pairs)
    pool = reviewed_pool(args)
    grouped = defaultdict(list)
    for row, note in pool:
        grouped[row["family"]].append((row, note))
    if len(grouped) != 6 or any(len(v) != 100 for v in grouped.values()):
        raise ValueError("Expected 100 reviewed train targets per family")
    selected = []
    for family, rows in sorted(grouped.items()):
        # The current LoRA checkpoint used an earlier ordinary-placement
        # subset. These late, disjoint train directories are for router data.
        for local_index in range(60, 72):
            target, _ = rows[local_index]
            target_directory = Path(target["target_game"]).parent.parent
            alternatives = [(source, note) for source, note in rows
                if Path(source["source_game"]).parent.parent != target_directory]
            source, note = min(alternatives, key=lambda pair: digest([
                "memrl134_category_source_v1", target["target_game"],
                pair[0]["source_game"]]))
            content = {"target_game": target["target_game"],
                "target_game_sha256": target["target_game_sha256"],
                "target_review_input_sha256": target["input_content_sha256"],
                "source_game": source["source_game"],
                "source_game_sha256": source["source_game_sha256"],
                "source_review_input_sha256": source["input_content_sha256"],
                "source_records_sha256": note["source_records_sha256"],
                "source_records": note["source_records"]}
            selected.append({**content, "split": "train" if local_index < 68
                                      else "dev", "family": family,
                             "input_content_sha256": digest(content),
                             "review_basis": "Both train-game files match reviewed hashes; source steps were separately replay-reviewed to official won; distinct task directories; target walkthrough unused"})
    report = {"protocol": "New content-bound ALFWorld router train/dev pairs from already reviewed train-only targets and successful source trajectories; per-family 8/4 split; no target walkthrough or valid_unseen reward used",
        "candidates_sha256": file_hash(args.candidates),
        "source_review_sha256": file_hash(args.source_review),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "pairs": selected}
    args.reviewed_pairs.parent.mkdir(parents=True, exist_ok=True)
    args.reviewed_pairs.write_text(json.dumps(report, ensure_ascii=False,
                                         indent=2) + "\n")
    print(json.dumps({"reviewed_pairs": len(selected), "train": 48,
                      "dev": 24}), flush=True)


def checked_pairs(args):
    review = json.loads(args.reviewed_pairs.read_text())
    if (review["candidates_sha256"] != file_hash(args.candidates) or
            review["source_review_sha256"] != file_hash(args.source_review) or
            review["checkpoint_sha256"] != file_hash(args.checkpoint)):
        raise ValueError("Router review lineage changed")
    pool = reviewed_pool(args)
    targets = {row["target_game"]: row for row, _ in pool}
    sources = {row["source_game"]: (row, note) for row, note in pool}
    seen = set()
    for item in review["pairs"]:
        content = {key: item[key] for key in (
            "target_game", "target_game_sha256", "target_review_input_sha256",
            "source_game", "source_game_sha256", "source_review_input_sha256",
            "source_records_sha256", "source_records")}
        target = targets[item["target_game"]]
        source, note = sources[item["source_game"]]
        if (item["target_game"] in seen or
                item["input_content_sha256"] != digest(content) or
                item["target_game_sha256"] != target["target_game_sha256"] or
                item["target_review_input_sha256"] != target["input_content_sha256"] or
                item["source_game_sha256"] != source["source_game_sha256"] or
                item["source_review_input_sha256"] != source["input_content_sha256"] or
                item["source_records_sha256"] != note["source_records_sha256"] or
                item["source_records"] != note["source_records"] or
                item["family"] != target["family"] or
                item["family"] != source["family"] or
                item["split"] not in ("train", "dev") or
                Path(item["target_game"]).parent.parent ==
                    Path(item["source_game"]).parent.parent or
                file_hash(args.data_root / item["target_game"]) !=
                    item["target_game_sha256"]):
            raise ValueError("Router target/source content binding changed")
        seen.add(item["target_game"])
    if len(seen) != 72 or sum(x["split"] == "train" for x in review["pairs"]) != 48:
        raise ValueError("Incomplete router target review")
    return review["pairs"]


def collect(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    pairs = checked_pairs(args)[args.offset:args.offset + args.limit]
    if not pairs:
        raise ValueError("Empty collection slice")
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    result = {"protocol": "Same-checkpoint ALFWorld router pairs; base/trajectory-LoRA, greedy admissible commands, 50 steps/64 tokens, 2-turn actor history, official won; no router fitted during collection",
        "reviewed_pairs_sha256": file_hash(args.reviewed_pairs),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "offset": args.offset, "limit": args.limit,
        "max_steps": 50, "max_new_tokens": 64, "actor_history_turns": 2,
        "games": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(args.output)
    for pair in pairs:
        try:
            fields = (contextual_source_fields(agent, tokenizer,
                pair["source_records"], args.device, 2048,
                pooling="both" if agent.task_conditioned and
                agent.task_pair_pooling == "mean" else "last")
                if agent.encoder_kind == "contextual" else
                tokenize_records(tokenizer, pair["source_records"], args.device,
                                 max_tokens=40, truncation_mode="head_tail"))
            arms = {}
            for arm, enabled in (("base", False), ("lora", True)):
                arms[arm] = run_episode(agent, tokenizer,
                    args.data_root / pair["target_game"],
                    fields if enabled else {}, adapter=enabled,
                    device=args.device, max_steps=50, max_new_tokens=64,
                    constrain_actions=True, actor_history_turns=2)
                if arms[arm]["status"] != "complete":
                    raise RuntimeError(f"{arm}: {arms[arm]}")
            if arms["base"]["initial_observation"] != arms["lora"]["initial_observation"]:
                raise RuntimeError("Paired reset changed")
            result["games"].append({"game": pair["target_game"],
                "split": pair["split"], "family": pair["family"],
                "input_content_sha256": pair["input_content_sha256"],
                "source_records_sha256": pair["source_records_sha256"],
                "arms": arms})
        except Exception as exc:
            result["failures"].append({"game": pair["target_game"],
                "error": f"{type(exc).__name__}: {exc}"})
        save()
        print(json.dumps({"completed": len(result["games"]),
            "failures": len(result["failures"]),
            "last": pair["target_game"],
            "base": result["games"][-1]["arms"]["base"]["reward"]
                if result["games"] and result["games"][-1]["game"] ==
                    pair["target_game"] else None,
            "lora": result["games"][-1]["arms"]["lora"]["reward"]
                if result["games"] and result["games"][-1]["game"] ==
                    pair["target_game"] else None}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "collect"))
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train600_reviewed_20261006.json"))
    parser.add_argument("--retry-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_same_game_retry_candidates_20261006.json"))
    parser.add_argument("--retry-review", type=Path, default=Path(
        "data/annotations/alf_same_game_retry_reviewed_20261006.json"))
    parser.add_argument("--all-source-review", type=Path, default=Path(
        "data/annotations/alf_expert_first_attempts_reviewed_20261006.json"))
    parser.add_argument("--reviewed-pairs", type=Path, default=Path(
        "data/annotations/alf_reward_router_train72_reviewed_20261007.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--data-root", type=Path, default=Path("ttcl/data/alfworld_delta"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.65)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=72)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        if args.output is None or args.offset < 0 or args.limit < 1:
            parser.error("Collect needs fresh output and positive offset/limit")
        collect(args)


if __name__ == "__main__":
    main()
