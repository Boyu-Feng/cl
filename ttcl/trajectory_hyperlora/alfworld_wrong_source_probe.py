"""Test whether ALFWorld LoRA gains depend on the chosen source trajectory."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.prepare_alf_next_task import (
    digest_json, family, validate_review,
)
from ttcl.trajectory_hyperlora.relational_router_pilot import tokenize_records


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", action="append", type=Path, required=True)
    parser.add_argument("--historical-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_candidates_20261005.json"))
    parser.add_argument("--historical-review", type=Path, default=Path(
        "data/annotations/trajectory_hyperlora_alf_20261005_reviewed_v2.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.40)
    parser.add_argument("--source-mismatch", choices=("same_family", "different_family"),
                        default="same_family")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; do not overwrite a frozen probe")
    checkpoint_sha = sha256(args.checkpoint)
    source_rows = validate_review(
        json.loads(args.historical_candidates.read_text()),
        json.loads(args.historical_review.read_text()))
    by_family = defaultdict(dict)
    for row in source_rows:
        if row["split"] == "train":
            by_family[row["family"]][row["source_game"]] = row
    positives = []
    seen = set()
    settings = None
    for path in args.reference:
        report = json.loads(path.read_text())
        if report["checkpoint_sha256"] != checkpoint_sha or \
                report.get("constrain_actions") is not True or \
                report.get("adapter_scale") != 1.0:
            raise ValueError("Reference checkpoint or action protocol mismatch")
        frozen = (report["max_steps"], report["max_new_tokens"])
        if settings is not None and frozen != settings:
            raise ValueError("Reference action budgets differ")
        settings = frozen
        for row in report["games"]:
            if row["game"] in seen:
                raise ValueError("Duplicate reference target")
            seen.add(row["game"])
            if row["base"]["status"] != "complete" or \
                    row["generated"]["status"] != "complete":
                raise ValueError("Incomplete reference pair")
            if row["base"]["reward"] == 0 and row["generated"]["reward"] == 1:
                positives.append(row)
    selected, exclusions = [], []
    for row in positives:
        target_family = family(row["game"])
        if args.source_mismatch == "same_family":
            pool = by_family[target_family].values()
        else:
            pool = (source for other_family, family_sources in by_family.items()
                    if other_family != target_family
                    for source in family_sources.values())
        choices = [source for source in pool if source["source_game"] not in (
            row["source_game"], row["game"])]
        if not choices:
            exclusions.append({"game": row["game"],
                               "reason": "no_distinct_reviewed_train_source"})
            continue
        source = min(choices, key=lambda source: digest_json(
            ["wrong_source", args.source_mismatch, row["game"],
             source["source_game"]]))
        game_path = args.data_root / row["game"]
        if sha256(game_path) != row["game_sha256"]:
            raise ValueError("Reference target game hash changed")
        selected.append((row, source, game_path))
    if not selected:
        raise ValueError("No source-specificity probes with reviewed alternatives")
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    result = {"protocol": "Train-reviewed source-swap control for base-failed, correct-source-LoRA-won ALFWorld games; same actor, target, reset, admissible-command constraint and 50-step budget",
              "source_mismatch": args.source_mismatch,
              "checkpoint_sha256": checkpoint_sha,
              "reference_sha256": {str(path): sha256(path) for path in args.reference},
              "historical_review_sha256": sha256(args.historical_review),
              "max_steps": settings[0], "max_new_tokens": settings[1],
              "exclusions": exclusions, "games": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for row, source, game_path in selected:
        fields = tokenize_records(tokenizer, source["source_records"], args.device)
        wrong = run_episode(agent, tokenizer, game_path, fields,
                            adapter=True, device=args.device,
                            max_steps=settings[0], max_new_tokens=settings[1],
                            constrain_actions=True)
        result["games"].append({
            "game": row["game"], "game_sha256": row["game_sha256"],
            "correct_source_game": row["source_game"],
            "wrong_source_game": source["source_game"],
            "wrong_source_episode_sha256": source["source_episode_sha256"],
            "wrong_source_reviewed_input_sha256": source["input_content_sha256"],
            "base_reward": row["base"]["reward"],
            "correct_source_reward": row["generated"]["reward"],
            "wrong_source": wrong})
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"game": row["game"],
                          "wrong_source_reward": wrong.get("reward"),
                          "wrong_source_status": wrong["status"]}), flush=True)


if __name__ == "__main__":
    main()
