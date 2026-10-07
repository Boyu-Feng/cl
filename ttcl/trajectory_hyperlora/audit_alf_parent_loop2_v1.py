"""Audit original-checkpoint LoRA with the same 134-game loop guard."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_memrl134_transfer import checked_targets
from ttcl.trajectory_hyperlora.alfworld_same_game_retry import file_hash


def audit(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    targets = checked_targets(args)
    new = json.loads(args.new_audit.read_text())
    if (new["target_review_sha256"] != file_hash(args.target_review) or
            len(new["games"]) != len(targets)):
        raise ValueError("New-checkpoint paired audit changed")
    new_by_game = {r["game"]: r for r in new["games"]}
    rows = {}
    part_hashes = {}
    for part in args.parts:
        report = json.loads(part.read_text())
        if (report["target_review_sha256"] != file_hash(args.target_review) or
                report["checkpoint_sha256"] != file_hash(args.checkpoint) or
                report["loop_guard_max"] != 2 or
                report["max_steps"] != 50 or
                report["max_new_tokens"] != 64 or
                report["actor_history_turns"] != 2 or
                report["failures"] or
                len(report["games"]) != report["limit"]):
            raise ValueError(f"Incomplete or changed parent shard: {part}")
        selected = targets[report["offset"]:report["offset"]+report["limit"]]
        if [x["game"] for x in report["games"]] != [x["game"] for x in selected]:
            raise ValueError("Parent target order changed")
        for target, row in zip(selected, report["games"], strict=True):
            if (row["game"] in rows or row["family"] != target["family"] or
                    row["target_input_content_sha256"] !=
                        target["input_content_sha256"] or
                    row["source_records_sha256"] !=
                        target["source_records_sha256"]):
                raise ValueError("Duplicate or changed parent binding")
            ep = row["lora"]
            if (ep["status"] != "complete" or ep["steps"] > 50 or
                    ep["steps"] != len(ep["trajectory"]) or
                    ep["invalid_commands"] != 0 or
                    ep["reward"] not in (0.0, 1.0) or
                    ep["initial_observation"] !=
                        new_by_game[row["game"]]["arms"]["base"]["initial_observation"]):
                raise ValueError("Invalid or unmatched parent episode")
            rows[row["game"]] = row
        part_hashes[str(part)] = file_hash(part)
    if set(rows) != set(new_by_game):
        raise ValueError("Missing or extra parent target")
    paired = [{"game": target["game"], "family": target["family"],
        "base_reward": new_by_game[target["game"]]["arms"]["base"]["reward"],
        "old_lora_reward": rows[target["game"]]["lora"]["reward"],
        "new_lora_reward": new_by_game[target["game"]]["arms"]["lora"]["reward"]}
        for target in targets]
    by_family = defaultdict(list)
    for row in paired:
        by_family[row["family"]].append(row)

    def summary(items):
        return {"n": len(items),
            "base": sum(x["base_reward"] for x in items),
            "old_lora": sum(x["old_lora_reward"] for x in items),
            "new_lora": sum(x["new_lora_reward"] for x in items),
            "new_only": sum(x["new_lora_reward"] > x["old_lora_reward"]
                            for x in items),
            "old_only": sum(x["new_lora_reward"] < x["old_lora_reward"]
                            for x in items)}

    result = {"protocol": "Paired original-versus-reward-distilled trajectory LoRA on the same frozen 134 ALFWorld targets and sources, with identical two-repeat action guard and actor budget",
        "target_review_sha256": file_hash(args.target_review),
        "old_checkpoint_sha256": file_hash(args.checkpoint),
        "new_audit_sha256": file_hash(args.new_audit),
        "part_sha256": part_hashes,
        "summary": {"overall": summary(paired),
                    "by_family": {f: summary(q) for f, q in sorted(by_family.items())}},
        "games": paired}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--new-audit", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_reward_selected_v1_loop2_official_audited_20261007.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--target-review", type=Path, default=Path(
        "data/annotations/alf_memrl134_hyperlora_train_sources_reviewed_20261006.json"))
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
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
