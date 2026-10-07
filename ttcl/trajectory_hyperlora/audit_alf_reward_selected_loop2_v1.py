"""Audit all shards of the reward-distilled, loop-guarded ALFWorld evaluation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from math import comb
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_memrl134_transfer import checked_targets
from ttcl.trajectory_hyperlora.alfworld_same_game_retry import file_hash


def summarize(rows):
    gains = sum(r["arms"]["lora"]["reward"] > r["arms"]["base"]["reward"]
                for r in rows)
    losses = sum(r["arms"]["lora"]["reward"] < r["arms"]["base"]["reward"]
                 for r in rows)
    discordant = gains + losses
    p = min(1.0, 2 * sum(comb(discordant, k) for k in range(
        min(gains, losses) + 1)) / (2 ** discordant)) if discordant else 1.0
    return {"n": len(rows),
            "base": sum(r["arms"]["base"]["reward"] for r in rows),
            "lora": sum(r["arms"]["lora"]["reward"] for r in rows),
            "gains": gains, "losses": losses,
            "exact_sign_p_two_sided": p}


def audit(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    targets = checked_targets(args)
    loop_runner = Path(__file__).with_name("alfworld_memrl134_loop_guard_v1.py")
    by_game = {}
    shards = {}
    for part in args.parts:
        data = json.loads(part.read_text())
        if (data["memrl_plan_sha256"] != file_hash(args.memrl_plan) or
                data["target_review_sha256"] != file_hash(args.target_review) or
                data["checkpoint_sha256"] != file_hash(args.checkpoint) or
                data["runner_sha256"] != file_hash(loop_runner) or
                data["loop_guard_max"] != 2 or
                data["max_steps"] != 50 or
                data["max_new_tokens"] != 64 or
                data["actor_history_turns"] != 2 or
                data["failures"] or data["family"] is not None or
                len(data["games"]) != data["limit"]):
            raise ValueError(f"Changed, incomplete, or failed shard: {part}")
        expected_slice = targets[data["offset"]:data["offset"]+data["limit"]]
        if [r["game"] for r in data["games"]] != [r["game"] for r in expected_slice]:
            raise ValueError(f"Target order or slice changed: {part}")
        for target, row in zip(expected_slice, data["games"], strict=True):
            if (row["game"] in by_game or row["family"] != target["family"] or
                    row["target_input_content_sha256"] !=
                        target["input_content_sha256"] or
                    row["source_records_sha256"] !=
                        target["source_records_sha256"]):
                raise ValueError("Duplicate target or changed source binding")
            arms = row["arms"]
            if (set(arms) != {"base", "lora"} or
                    arms["base"]["initial_observation"] !=
                        arms["lora"]["initial_observation"]):
                raise ValueError("Missing paired arms or inconsistent reset")
            for arm in arms.values():
                if (arm["status"] != "complete" or arm["steps"] > 50 or
                        arm["steps"] != len(arm["trajectory"]) or
                        arm["invalid_commands"] != 0 or
                        arm["reward"] not in (0.0, 1.0)):
                    raise ValueError("Invalid official episode")
            by_game[row["game"]] = row
        shards[str(part)] = file_hash(part)
    if set(by_game) != {r["game"] for r in targets}:
        raise ValueError("Missing or extra official target")
    ordered = [by_game[target["game"]] for target in targets]
    families = defaultdict(list)
    for row in ordered:
        families[row["family"]].append(row)
    result = {"protocol": "Complete, paired 134 valid_unseen official-won audit; same fixed target/source binding and actor budget; frozen reward-distilled trajectory LoRA with 2-repeat observation-action guard applied identically to base and LoRA",
        "target_review_sha256": file_hash(args.target_review),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "shard_sha256": shards,
        "summary": {"overall": summarize(ordered),
                    "by_family": {f: summarize(rows) for f, rows in sorted(families.items())}},
        "games": ordered}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
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
