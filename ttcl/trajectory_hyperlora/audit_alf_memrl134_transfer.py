"""Merge and audit two disjoint MemRL-134 trajectory-LoRA rollout shards."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from math import comb
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_same_game_retry import file_hash


def exact_sign_p(gains: int, losses: int) -> float:
    n = gains + losses
    if not n:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(gains, losses) + 1)) / 2**n)


def audited_summary(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["family"]].append(row)

    def score(items: list[dict]) -> dict:
        base = sum(item["arms"]["base"]["reward"] for item in items)
        lora = sum(item["arms"]["lora"]["reward"] for item in items)
        gains = sum(item["arms"]["base"]["reward"] == 0 and
                    item["arms"]["lora"]["reward"] == 1 for item in items)
        losses = sum(item["arms"]["base"]["reward"] == 1 and
                     item["arms"]["lora"]["reward"] == 0 for item in items)
        return {"n": len(items), "base": base, "lora": lora,
                "gains": gains, "losses": losses,
                "exact_sign_p_two_sided": exact_sign_p(gains, losses)}

    return {"overall": score(rows),
            "by_family": {family: score(items)
                          for family, items in sorted(groups.items())}}


def audit(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    prefix = json.loads(args.prefix.read_text())
    suffix = json.loads(args.suffix.read_text())
    review = json.loads(args.target_review.read_text())
    if (prefix["offset"] != 0 or suffix["offset"] != 90 or
            len(prefix["games"]) != 90 or len(suffix["games"]) != 44 or
            prefix["failures"] or suffix["failures"] or
            prefix["target_review_sha256"] != file_hash(args.target_review) or
            suffix["target_review_sha256"] != file_hash(args.target_review) or
            prefix["checkpoint_sha256"] != suffix["checkpoint_sha256"] or
            prefix["runner_sha256"] != suffix["runner_sha256"] or
            prefix["max_steps"] != 50 or suffix["max_steps"] != 50 or
            prefix["max_new_tokens"] != 64 or suffix["max_new_tokens"] != 64 or
            prefix["actor_history_turns"] != 2 or
            suffix["actor_history_turns"] != 2):
        raise ValueError("Shard lineage, budget, or completion changed")
    rows = prefix["games"] + suffix["games"]
    targets = review["targets"]
    if len(rows) != 134 or len(targets) != 134 or len({row["game"] for row in rows}) != 134:
        raise ValueError("Shard merge missing or repeated targets")
    for row, target in zip(rows, targets, strict=True):
        if (row["game"] != target["game"] or
                row["family"] != target["family"] or
                row["target_input_content_sha256"] != target["input_content_sha256"] or
                row["source_records_sha256"] != target["source_records_sha256"] or
                any(arm["status"] != "complete" or arm["steps"] > 50 or
                    arm["reward"] not in (0., 1.)
                    for arm in row["arms"].values()) or
                row["arms"]["base"]["initial_observation"] !=
                    row["arms"]["lora"]["initial_observation"]):
            raise ValueError(f"Invalid paired result for {target['game']}")
    result = {
        "protocol": "Audited merge of frozen MemRL 134 valid_unseen single-attempt paired LoRA probe; prefix process intentionally stopped after exactly 90 completed rows, suffix covers remaining 44; no failures or duplicated games",
        "target_review_sha256": file_hash(args.target_review),
        "prefix_sha256": file_hash(args.prefix),
        "suffix_sha256": file_hash(args.suffix),
        "checkpoint_sha256": prefix["checkpoint_sha256"],
        "runner_sha256": prefix["runner_sha256"],
        "max_steps": 50, "max_new_tokens": 64,
        "actor_history_turns": 2,
        "summary": audited_summary(rows),
        "games": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--suffix", type=Path, required=True)
    parser.add_argument("--target-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
