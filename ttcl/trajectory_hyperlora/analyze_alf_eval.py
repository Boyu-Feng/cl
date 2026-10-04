"""Audit paired success on one frozen ALFWorld evaluation rollout."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from ttcl.trajectory_hyperlora.prepare_alf_eval_pairs import validate_eval_review


def paired_counts(rows: list[dict], left: str, right: str) -> dict:
    result = Counter((int(row[left]["reward"]), int(row[right]["reward"]))
                     for row in rows)
    positive = result[(0, 1)]
    negative = result[(1, 0)]
    discordant = positive + negative
    exact_two_sided_p = min(1.0, 2.0 * sum(
        math.comb(discordant, k) for k in range(
            max(positive, negative), discordant + 1)) / (2 ** discordant)
        ) if discordant else 1.0
    return {"n": len(rows),
            "left_successes": result[(1, 0)] + result[(1, 1)],
            "right_successes": result[(0, 1)] + result[(1, 1)],
            "right_only": positive, "left_only": negative,
            "both_success": result[(1, 1)], "both_fail": result[(0, 0)],
            "paired_success_difference": positive - negative,
            "exact_two_sided_sign_p": exact_two_sided_p}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path(
        "ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json"))
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_eval_pairs_20261005_candidates.json"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/trajectory_hyperlora_alf_eval_pairs_20261005_reviewed.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; do not overwrite frozen analysis")
    manifest = json.loads(args.candidates.read_text())
    reviewed = validate_eval_review(manifest, json.loads(args.review.read_text()),
                                    args.plan, args.data_root)
    bound = {row["target_game"]: row for row in reviewed}
    rollout = json.loads(args.rollout.read_text())
    if (rollout["plan_sha256"] != hashlib.sha256(args.plan.read_bytes()).hexdigest()
            or rollout["eval_manifest_sha256"] != hashlib.sha256(
                args.candidates.read_bytes()).hexdigest()
            or rollout["eval_review_sha256"] != hashlib.sha256(
                args.review.read_bytes()).hexdigest()
            or rollout["max_steps"] != 30 or rollout["max_new_tokens"] != 64):
        raise ValueError("Frozen evaluation bindings or budgets changed")
    rows = rollout["games"]
    if len(rows) != len(bound) or {row["game"] for row in rows} != set(bound):
        raise ValueError("Evaluation rollout is incomplete or has extra games")
    for row in rows:
        reviewed_row = bound[row["game"]]
        if (row["game_sha256"] != reviewed_row["target_game_sha256"] or
                row["input_content_sha256"] != reviewed_row["input_content_sha256"]):
            raise ValueError("Evaluation row input binding changed")
        for arm in ("base", "mean_adapter"):
            if row[arm]["status"] != "complete":
                raise ValueError("Infrastructure failure cannot be scored as zero")
        if row["source_game"] is None:
            if row["source_adapter"] != "unsupported_no_reviewed_train_source":
                raise ValueError("Missing source arm has an unexpected result")
        elif row["source_adapter"]["status"] != "complete":
            raise ValueError("Source arm infrastructure failure")
    supported = [row for row in rows if row["source_game"] is not None]
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    result = {"protocol": "One-shot paired frozen ALFWorld evaluation analysis; primary base versus global mean LoRA across all 36 games; no training on evaluation outcomes",
              "rollout_sha256": hashlib.sha256(args.rollout.read_bytes()).hexdigest(),
              "checkpoint_sha256": rollout["checkpoint_sha256"],
              "mean_adapter_sha256": rollout["mean_adapter_sha256"],
              "primary_base_vs_mean": paired_counts(rows, "base", "mean_adapter"),
              "supported_base_vs_source": paired_counts(
                  supported, "base", "source_adapter"),
              "supported_mean_vs_source": paired_counts(
                  supported, "mean_adapter", "source_adapter"),
              "families": {family: paired_counts(items, "base", "mean_adapter")
                           for family, items in sorted(by_family.items())}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("families", "protocol", "rollout_sha256",
                                     "checkpoint_sha256", "mean_adapter_sha256")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
