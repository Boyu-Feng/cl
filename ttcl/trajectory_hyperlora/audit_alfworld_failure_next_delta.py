"""Independent ALFWorld replay for the v34/v35 next-task Delta-B pilots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay


def audit(args):
    review = json.loads(args.review.read_text())
    report = json.loads(args.report.read_text())
    if (review["targets_sha256"] != digest(review["targets"]) or
            report["review_sha256"] != file_hash(args.review) or
            report["failures"] or len(report["rows"]) != len(review["targets"]) or
            any(report.get(k) != review[k] for k in (
                "method", "checkpoint_sha256", "runner_sha256", "actor_sha256"))):
        raise ValueError("Changed reviewed run")
    totals = {"baseline": 0.}
    if args.mode == "v34":
        totals.update(toward_failure=0., away_from_failure=0.)
    else:
        totals["transfer"] = 0.
    for binding, row in zip(review["targets"], report["rows"], strict=True):
        if (row["source_index"] != binding["source_index"] or
                row["target_index"] != binding["target_index"] or
                row["input_content_sha256"] != binding["input_content_sha256"] or
                row["baseline_episode_sha256"] !=
                    binding["baseline_episode_sha256" if args.mode == "v34"
                            else "target_baseline_episode_sha256"] or
                file_hash(args.data_root / binding["target_game"]) !=
                    binding["target_game_sha256"]):
            raise ValueError("Changed target or baseline binding")
        totals["baseline"] += row["baseline_reward"]
        if args.mode == "v34":
            episodes = {name: value["episode"] for name, value in row["arms"].items()}
            if set(episodes) != {"toward_failure", "away_from_failure"}:
                raise ValueError("Changed v34 arms")
        else:
            episodes = {"transfer": row["transfer_episode"]}
            if (row["residual_info"]["residual_effective_norm"] >
                    .25 * row["residual_info"]["expert_effective_norm"] + 1e-6):
                raise ValueError("Residual exceeded effective LoRA bound")
        for name, episode in episodes.items():
            if episode["status"] != "complete" or episode["steps"] > 50:
                raise ValueError("Incomplete or over-budget episode")
            replay(args.data_root / binding["target_game"], episode)
            totals[name] += episode["reward"]
    expected = {"pairs": len(report["rows"]), **totals, "failures": 0}
    if report.get("summary") != expected:
        raise ValueError("Changed official outcome totals")
    return {"audit": "passed", "summary": expected,
        "replayed_episodes": len(report["rows"]) * (2 if args.mode == "v34" else 1),
        "report_sha256": file_hash(args.report),
        "review_sha256": file_hash(args.review)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("v34", "v35"))
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("ttcl/data/alfworld_delta"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
