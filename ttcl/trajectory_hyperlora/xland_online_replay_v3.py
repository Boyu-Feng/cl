"""Replay fixed audited source histories through a new frozen hypernetwork.

This separates adapter-training effects from changes in online exploration:
old and new checkpoints receive identical selected source transitions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.xland_online_from_empty import (
    digest, load_agent, predict, sha256, source_effect,
)


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    review = json.loads(args.review.read_text())
    old = json.loads(args.old_online.read_text())
    if (review["target_count"] != 90 or
            old["review_sha256"] != sha256(args.review) or
            old["trial_budget"] != 3 or
            old["summary"]["n"] != 90 or len(old["rows"]) != 90):
        raise ValueError("Old online history lineage changed")
    audited = {(item["item_id"], kind): item["arms"][kind]
               for item in review["targets"] for kind in item["arms"]}
    agent, tokenizer, choice_ids = load_agent(args)
    output = {
        "protocol": "Fixed-source-history counterfactual: new hypernetwork predicts target after each exact old online selected source prefix; selected actions and audited transitions unchanged; no target outcomes fed back",
        "review_sha256": sha256(args.review),
        "old_online_sha256": sha256(args.old_online),
        "new_checkpoint_sha256": sha256(args.checkpoint),
        "runner_sha256": sha256(Path(__file__)),
        "rows": [],
    }
    for previous in old["rows"]:
        arm = audited[(previous["item_id"], previous["kind"])]
        history = []
        rounds = []
        for entry in previous["rounds"]:
            if (entry["history_sha256"] != digest(history) or
                    entry["history_length"] != len(history) or
                    entry["correct_action"] != arm["target_action"]):
                raise ValueError("Old prefix or reviewed target changed")
            action, content_hash = predict(
                agent, tokenizer, choice_ids, arm["target_state"],
                arm["goal"], history, args.device)
            if content_hash != entry["target_input_sha256"]:
                raise ValueError("New target input differs from old bound input")
            new_correct = source_effect(
                arm["target_transitions"][str(action)], arm["goal"])
            rounds.append({"round": entry["round"],
                           "old_action": entry["target_action"],
                           "old_correct": entry["target_product_effect"],
                           "new_action": action, "new_correct": new_correct,
                           "target_input_sha256": content_hash})
            if entry["round"] < 3:
                trial_action = entry["own_trial_action"]
                source_content = {"source_episodes": history,
                    "target_initial_state": arm["source_state"],
                    "goal": arm["goal"]}
                step = arm["source_trials"][str(trial_action)]
                if (entry["own_trial_input_sha256"] != digest(source_content) or
                        entry["own_trial_step_sha256"] != digest(step)):
                    raise ValueError("Old selected source transition changed")
                history.append({"goal": arm["goal"], "steps": [step]})
        output["rows"].append({"item_id": previous["item_id"],
                               "kind": previous["kind"], "rounds": rounds})
    output["summary"] = {
        "n": len(output["rows"]),
        "by_round": [{"round": turn,
            "old": sum(row["rounds"][turn]["old_correct"]
                       for row in output["rows"]),
            "new": sum(row["rounds"][turn]["new_correct"]
                       for row in output["rows"]),
            "new_only": sum(row["rounds"][turn]["new_correct"] and
                            not row["rounds"][turn]["old_correct"]
                            for row in output["rows"]),
            "old_only": sum(row["rounds"][turn]["old_correct"] and
                            not row["rounds"][turn]["new_correct"]
                            for row in output["rows"])}
            for turn in range(4)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output["summary"]), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/xland_multi_novelty_v2_reviewed_20261007.json"))
    parser.add_argument("--old-online", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-annotations", type=Path, default=Path(
        "data/annotations/xland_multi_mechanism_reviewed_v1_20261005.json"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.55)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.gpu_fraction <= 1:
        parser.error("Invalid GPU memory fraction")
    run(args)


if __name__ == "__main__":
    main()
