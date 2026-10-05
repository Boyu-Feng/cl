"""Swap hidden mechanisms while preserving each online actor's trial actions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.collect_xland_multi_mechanism import KINDS
from ttcl.trajectory_hyperlora.xland_online_from_empty import (
    digest, load_agent, predict, sha256,
)


def history_for(arm: dict, actions: list[int]) -> list[dict]:
    return [{"goal": arm["goal"],
             "steps": [arm["source_trials"][str(action)]]}
            for action in actions]


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.online_report.read_bytes()
    report = json.loads(raw)
    review = json.loads(args.review.read_text())
    if (report["review_sha256"] != sha256(args.review) or
            report["checkpoint_sha256"] != sha256(args.checkpoint) or
            len(report["rows"]) != 3 * len(review["targets"])):
        raise ValueError("Online rollout lineage or completeness changed")
    agent, tokenizer, choice_ids = load_agent(args)
    rows = []
    for index, item in enumerate(review["targets"]):
        for arm_index, kind in enumerate(KINDS):
            original = report["rows"][index * len(KINDS) + arm_index]
            if original["item_id"] != item["item_id"] or \
                    original["kind"] != kind:
                raise ValueError("Online result order changed")
            arm = item["arms"][kind]
            wrong_kind = KINDS[(arm_index + 1) % len(KINDS)]
            other = item["arms"][wrong_kind]
            if (other["goal"] != arm["goal"] or
                    other["source_state"] != arm["source_state"] or
                    other["target_state"] != arm["target_state"]):
                raise ValueError("Counterfactual changes public task")
            actions = [step["own_trial_action"] for step in
                       original["rounds"][:-1]]
            correct_history = history_for(arm, actions)
            wrong_history = history_for(other, actions)
            correct, correct_hash = predict(agent, tokenizer, choice_ids,
                arm["target_state"], arm["goal"], correct_history, args.device)
            wrong, wrong_hash = predict(agent, tokenizer, choice_ids,
                arm["target_state"], arm["goal"], wrong_history, args.device)
            if (correct != original["rounds"][-1]["target_action"] or
                    correct_hash != original["rounds"][-1]["target_input_sha256"]):
                raise RuntimeError("Correct-history rerun differs")
            rows.append({"item_id": item["item_id"], "kind": kind,
                         "wrong_kind": wrong_kind,
                         "trial_actions": actions,
                         "correct_source_sha256": digest(correct_history),
                         "wrong_source_sha256": digest(wrong_history),
                         "correct_prediction": correct,
                         "wrong_prediction": wrong,
                         "wrong_input_sha256": wrong_hash,
                         "target_action": arm["target_action"]})
    n = len(rows)
    result = {"protocol": "Exploratory counterfactual feedback swap on frozen from-empty online XLand trials; exact same model-selected action sequence, public state/goal, checkpoint and target",
              "online_report_sha256": hashlib.sha256(raw).hexdigest(),
              "review_sha256": sha256(args.review),
              "checkpoint_sha256": sha256(args.checkpoint),
              "summary": {"n": n,
                          "correct_history": sum(x["correct_prediction"] ==
                                                 x["target_action"] for x in rows),
                          "wrong_feedback": sum(x["wrong_prediction"] ==
                                                x["target_action"] for x in rows),
                          "prediction_changes": sum(x["correct_prediction"] !=
                                                    x["wrong_prediction"] for x in rows)},
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False,
                                      indent=2) + "\n")
    print(json.dumps(result["summary"]), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-report", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-annotations", type=Path, default=Path(
        "data/annotations/xland_multi_mechanism_reviewed_v1_20261005.json"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.55)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
