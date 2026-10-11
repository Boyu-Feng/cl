"""Recompute v40 pre-attempt corrections and replay every new ALFWorld episode."""

from __future__ import annotations

import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import bounded_readout, save
from ttcl.trajectory_hyperlora.train_alfworld_failure_distill_v40 import preattempt, predict
from ttcl.trajectory_hyperlora.evaluate_alfworld_failure_distill_v40 import (
    THRESHOLDS, binding, checked_model, inputs, parser,
)
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay


def audit(args):
    samples = inputs(args)
    fixed, targets = binding(args, samples)
    review = json.loads(args.review.read_text())
    report = json.loads(args.output.read_text())
    if (review.get("targets") != targets or review.get("targets_sha256") != digest(targets)
            or any(review.get(k) != v or report.get(k) != v for k, v in fixed.items())
            or report.get("review_sha256") != file_hash(args.review)
            or len(report.get("rows", [])) != len(samples) or report.get("failures")):
        raise ValueError("Changed or incomplete v40 run")
    model = checked_model(args)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    old_wins = 0
    modified_wins = 0
    activated = 0
    for i, ((game, _, records, _, _, old), row, binding_row) in enumerate(zip(
            samples, report["rows"], targets, strict=True)):
        if (row["index"] != i or row["game"] != game or
                row["input_content_sha256"] != binding_row["input_content_sha256"] or
                row["old_first_episode_sha256"] != digest(old) or
                row["old_first_reward"] != old["reward"]):
            raise ValueError(f"Changed target at {i}")
        latent, expert, _ = preattempt(agent, tokenizer, records,
                                      old["initial_observation"], args.device)
        probability, predicted = predict(model, latent)
        applied, info = bounded_readout(agent, predicted, expert)
        if (row["prelatent_sha256"] != digest(latent.tolist()) or
                row["expert_factor_sha256"] != factors_hash(expert) or
                row["predicted_factor_sha256"] != factors_hash(predicted) or
                row["applied_factor_sha256"] != factors_hash(applied) or
                abs(row["gate_probability"] - probability) > 1e-7 or
                row["read_info"] != info or
                info["applied_effective_norm"] > .25 * info["reference_effective_norm"] + 1e-6):
            raise ValueError(f"Predicted correction changed at {i}")
        expected_active = args.split == "dev" or probability >= fixed["selected_threshold"]
        if row["activated"] != expected_active:
            raise ValueError(f"Gate changed at {i}")
        ep = row["episode"]
        if expected_active:
            if (ep is None or ep["status"] != "complete" or ep["steps"] > 50 or
                    ep["initial_observation"] != old["initial_observation"]):
                raise ValueError(f"Invalid episode at {i}")
            replay(args.data_root / game, ep)
            modified_wins += ep["reward"]
            activated += 1
        elif ep is not None:
            raise ValueError(f"Unexpected modified episode at {i}")
        old_wins += old["reward"]
    summary = {"n": len(samples), "old_first_wins": old_wins,
        "activated": activated, "modified_attempt_wins": modified_wins,
        "failures": 0}
    if args.split == "dev":
        scores = {str(t): sum(x["episode"]["reward"] if x["gate_probability"] >= t
                      else x["old_first_reward"] for x in report["rows"])
                  for t in THRESHOLDS}
        selected = max(THRESHOLDS, key=lambda x: (scores[str(x)], x))
        if (report["summary"] != summary or report["threshold_scores"] != scores
                or report["selected_threshold"] != selected):
            raise ValueError("Development threshold selection changed")
    else:
        summary["gated_one_attempt_wins"] = sum(
            x["episode"]["reward"] if x["activated"] else x["old_first_reward"]
            for x in report["rows"])
        if report["summary"] != summary:
            raise ValueError("Unseen official score changed")
    return {"audit": "passed", "split": args.split, "summary": summary,
        "replayed_episodes": activated, "report_sha256": file_hash(args.output),
        "review_sha256": file_hash(args.review)}


if __name__ == "__main__":
    p = parser()
    p.add_argument("--audit-output", type=Path, required=True)
    args = p.parse_args()
    if args.command != "evaluate" or args.audit_output.exists():
        raise ValueError("Audit requires evaluate and fresh output")
    result = audit(args)
    save(args.audit_output, result)
    print(json.dumps(result), flush=True)
