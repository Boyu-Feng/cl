"""Replay v39 official unseen episodes and reconstruct every memory transition."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_failure_retry_residual_v31b import bounded_residual
from ttcl.trajectory_hyperlora.alfworld_success_gated_factors_v27 import episode_factor
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_source_fields
from ttcl.trajectory_hyperlora.evaluate_alfworld_failure_retry_v31b import source_factors
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import bounded_readout, update_state, save
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v39_134 import bound, parser
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay


def effective_cosine(agent, left, right):
    if left is None or right is None:
        return None
    dot = 0.
    left_sq = 0.
    right_sq = 0.
    for layer, a, b in zip(agent.adapters, left, right, strict=True):
        x = a.squeeze(0).to(layer.a.device, dtype=layer.a.dtype).float()
        y = b.squeeze(0).to(layer.a.device, dtype=layer.a.dtype).float()
        gram = layer.a.detach().float() @ layer.a.detach().float().T
        scale = layer.rank ** 2
        dot += float(((x.T @ y) * gram).sum()) / scale
        left_sq += float(((x.T @ x) * gram).sum()) / scale
        right_sq += float(((y.T @ y) * gram).sum()) / scale
    return dot / math.sqrt(left_sq * right_sq) if left_sq > 0 and right_sq > 0 else None


def audit(args):
    targets, first, fixed, bindings = bound(args)
    review = json.loads(args.review.read_text())
    report = json.loads(args.output.read_text())
    if (review.get("targets") != bindings or review.get("targets_sha256") != digest(bindings)
            or any(review.get(k) != v or report.get(k) != v for k, v in fixed.items())
            or report.get("review_sha256") != file_hash(args.review)
            or len(report.get("rows", [])) != 134 or report.get("failures")):
        raise ValueError("Changed or incomplete v39 report")
    agent, tokenizer = load_agent(args.model, args.checkpoint, args.device, args.gpu_fraction)
    state = None
    count = 0
    wins = 0
    old_wins = 0
    old_local_alignments = []
    for i, (target, old, binding, row) in enumerate(zip(
            targets, first, bindings, report["rows"], strict=True)):
        if (row["index"] != i or row["game"] != binding["game"] or
                row["input_content_sha256"] != binding["input_content_sha256"] or
                row["old_first_episode_sha256"] != digest(old["lora"]) or
                row["old_first_reward"] != old["lora"]["reward"] or
                file_hash(args.data_root / row["game"]) != binding["game_sha256"]):
            raise ValueError(f"Changed target at {i}")
        fields = contextual_source_fields(agent, tokenizer, target["source_records"],
                                          args.device, 2048, pooling="both")
        initial = old["lora"]["initial_observation"]
        expert = source_factors(agent, tokenizer, fields, initial, args.device)
        if (row["expert_factor_sha256"] != factors_hash(expert) or
                row["state_sha256_before"] != factors_hash(state) or
                row["failure_updates_before"] != count):
            raise ValueError(f"Changed read inputs at {i}")
        applied, info = bounded_readout(agent, state, expert)
        if (row["applied_sha256"] != factors_hash(applied) or
                row["read_info"] != info or
                info["applied_effective_norm"] > .25 * info["reference_effective_norm"] + 1e-6):
            raise ValueError(f"Readout changed at {i}")
        if not old["lora"]["reward"] and state is not None:
            old_failed, _, _, _ = episode_factor(agent, tokenizer, old["lora"],
                argparse.Namespace(device=args.device, context_tokens=2048))
            old_toward, _ = bounded_residual(agent, expert, old_failed, max_relative=.25)
            alignment = effective_cosine(agent, applied, [-x for x in old_toward])
            if alignment is not None:
                old_local_alignments.append(alignment)
        ep = row["episode"]
        if ep["status"] != "complete" or ep["steps"] > 50 or ep["initial_observation"] != initial:
            raise ValueError(f"Invalid episode at {i}")
        replay(args.data_root / row["game"], ep)
        wins += ep["reward"]
        old_wins += old["lora"]["reward"]
        if ep["reward"]:
            if row["write_kind"] != "none":
                raise ValueError(f"Wrote success at {i}")
        else:
            failed, _, original, retained = episode_factor(agent, tokenizer, ep,
                argparse.Namespace(device=args.device, context_tokens=2048))
            toward, delta_info = bounded_residual(agent, expert, failed, max_relative=.25)
            delta = [-x for x in toward]
            if (row["write_kind"] != "own_failure" or
                    row["failed_factor_sha256"] != factors_hash(failed) or
                    row["delta_sha256"] != factors_hash(delta) or
                    row["delta_info"] != delta_info or
                    row["failed_source_tokens_original"] != original or
                    row["failed_source_tokens_retained"] != retained):
                raise ValueError(f"Failure write changed at {i}")
            state = update_state(state, delta, count, "failure_mean")
            count += 1
        if (row["state_sha256_after"] != factors_hash(state) or
                row["failure_updates_after"] != count):
            raise ValueError(f"Broken state after {i}")
    summary = {"n": 134, "old_first_wins": old_wins,
        "failure_mean_wins": wins, "failure_updates": count, "failures": 0}
    if report["summary"] != summary:
        raise ValueError("Score summary changed")
    return {"audit": "passed", "summary": summary, "replayed_episodes": 134,
        "diagnostic_only_old_failed_local_delta_alignment": {
            "n": len(old_local_alignments),
            "mean": statistics.mean(old_local_alignments) if old_local_alignments else None,
            "median": statistics.median(old_local_alignments) if old_local_alignments else None,
            "negative": sum(x < 0 for x in old_local_alignments),
            "below_0_1": sum(x < .1 for x in old_local_alignments)},
        "report_sha256": file_hash(args.output), "review_sha256": file_hash(args.review)}


if __name__ == "__main__":
    p = parser()
    p.add_argument("--audit-output", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_persistent_failure_delta_v39_134_audit_20261010.json"))
    args = p.parse_args()
    if args.command != "evaluate" or args.audit_output.exists():
        raise ValueError("Audit requires evaluate and fresh output")
    result = audit(args)
    save(args.audit_output, result)
    print(json.dumps(result), flush=True)
