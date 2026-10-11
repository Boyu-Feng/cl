"""Paired one-step test: can a failed own episode improve the next task?

Reconstruct the audited v27 success-only B chain. At each failure, derive a
bounded delta from that failed trajectory and run the *next* official train
task with B + delta. The unchanged-B outcome comes from the same audited v27
chain. This is a local counterfactual, not a closed-loop policy score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import (
    factors_hash, mean_factors,
)
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_success_gated_factors_v27 import episode_factor
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contrastive_failure_factors_v30 import factor_norm


def delta_candidates(old, failed, fraction=.25):
    """Output two signed, norm-bounded Delta-B candidates from a failure."""
    if not 0 < fraction <= 1 or not failed or (
            old is not None and len(old) != len(failed)):
        raise ValueError("Invalid factor update")
    center = ([torch.zeros_like(x) for x in failed] if old is None else old)
    direction = [f.detach().cpu().float() - b.detach().cpu().float()
                 for f, b in zip(failed, center, strict=True)]
    reference = factor_norm(center) if old is not None else factor_norm(failed)
    norm = factor_norm(direction)
    scale = min(fraction, fraction * reference / norm) if norm else 0.
    return {
        "toward_failure": [b + scale * d for b, d in zip(center, direction, strict=True)],
        "away_from_failure": [b - scale * d for b, d in zip(center, direction, strict=True)],
    }, {"direction_norm": norm, "reference_norm": reference,
        "delta_norm": scale * norm, "scale": scale}


def load_inputs(args):
    source = json.loads(args.source_report.read_text())
    audit = json.loads(args.source_audit.read_text())
    if (source["sequence"] != args.sequence or len(source["games"]) != 18 or
            source["failures"] or source["summary"]["n"] != 18 or
            source["actor_sha256"] != file_hash(Path(__file__).with_name("alfworld_zero_shot_probe.py")) or
            source["checkpoint_sha256"] != file_hash(args.checkpoint) or
            source["model_config_sha256"] != file_hash(args.model / "config.json") or
            source["review_sha256"] != file_hash(args.source_review) or
            audit.get("audit") != "passed" or audit.get("n") != 18 or
            audit.get("report_sha256") != file_hash(args.source_report) or
            audit.get("review_sha256") != file_hash(args.source_review)):
        raise ValueError("Changed or incomplete v27 source lineage")
    return source


def bindings(args, source):
    fixed = {"method": "v34 local next-task failure Delta-B paired pilot",
        "sequence": args.sequence, "source_report_sha256": file_hash(args.source_report),
        "source_audit_sha256": file_hash(args.source_audit),
        "source_review_sha256": file_hash(args.source_review),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "runner_sha256": file_hash(Path(__file__)),
        "actor_sha256": source["actor_sha256"],
        "factor_helper_sha256": file_hash(Path(__file__).with_name("alfworld_success_gated_factors_v27.py")),
        "fraction": args.fraction, "max_steps": 50,
        "max_new_tokens": 64, "actor_history_turns": 2,
        "loop_guard_max": 2, "context_tokens": args.context_tokens}
    targets = []
    for i, row in enumerate(source["games"][:-1]):
        if bool(row["online"]["reward"]):
            continue
        nxt = source["games"][i + 1]
        game = nxt["game"]
        if "/train/" not in "/" + game:
            raise ValueError("Nontraining target")
        content = {"source_index": i, "target_index": i + 1,
            "source_game": row["game"], "target_game": game,
            "source_game_sha256": file_hash(args.data_root / row["game"]),
            "target_game_sha256": file_hash(args.data_root / game),
            "source_episode_sha256": digest(row["online"]),
            "baseline_episode_sha256": digest(nxt["online"]),
            "baseline_factor_sha256": nxt["factor_sha256_before"],
            "runner_sha256": fixed["runner_sha256"],
            "checkpoint_sha256": fixed["checkpoint_sha256"]}
        targets.append({**content, "input_content_sha256": digest(content),
            "reviewed_target": True})
    return fixed, targets


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    source = load_inputs(args)
    fixed, targets = bindings(args, source)
    save(args.review, {"protocol": "Fresh content-bound next-task targets from audited v27 own failures; train split only; paired local counterfactual; choose sign on train_seq0_6 only",
        **fixed, "targets": targets, "targets_sha256": digest(targets)})
    print(json.dumps({"reviewed_targets": len(targets)}))


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    source = load_inputs(args)
    fixed, targets = bindings(args, source)
    review = json.loads(args.review.read_text())
    if (any(review.get(k) != v for k, v in fixed.items()) or
            review.get("targets") != targets or
            review.get("targets_sha256") != digest(targets)):
        raise ValueError("Changed v34 target bindings")
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    state, accepted = None, 0
    target_by_source = {x["source_index"]: x for x in targets}
    report = {"protocol": review["protocol"], "review_sha256": file_hash(args.review),
        **fixed, "rows": [], "failures": []}
    for i, item in enumerate(source["games"]):
        if factors_hash(state) != item["factor_sha256_before"]:
            raise ValueError(f"Source B chain mismatch before {i}")
        episode = item["online"]
        if bool(episode["reward"]):
            increment, _, _, _ = episode_factor(agent, tokenizer, episode, args)
            state = mean_factors(state, increment, accepted)
            accepted += 1
        elif i in target_by_source:
            binding = target_by_source[i]
            failed, _, original, retained = episode_factor(
                agent, tokenizer, episode, args)
            candidates, info = delta_candidates(state, failed, args.fraction)
            nxt = source["games"][i + 1]
            row = {"source_index": i, "target_index": i + 1,
                "input_content_sha256": binding["input_content_sha256"],
                "factor_sha256_before": factors_hash(state),
                "source_tokens_original": original,
                "source_tokens_retained": retained,
                "failed_factor_sha256": factors_hash(failed),
                "delta": info, "baseline_reward": nxt["online"]["reward"],
                "baseline_episode_sha256": digest(nxt["online"]), "arms": {}}
            for name, factors in candidates.items():
                outcome = run_episode(agent, tokenizer,
                    args.data_root / binding["target_game"], {}, adapter=False,
                    fixed_adapter=factors, device=args.device, max_steps=50,
                    max_new_tokens=64, constrain_actions=True,
                    actor_history_turns=2, loop_guard_max=2)
                row["arms"][name] = {"factor_sha256": factors_hash(factors),
                                     "episode": outcome}
                if outcome["status"] != "complete" or (
                        outcome["initial_observation"] !=
                        nxt["online"]["initial_observation"]):
                    report["failures"].append({"source_index": i,
                        "arm": name, "status": outcome["status"]})
            report["rows"].append(row)
            save(args.output, report)
            print(json.dumps({"source_index": i, "rows": len(report["rows"]),
                "failures": len(report["failures"])}), flush=True)
            if report["failures"]:
                raise RuntimeError("Incomplete official counterfactual")
        if factors_hash(state) != item["factor_sha256_after"]:
            raise ValueError(f"Source B chain mismatch after {i}")
    report["summary"] = {"pairs": len(report["rows"]),
        "baseline": sum(x["baseline_reward"] for x in report["rows"]),
        **{arm: sum(x["arms"][arm]["episode"]["reward"] for x in report["rows"])
           for arm in ("toward_failure", "away_from_failure")},
        "failures": len(report["failures"])}
    save(args.output, report)
    print(json.dumps(report["summary"]), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--sequence", required=True,
                        choices=("train_seq0_6", "train_seq6_6"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("ttcl/data/alfworld_delta"))
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--source-review", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=.25)
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--gpu-fraction", type=float, default=.65)
    args = parser.parse_args()
    if args.fraction <= 0 or args.fraction > 1 or args.context_tokens < 2:
        parser.error("Invalid failure-delta protocol")
    (prepare if args.command == "prepare" else evaluate)(args)


if __name__ == "__main__":
    main()
