"""One-shot ALFWorld frozen evaluation-split comparison of LoRA variants.

This uses the official frozen game split and environment win signal. The actor
is greedy and action-constrained, so its numbers are an internal comparison,
not a reproduction of the historical stochastic Delta-Mem protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.prepare_alf_eval_pairs import (
    sha256, validate_eval_review,
)
from ttcl.trajectory_hyperlora.prepare_alf_next_task import (
    partition, validate_review,
)
from ttcl.trajectory_hyperlora.relational_router_pilot import tokenize_records


def mean_adapter(agent, tokenizer, rows: list[dict], device: str) -> tuple[list[torch.Tensor], str, int]:
    sources = {}
    for row in rows:
        if partition(row["source_game"]) != "train":
            raise ValueError("Mean adapter source outside training partition")
        previous = sources.get(row["source_game"])
        if previous and (previous["source_episode_sha256"] != row["source_episode_sha256"]
                         or previous["source_records"] != row["source_records"]):
            raise ValueError("Conflicting reviewed source content")
        sources[row["source_game"]] = row
    factors = []
    for game in sorted(sources):
        fields = tokenize_records(tokenizer, sources[game]["source_records"],
                                  device)
        with torch.no_grad():
            agent.set_source(fields)
            factors.append([layer.b.detach().cpu().clone()
                            for layer in agent.adapters])
            agent.set_source(None)
    means = [torch.stack([item[layer] for item in factors]).mean(0)
             for layer in range(len(agent.adapters))]
    digest = hashlib.sha256()
    for factor in means:
        digest.update(factor.contiguous().numpy().tobytes())
    return means, digest.hexdigest(), len(sources)


def summarize(games: list[dict]) -> dict:
    complete = [row for row in games if row["base"]["status"] ==
                row["mean_adapter"]["status"] == "complete"]
    source_complete = [row for row in complete if isinstance(row["source_adapter"], dict)
                       and row["source_adapter"]["status"] == "complete"]
    return {"completed_base_mean_pairs": len(complete),
            "base_successes": sum(row["base"]["reward"] for row in complete),
            "mean_adapter_successes": sum(row["mean_adapter"]["reward"]
                                          for row in complete),
            "completed_source_triples": len(source_complete),
            "source_subset_base_successes": sum(row["base"]["reward"]
                                                for row in source_complete),
            "source_subset_mean_successes": sum(row["mean_adapter"]["reward"]
                                                for row in source_complete),
            "source_adapter_successes": sum(row["source_adapter"]["reward"]
                                            for row in source_complete),
            "infrastructure_failures": sum(
                row[arm]["status"] != "complete"
                for row in games for arm in ("base", "mean_adapter")
            ) + sum(isinstance(row["source_adapter"], dict) and
                    row["source_adapter"]["status"] != "complete"
                    for row in games)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path(
        "ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json"))
    parser.add_argument("--eval-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_eval_pairs_20261005_candidates.json"))
    parser.add_argument("--eval-review", type=Path, default=Path(
        "data/annotations/trajectory_hyperlora_alf_eval_pairs_20261005_reviewed.json"))
    parser.add_argument("--historical-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_candidates_20261005.json"))
    parser.add_argument("--historical-review", type=Path, default=Path(
        "data/annotations/trajectory_hyperlora_alf_20261005_reviewed_v2.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; do not overwrite frozen evaluation")
    plan = json.loads(args.plan.read_text())
    if plan["max_steps"] != 30 or plan["actor_max_tokens"] != 64:
        raise ValueError("Frozen evaluation action budgets changed")
    candidates = json.loads(args.eval_candidates.read_text())
    review = json.loads(args.eval_review.read_text())
    targets = validate_eval_review(candidates, review, args.plan, args.data_root)
    historical = validate_review(
        json.loads(args.historical_candidates.read_text()),
        json.loads(args.historical_review.read_text()))
    if candidates["historical_review_sha256"] != sha256(args.historical_review):
        raise ValueError("Evaluation source-review lineage changed")
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    means, mean_sha, mean_sources = mean_adapter(
        agent, tokenizer, historical, args.device)
    report = {"protocol": "Frozen ALFWorld evaluation games and environment win score; greedy constrained Qwen actor; official plan 30-step/64-token budgets; source/mean adapters fixed before evaluation; not historical Delta-Mem stochastic protocol",
              "plan_sha256": sha256(args.plan),
              "checkpoint_sha256": sha256(args.checkpoint),
              "eval_manifest_sha256": sha256(args.eval_candidates),
              "eval_review_sha256": sha256(args.eval_review),
              "train_source_review_sha256": sha256(args.historical_review),
              "mean_adapter_sha256": mean_sha,
              "mean_source_count": mean_sources,
              "max_steps": plan["max_steps"],
              "max_new_tokens": plan["actor_max_tokens"],
              "games": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for row in targets:
        game_path = args.data_root / row["target_game"]
        entry = {"game": row["target_game"],
                 "game_sha256": row["target_game_sha256"],
                 "family": row["family"],
                 "source_game": row["source_game"],
                 "source_episode_sha256": row["source_episode_sha256"],
                 "input_content_sha256": row["input_content_sha256"]}
        entry["base"] = run_episode(
            agent, tokenizer, game_path, {}, adapter=False,
            device=args.device, max_steps=plan["max_steps"],
            max_new_tokens=plan["actor_max_tokens"], constrain_actions=True)
        entry["mean_adapter"] = run_episode(
            agent, tokenizer, game_path, {}, adapter=False,
            fixed_adapter=means, device=args.device,
            max_steps=plan["max_steps"],
            max_new_tokens=plan["actor_max_tokens"], constrain_actions=True)
        if row["source_game"] is None:
            entry["source_adapter"] = "unsupported_no_reviewed_train_source"
        else:
            fields = tokenize_records(tokenizer, row["source_records"], args.device)
            entry["source_adapter"] = run_episode(
                agent, tokenizer, game_path, fields, adapter=True,
                device=args.device, max_steps=plan["max_steps"],
                max_new_tokens=plan["actor_max_tokens"], constrain_actions=True)
        report["games"].append(entry)
        report["summary"] = summarize(report["games"])
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"game": row["target_game"],
                          "base": entry["base"].get("reward"),
                          "mean": entry["mean_adapter"].get("reward"),
                          "source": entry["source_adapter"].get("reward")
                              if isinstance(entry["source_adapter"], dict)
                              else entry["source_adapter"],
                          "summary": report["summary"]}), flush=True)


if __name__ == "__main__":
    main()
