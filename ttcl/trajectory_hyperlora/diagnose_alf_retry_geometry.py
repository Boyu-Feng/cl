"""Read-only geometry of trajectory-generated ALFWorld LoRA updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch

from ttcl.trajectory_hyperlora.alfworld_expert_first_attempts import checked_all_sources
from ttcl.trajectory_hyperlora.alfworld_same_game_retry import file_hash
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.diagnose_source_adapters import effective_gram
from ttcl.trajectory_hyperlora.relational_router_pilot import tokenize_records


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = checked_all_sources(args)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    adapters, latents = [], []
    for row in rows:
        fields = tokenize_records(tokenizer, row["source_records"],
            args.device, max_tokens=args.source_max_tokens,
            truncation_mode=args.source_truncation,
            repeat_initial_observation=args.repeat_initial_observation)
        with torch.no_grad():
            latents.append(agent.encode(fields).detach().cpu().squeeze(0).double())
            agent.set_source(fields)
            adapters.append([layer.b.detach().cpu().squeeze(0).clone()
                             for layer in agent.adapters])
            agent.set_source(None)
    factors = [layer.a.detach().cpu() for layer in agent.adapters]
    gram = effective_gram(adapters, factors)
    norms = gram.diag().clamp_min(0).sqrt()
    cosines = gram / (norms[:, None] * norms[None, :]).clamp_min(1e-12)
    latent = torch.stack(latents)
    centered = latent - latent.mean(0, keepdim=True)
    same, different = [], []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            value = float(cosines[left, right])
            (same if rows[left]["family"] == rows[right]["family"]
             else different).append(value)
    result = {"protocol": "Read-only effective B@A geometry of 42 content-reviewed ALFWorld train first attempts; no test input or reward used",
              "checkpoint_sha256": file_hash(args.checkpoint),
              "source_review_sha256": file_hash(args.source_review),
              "source_max_tokens": args.source_max_tokens,
              "source_truncation": args.source_truncation,
              "repeat_initial_observation": args.repeat_initial_observation,
              "context_mode": agent.context_mode,
              "source_count": len(rows),
              "mean_effective_norm": float(norms.mean()),
              "norm_coefficient_of_variation": float(norms.std() /
                  norms.mean().clamp_min(1e-12)),
              "same_family_mean_cosine": statistics.mean(same),
              "cross_family_mean_cosine": statistics.mean(different),
              "min_pair_cosine": min(same + different),
              "latent_centered_rms_over_mean_norm": float(
                  centered.square().sum(1).mean().sqrt() /
                  latent.mean(0).norm().clamp_min(1e-12))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False,
                                      indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path(
        "ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json"))
    parser.add_argument("--first-attempts", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_rl_20261005/train_rewards_complete.json"))
    parser.add_argument("--retry-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_same_game_retry_candidates_20261006.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_expert_first_attempts_candidates_20261006.json"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alf_expert_first_attempts_reviewed_20261006.json"))
    parser.add_argument("--source-max-tokens", type=int, default=40)
    parser.add_argument("--source-truncation", choices=("head", "head_tail"),
                        default="head_tail")
    parser.add_argument("--repeat-initial-observation", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.source_max_tokens < 2 or not 0 < args.gpu_fraction <= 1 or
            (args.repeat_initial_observation and
             args.source_truncation != "head_tail")):
        parser.error("Invalid source geometry budget")
    run(args)


if __name__ == "__main__":
    main()
