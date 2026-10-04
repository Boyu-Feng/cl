"""Ablate source specificity with the average LoRA from reviewed train sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.prepare_alf_next_task import (
    partition, validate_review,
)
from ttcl.trajectory_hyperlora.relational_router_pilot import tokenize_records


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_candidates_20261005.json"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/trajectory_hyperlora_alf_20261005_reviewed_v2.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; do not overwrite a frozen mean-adapter probe")
    reference = json.loads(args.reference.read_text())
    if (reference["checkpoint_sha256"] != sha256(args.checkpoint) or
            reference.get("constrain_actions") is not True or
            reference.get("adapter_scale") != 1.0 or
            reference.get("max_steps") != 50):
        raise ValueError("Reference actor/protocol mismatch")
    approved = validate_review(json.loads(args.candidates.read_text()),
                               json.loads(args.review.read_text()))
    sources = {}
    for row in approved:
        if partition(row["source_game"]) != "train":
            raise ValueError("Source outside reviewed training partition")
        prior = sources.get(row["source_game"])
        if prior and (prior["source_episode_sha256"] != row["source_episode_sha256"]
                      or prior["source_records"] != row["source_records"]):
            raise ValueError("Conflicting reviewed source versions")
        sources[row["source_game"]] = row
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    factors = []
    for game in sorted(sources):
        fields = tokenize_records(tokenizer, sources[game]["source_records"],
                                  args.device)
        with torch.no_grad():
            agent.set_source(fields)
            factors.append([layer.b.detach().cpu().clone()
                            for layer in agent.adapters])
            agent.set_source(None)
    means = [torch.stack([item[layer] for item in factors]).mean(0)
             for layer in range(len(agent.adapters))]
    adapter_digest = hashlib.sha256()
    for factor in means:
        adapter_digest.update(factor.contiguous().numpy().tobytes())
    report = {"protocol": "Global average of LoRA factors generated from reviewed train trajectories; no specific source at target time; same actor and 50-step constrained action budget as reference",
              "checkpoint_sha256": sha256(args.checkpoint),
              "reference_sha256": sha256(args.reference),
              "review_sha256": sha256(args.review),
              "source_count": len(sources),
              "mean_adapter_sha256": adapter_digest.hexdigest(),
              "max_steps": reference["max_steps"],
              "max_new_tokens": reference["max_new_tokens"],
              "games": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for row in reference["games"]:
        game_path = args.data_root / row["game"]
        if sha256(game_path) != row["game_sha256"]:
            raise ValueError("Target game hash changed")
        if row["base"]["status"] != "complete" or \
                row["generated"]["status"] != "complete":
            raise ValueError("Reference pair is incomplete")
        result = run_episode(agent, tokenizer, game_path, {},
                             adapter=False, fixed_adapter=means,
                             device=args.device,
                             max_steps=reference["max_steps"],
                             max_new_tokens=reference["max_new_tokens"],
                             constrain_actions=True)
        report["games"].append({"game": row["game"],
                                "game_sha256": row["game_sha256"],
                                "base_reward": row["base"]["reward"],
                                "source_lora_reward": row["generated"]["reward"],
                                "mean_adapter": result})
        complete = [item for item in report["games"]
                    if item["mean_adapter"]["status"] == "complete"]
        report["summary"] = {
            "completed": len(complete),
            "base_successes": sum(item["base_reward"] for item in complete),
            "source_lora_successes": sum(item["source_lora_reward"]
                                         for item in complete),
            "mean_adapter_successes": sum(item["mean_adapter"]["reward"]
                                          for item in complete)}
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"game": row["game"],
                          "mean_reward": result.get("reward"),
                          "summary": report["summary"]}), flush=True)


if __name__ == "__main__":
    main()
