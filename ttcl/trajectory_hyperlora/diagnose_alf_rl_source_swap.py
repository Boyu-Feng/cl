"""Read-only same-family source swap for a frozen ALFWorld reward policy.

The alternative trajectories were already reviewed as successful train-only
sources. This changes only the policy's source latent; it does not roll out the
actor or claim any alternative-source environment reward.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.prepare_alf_next_task import (
    digest_json, partition, validate_review,
)
from ttcl.trajectory_hyperlora.prepare_alf_rl_holdout import validate
from ttcl.trajectory_hyperlora.train_alfworld_reward_policy import (
    ARMS, AdapterPolicy, features, source_latents,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise ValueError("Fresh diagnostic output required")
    targets = validate(json.loads(args.candidates.read_text()),
                       json.loads(args.review.read_text()), args.plan, args.data_root)
    historical = validate_review(
        json.loads(args.historical_candidates.read_text()),
        json.loads(args.historical_review.read_text()))
    learned = torch.load(args.policy_checkpoint, map_location="cpu", weights_only=True)
    if (learned["plan_sha256"] != sha256(args.plan) or
            learned["checkpoint_sha256"] != sha256(args.checkpoint) or
            learned["historical_review_sha256"] != sha256(args.historical_review)):
        raise ValueError("Frozen policy lineage mismatch")
    sources = defaultdict(dict)
    for row in historical:
        if partition(row["source_game"]) != "train":
            raise ValueError("Alternative source is not reviewed train data")
        sources[row["family"]][row["source_game"]] = row
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    latents = source_latents(agent, tokenizer, historical, args.device)
    if sorted(latents) != learned["source_games"]:
        raise ValueError("Source universe changed")
    policy = AdapterPolicy(learned["source_basis"].shape[1] + 6)
    policy.load_state_dict(learned["policy_state"])
    policy.eval()
    rows = []
    with torch.no_grad():
        for target in targets:
            correct = target["source_game"]
            if correct is None:
                continue
            choices = {game: source for game, source in
                       sources[target["family"]].items() if game != correct}
            if not choices:
                continue
            wrong = min(choices, key=lambda game: digest_json(
                ["rl_same_family_source_swap_v1", target["target_game"], game]))
            candidate = choices[wrong]
            result = {"target_game": target["target_game"],
                      "target_game_sha256": target["target_game_sha256"],
                      "family": target["family"],
                      "correct_source_game": correct,
                      "correct_source_sha256": target["source_episode_sha256"],
                      "alternative_source_game": wrong,
                      "alternative_source_sha256": candidate["source_episode_sha256"],
                      "alternative_input_content_sha256": digest_json({
                          "target_game": target["target_game"],
                          "target_game_sha256": target["target_game_sha256"],
                          "initial_observation": target["initial_observation"],
                          "source_game": wrong,
                          "source_episode_sha256": candidate["source_episode_sha256"],
                          "source_records": candidate["source_records"]})}
            for name, source_game in (("correct", correct), ("alternative", wrong)):
                x = features(target["family"], latents[source_game],
                             learned["source_mean"], learned["source_basis"],
                             learned["source_scale"])
                probabilities = policy(x).softmax(-1)
                result[name] = {
                    "chosen_arm": ARMS[int(probabilities.argmax())],
                    "probabilities": {arm: float(probabilities[index])
                                      for index, arm in enumerate(ARMS)}}
            result["choice_changed"] = (result["correct"]["chosen_arm"] !=
                                        result["alternative"]["chosen_arm"])
            result["total_variation"] = .5 * sum(abs(
                result["correct"]["probabilities"][arm] -
                result["alternative"]["probabilities"][arm]) for arm in ARMS)
            rows.append(result)
    output = {"protocol": "Read-only same-family approved source latent swap; frozen policy and actor; no environment rollout for alternative source; no reward or label used",
              "policy_checkpoint_sha256": sha256(args.policy_checkpoint),
              "holdout_review_sha256": sha256(args.review),
              "historical_review_sha256": sha256(args.historical_review),
              "n": len(rows),
              "choice_changed": sum(row["choice_changed"] for row in rows),
              "mean_total_variation": sum(row["total_variation"] for row in rows)
                  / max(len(rows), 1),
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"n": output["n"], "choice_changed": output["choice_changed"],
                      "mean_total_variation": output["mean_total_variation"]}))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_warmstart_20261005/seed42_v2.pt"))
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path(
        "ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json"))
    parser.add_argument("--historical-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_candidates_20261005.json"))
    parser.add_argument("--historical-review", type=Path, default=Path(
        "data/annotations/trajectory_hyperlora_alf_20261005_reviewed_v2.json"))
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_rl_20261005/valid_seen_candidates.json"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/trajectory_hyperlora_alf_rl_valid_seen_20261005_reviewed.json"))
    parser.add_argument("--data-root", type=Path, default=Path("ttcl/data/alfworld_delta"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.40)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
