"""ALFWorld adapter for generic reward-trained trajectory-to-LoRA latent codes.

Uses only reviewed training source/target pairs. The Qwen actor, old trajectory
encoder and LoRA factor heads are frozen; a new source-conditioned correction
head is trained from real paired terminal reward. No checkpoint is overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
from torch import nn

from ttcl.trajectory_hyperlora.alfworld_rl_collect import load_train_targets, sha256
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.relational_router_pilot import tokenize_records
from ttcl.trajectory_hyperlora.train_alfworld_reward_policy import rl_partition
from ttcl.trajectory_hyperlora.trajectory_lora_rl import latent_adapter_reinforce


class LatentCorrection(nn.Module):
    def __init__(self, width: int, code_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 32),
                                 nn.Tanh(), nn.Linear(32, code_dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, source_latent: torch.Tensor) -> torch.Tensor:
        return self.net(source_latent.float())


def pca_basis(latents: dict[str, torch.Tensor], rank: int) -> torch.Tensor:
    matrix = torch.stack([latents[key].squeeze(0).float()
                          for key in sorted(latents)])
    centered = matrix - matrix.mean(0)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    if rank > vh.shape[0]:
        raise ValueError("Not enough distinct source latents for latent code")
    return vh[:rank].T.contiguous()


def factors(agent, latent: torch.Tensor) -> list[torch.Tensor]:
    with torch.no_grad():
        return [head(latent).reshape(1, adapter.base.out_features,
                                          adapter.rank).detach()
                for head, adapter in zip(agent.b_heads, agent.adapters, strict=True)]


def run(args: argparse.Namespace) -> dict:
    if args.output.exists() or args.save_checkpoint.exists():
        raise FileExistsError("Use fresh result and checkpoint paths")
    if args.train_limit < 1 or args.epochs < 1 or args.code_dim < 1 or args.sigma <= 0:
        raise ValueError("Invalid declared training budget")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    targets, historical = load_train_targets(args)
    rewards = json.loads(args.rewards.read_text())
    by_target = {row["target_game"]: row for row in targets}
    if (rewards["plan_sha256"] != sha256(args.plan) or
            rewards["checkpoint_sha256"] != sha256(args.checkpoint) or
            rewards["historical_review_sha256"] != sha256(args.historical_review) or
            rewards["reward_review_sha256"] != sha256(args.reward_review) or
            rewards["target_bindings"] != {game: row["input_content_sha256"]
                                            for game, row in by_target.items()} or
            rewards["max_steps"] != 30 or rewards["max_new_tokens"] != 64):
        raise ValueError("Frozen reward lineage or budget mismatch")
    prior = {row["game"]: row for row in rewards["games"]}
    if set(prior) != set(by_target):
        raise ValueError("Reward collection does not cover reviewed targets")
    train = [row for row in rewards["games"] if rl_partition(row["game"]) == "train"]
    dev = [row for row in rewards["games"] if rl_partition(row["game"]) == "dev"]
    # Balance baseline successes and failures before limiting the pilot; the
    # selection is fixed before any new reward is observed.
    success = sorted((row for row in train if row["base"]["reward"] == 1),
                     key=lambda row: row["game"])
    failure = sorted((row for row in train if row["base"]["reward"] == 0),
                     key=lambda row: row["game"])
    rng.shuffle(success)
    rng.shuffle(failure)
    chosen = []
    for index in range(max(len(success), len(failure))):
        if index < len(failure): chosen.append(failure[index])
        if index < len(success): chosen.append(success[index])
    chosen = chosen[:args.train_limit]
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    source_by_game = {}
    for row in historical:
        if row["split"] != "train":
            continue
        game = row["source_game"]
        if game in source_by_game and source_by_game[game] != row["source_records"]:
            raise ValueError("Conflicting reviewed source trajectory")
        source_by_game[game] = row["source_records"]
    latent_by_game = {}
    with torch.no_grad():
        for game, records in sorted(source_by_game.items()):
            latent_by_game[game] = agent.encode(tokenize_records(
                tokenizer, records, args.device)).detach().float()
    basis = pca_basis(latent_by_game, args.code_dim).to(args.device)
    correction = LatentCorrection(next(iter(latent_by_game.values())).shape[-1],
                                  args.code_dim).to(args.device)
    optimizer = torch.optim.AdamW(correction.parameters(), lr=args.lr,
                                  weight_decay=0)
    report = {"protocol": "Exploratory ALFWorld train-only stochastic latent correction to frozen trajectory hyper-LoRA; real terminal-reward REINFORCE; fixed actor/encoder/factor heads; paired saved base reward; 30 steps/64 tokens",
              "seed": args.seed, "sigma": args.sigma, "lr": args.lr,
              "epochs": args.epochs, "train_limit": args.train_limit,
              "code_dim": args.code_dim,
              "plan_sha256": sha256(args.plan),
              "checkpoint_sha256": sha256(args.checkpoint),
              "historical_review_sha256": sha256(args.historical_review),
              "reward_review_sha256": sha256(args.reward_review),
              "rewards_sha256": sha256(args.rewards),
              "train_games": [row["game"] for row in chosen],
              "dev_games": [row["game"] for row in dev],
              "train": [], "dev": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save() -> None:
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(args.output)
    def evaluate(rows, stage):
        for row in rows:
            target = by_target[row["game"]]
            source = latent_by_game[target["source_game"]]
            with torch.no_grad():
                code = correction(source)
                adapted = source + code @ basis.T
                compiled = factors(agent, adapted)
            outcome = run_episode(agent, tokenizer, args.data_root / row["game"],
                {}, adapter=False, fixed_adapter=compiled, device=args.device,
                max_steps=30, max_new_tokens=64, constrain_actions=True)
            record = {"game": row["game"],
                "input_content_sha256": target["input_content_sha256"],
                "game_sha256": target["target_game_sha256"],
                "source_episode_sha256": target["source_episode_sha256"],
                "old_base_reward": row["base"]["reward"],
                "old_source_reward": row["source"]["reward"],
                "new": outcome}
            report[stage].append(record)
            save()
            print(json.dumps({"stage": stage, "game": row["game"],
                              "old_source": row["source"]["reward"],
                              "new": outcome.get("reward"),
                              "status": outcome["status"]}), flush=True)
            if outcome["status"] != "complete" or \
                    outcome.get("initial_observation") != row["base"]["initial_observation"]:
                report["failures"].append({"stage": stage, "game": row["game"],
                                           "status": outcome["status"]})
                save()
                raise RuntimeError("ALFWorld failure or reset mismatch")
    try:
        for epoch in range(args.epochs):
            for row in chosen:
                target = by_target[row["game"]]
                source = latent_by_game[target["source_game"]]
                mean = correction(source)
                sampled = mean.detach() + args.sigma * torch.randn_like(mean)
                with torch.no_grad():
                    compiled = factors(agent, source + sampled @ basis.T)
                outcome = run_episode(agent, tokenizer,
                    args.data_root / row["game"], {}, adapter=False,
                    fixed_adapter=compiled, device=args.device,
                    max_steps=30, max_new_tokens=64, constrain_actions=True)
                if outcome["status"] != "complete" or \
                        outcome.get("initial_observation") != row["base"]["initial_observation"]:
                    report["failures"].append({"stage": "train", "epoch": epoch,
                        "game": row["game"], "outcome": outcome})
                    save()
                    raise RuntimeError("ALFWorld failure or reset mismatch")
                loss = latent_adapter_reinforce(mean, sampled, args.sigma,
                    outcome["reward"], row["base"]["reward"])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(correction.parameters(), 1.0)
                optimizer.step()
                record = {"epoch": epoch, "game": row["game"],
                    "input_content_sha256": target["input_content_sha256"],
                    "source_episode_sha256": target["source_episode_sha256"],
                    "base_reward": row["base"]["reward"],
                    "old_source_reward": row["source"]["reward"],
                    "sampled_reward": outcome["reward"],
                    "sampled_code": sampled.detach().cpu().flatten().tolist(),
                    "budget_used": outcome["steps"],
                    "loss": float(loss.detach())}
                report["train"].append(record)
                save()
                print(json.dumps({"stage": "train", "epoch": epoch,
                    "game": row["game"], "base": row["base"]["reward"],
                    "old_source": row["source"]["reward"],
                    "sampled": outcome["reward"]}), flush=True)
        evaluate(dev, "dev")
    finally:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"correction": correction.state_dict(),
                    "basis": basis.detach().cpu(),
                    "config": {key: report[key] for key in
                        ("protocol", "seed", "sigma", "code_dim", "lr",
                         "checkpoint_sha256", "historical_review_sha256",
                         "rewards_sha256")}}, args.save_checkpoint)
        save()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_warmstart_20261005/seed42_v2.pt"))
    parser.add_argument("--plan", type=Path, default=Path(
        "ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json"))
    parser.add_argument("--historical-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_candidates_20261005.json"))
    parser.add_argument("--historical-review", type=Path, default=Path(
        "data/annotations/trajectory_hyperlora_alf_20261005_reviewed_v2.json"))
    parser.add_argument("--reward-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_reward_pairs_20261005_candidates.json"))
    parser.add_argument("--reward-review", type=Path, default=Path(
        "data/annotations/trajectory_hyperlora_alf_reward_pairs_20261005_reviewed.json"))
    parser.add_argument("--rewards", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_rl_20261005/train_rewards_complete.json"))
    parser.add_argument("--data-root", type=Path, default=Path("ttcl/data/alfworld_delta"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.65)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--code-dim", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=.5)
    parser.add_argument("--lr", type=float, default=.01)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-checkpoint", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
