"""Optimize a trajectory-conditioned LoRA choice from real ALFWorld reward.

This is a full-information contextual bandit: each train game is rolled out
with the same frozen actor under base, mean, and source LoRA, then the policy
gradient of expected terminal reward is computed exactly over those actions.
No action imitation labels or development/test rewards enter the optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ttcl.trajectory_hyperlora.paired_reward_gate import visible_family
from ttcl.trajectory_hyperlora.alfworld_rl_collect import load_train_targets, sha256
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.relational_router_pilot import tokenize_records


FAMILIES = ("look_at_obj_in_light", "pick_and_place_simple",
            "pick_clean_then_place_in_recep", "pick_cool_then_place_in_recep",
            "pick_heat_then_place_in_recep", "pick_two_obj_and_place")
ARMS = ("base", "mean", "source")


def rl_partition(game: str) -> str:
    digest = hashlib.sha256(("alf_rl_dev_v1:" + game).encode()).digest()
    return "dev" if int.from_bytes(digest[:8], "big") % 5 == 0 else "train"


def source_latents(agent, tokenizer, rows: list[dict], device: str) -> dict[str, torch.Tensor]:
    sources = {}
    for row in rows:
        game = row["source_game"]
        if game in sources:
            previous = sources[game]
            if (previous["source_episode_sha256"] != row["source_episode_sha256"] or
                    previous["source_records"] != row["source_records"]):
                raise ValueError("Conflicting source trajectory contents")
        sources[game] = row
    result = {}
    with torch.no_grad():
        for game, row in sorted(sources.items()):
            fields = tokenize_records(tokenizer, row["source_records"], device)
            result[game] = agent.encode(fields).detach().float().squeeze(0).cpu()
    return result


def pca_source_features(latents: dict[str, torch.Tensor], rank: int = 4
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    matrix = torch.stack(list(latents.values())).float()
    mean = matrix.mean(0)
    centered = matrix - mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[:min(rank, vh.shape[0])].T.contiguous()
    projected = centered @ basis
    scale = projected.std(0).clamp_min(1e-5)
    return mean, basis, scale


def features(family: str, latent: torch.Tensor | None, mean: torch.Tensor,
             basis: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if family not in FAMILIES:
        raise ValueError(f"Unknown ALFWorld family: {family}")
    one_hot = F.one_hot(torch.tensor(FAMILIES.index(family)), len(FAMILIES)).float()
    projection = torch.zeros(basis.shape[1]) if latent is None else \
        ((latent.float().cpu() - mean) @ basis / scale).clamp(-3, 3)
    return torch.cat([one_hot, projection])


class AdapterPolicy(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.linear = nn.Linear(width, len(ARMS))
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, *, has_source: bool = True) -> torch.Tensor:
        logits = self.linear(x)
        if not has_source:
            logits = logits.clone()
            logits[..., ARMS.index("source")] = -1e9
        return logits


def score(rows: list[dict], policy: AdapterPolicy, x: torch.Tensor) -> dict:
    with torch.no_grad():
        logits = policy(x)
        chosen = logits.argmax(-1).tolist()
        probs = logits.softmax(-1)
    chosen_rewards = [row[ARMS[index]]["reward"]
                      for row, index in zip(rows, chosen, strict=True)]
    return {"n": len(rows),
            "successes": sum(chosen_rewards),
            "base_successes": sum(row["base"]["reward"] for row in rows),
            "mean_successes": sum(row["mean"]["reward"] for row in rows),
            "source_successes": sum(row["source"]["reward"] for row in rows),
            "expected_successes": float(sum(
                sum(float(probs[i, j]) * row[arm]["reward"]
                    for j, arm in enumerate(ARMS))
                for i, row in enumerate(rows))),
            "choices": {arm: chosen.count(index)
                        for index, arm in enumerate(ARMS)}}


def run(args: argparse.Namespace) -> dict:
    if args.output.exists() or args.save_checkpoint.exists() or args.steps < 1:
        raise ValueError("Fresh outputs and positive RL step budget required")
    targets, historical = load_train_targets(args)
    by_game = {row["target_game"]: row for row in targets}
    rewards = json.loads(args.rewards.read_text())
    if (rewards["plan_sha256"] != sha256(args.plan) or
            rewards["checkpoint_sha256"] != sha256(args.checkpoint) or
            rewards["historical_review_sha256"] != sha256(args.historical_review) or
            rewards["reward_review_sha256"] != sha256(args.reward_review) or
            rewards["max_steps"] != 30 or rewards["max_new_tokens"] != 64 or
            rewards["target_bindings"] != {game: row["input_content_sha256"]
                                           for game, row in by_game.items()}):
        raise ValueError("ALFWorld reward lineage or budget mismatch")
    rows = rewards["games"]
    if len(rows) != len(by_game) or len({row["game"] for row in rows}) != len(rows):
        raise ValueError("Reward collection incomplete or duplicated")
    for row in rows:
        target = by_game[row["game"]]
        if (row["input_content_sha256"] != target["input_content_sha256"] or
                row["source_episode_sha256"] != target["source_episode_sha256"] or
                row["game_sha256"] != target["target_game_sha256"]):
            raise ValueError("Reward target content binding mismatch")
        for arm in ARMS:
            if row[arm]["status"] != "complete" or row[arm]["reward"] not in (0, 1):
                raise ValueError("Incomplete or invalid environment reward")
            if row[arm]["initial_observation"] != row["base"]["initial_observation"]:
                raise ValueError("Paired ALFWorld resets have different observations")
        if visible_family(row["base"]["initial_observation"]) != row["family"]:
            raise ValueError("Actor-visible task family disagrees with metadata")
    torch.manual_seed(args.seed)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    latent_by_source = source_latents(agent, tokenizer, historical, args.device)
    mean, basis, scale = pca_source_features(latent_by_source)
    train_rows = [row for row in rows if rl_partition(row["game"]) == "train"]
    dev_rows = [row for row in rows if rl_partition(row["game"]) == "dev"]
    if not train_rows or not dev_rows:
        raise ValueError("RL train/dev partition empty")
    def matrix(subset: list[dict]) -> torch.Tensor:
        return torch.stack([features(visible_family(row["base"]["initial_observation"]),
                                     latent_by_source[row["source_game"]],
                                     mean, basis, scale) for row in subset])
    train_x, dev_x = matrix(train_rows), matrix(dev_rows)
    train_reward = torch.tensor([[row[arm]["reward"] for arm in ARMS]
                                 for row in train_rows], dtype=torch.float32)
    policy = AdapterPolicy(train_x.shape[1])
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr,
                                  weight_decay=0)
    history = []
    for step in range(args.steps):
        logits = policy(train_x)
        probabilities = logits.softmax(-1)
        advantage = train_reward - train_reward.mean(-1, keepdim=True)
        expected_advantage = (probabilities * advantage).sum(-1).mean()
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1).mean()
        l2 = policy.linear.weight.square().mean() + policy.linear.bias.square().mean()
        loss = -expected_advantage - args.entropy_weight * entropy + args.l2_weight * l2
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 20 == 0 or step == args.steps - 1:
            history.append({"step": step + 1, "train": score(train_rows, policy, train_x),
                            "dev": score(dev_rows, policy, dev_x)})
    result = {"protocol": "Train-only full-information contextual-bandit policy gradient over base/public/source LoRA; true ALFWorld terminal reward; actor, source encoder and LoRA factors frozen; source latent PCA plus goal family input; argmax evaluation",
              "seed": args.seed, "steps": args.steps, "lr": args.lr,
              "entropy_weight": args.entropy_weight, "l2_weight": args.l2_weight,
              "rewards_sha256": sha256(args.rewards),
              "plan_sha256": sha256(args.plan),
              "checkpoint_sha256": sha256(args.checkpoint),
              "historical_review_sha256": sha256(args.historical_review),
              "mean_adapter_sha256": rewards["mean_adapter_sha256"],
              "max_steps": 30, "max_new_tokens": 64,
              "train_games": [row["game"] for row in train_rows],
              "dev_games": [row["game"] for row in dev_rows],
              "history": history,
              "final_train": score(train_rows, policy, train_x),
              "final_dev": score(dev_rows, policy, dev_x)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy_state": policy.state_dict(), "source_mean": mean,
                "source_basis": basis, "source_scale": scale,
                "source_games": sorted(latent_by_source),
                "rewards_sha256": result["rewards_sha256"],
                "plan_sha256": result["plan_sha256"],
                "checkpoint_sha256": result["checkpoint_sha256"],
                "historical_review_sha256": result["historical_review_sha256"],
                "mean_adapter_sha256": result["mean_adapter_sha256"],
                "max_steps": 30, "max_new_tokens": 64}, args.save_checkpoint)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"train": result["final_train"], "dev": result["final_dev"]}),
          flush=True)
    return result


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
    parser.add_argument("--data-root", type=Path, default=Path("ttcl/data/alfworld_delta"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.40)
    parser.add_argument("--rewards", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=.05)
    parser.add_argument("--entropy-weight", type=float, default=.01)
    parser.add_argument("--l2-weight", type=float, default=.02)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-checkpoint", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
