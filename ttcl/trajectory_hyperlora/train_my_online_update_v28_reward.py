"""Jointly update event encoder, utility head, and LoRA residual with reward.

This is a train-domain pilot. The frozen Qwen actor samples actions greedily;
stochasticity is an explicit Gaussian parameter code, so the score-function
gradient applies to the parameter policy and not to invented actor logits.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import torch

from ttcl.trajectory_hyperlora.alfworld_actor_generic_v28 import (
    load_agent, run_episode,
)
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v2 import (
    checked, save,
)
from ttcl.trajectory_hyperlora.my_online_update_v28_model import OnlineUpdateModel
from ttcl.trajectory_hyperlora.train_my_online_update_v28 import checked_dataset
from ttcl.trajectory_hyperlora.train_alf_feedback_residual_v1 import (
    ResidualInjection, latent_basis,
)
from ttcl.trajectory_hyperlora.train_alf_future_utility_selector_v1 import (
    target_reward_matrix,
)
from ttcl.trajectory_hyperlora.trajectory_lora_rl import latent_adapter_reinforce


def reviewed(args):
    parent_args = SimpleNamespace(**vars(args))
    parent_args.review = args.source_review
    sources, targets, _ = checked(parent_args)
    source, queries, rewards, base, groups, bindings = checked_dataset(args)
    pretrained = torch.load(args.pretrained, map_location="cpu",
                            weights_only=True)
    if (pretrained["bindings"] != bindings or
            len(sources) != 8 or len(targets) != 18 or
            pretrained["stage"] != "utility_pretrain_only" or
            file_hash(args.checkpoint) != bindings["checkpoint_sha256"]):
        raise ValueError("Changed utility model or source-target lineage")
    rng = random.Random(args.seed)
    pair_ids = [(si, ti) for ti in range(18) for si in range(8)]
    rng.shuffle(pair_ids)
    selected = pair_ids[:args.max_rollouts]
    if len(selected) != args.max_rollouts:
        raise ValueError("Requested more pairs than reviewed")
    fixed = {"method": "我的在线更新 v28 joint reward training",
        "pretrained_sha256": file_hash(args.pretrained),
        "old_review_sha256": file_hash(args.source_review),
        "feature_bindings": bindings,
        "checkpoint_sha256": file_hash(args.checkpoint),
        "runner_sha256": file_hash(Path(__file__)),
        "model_code_sha256": file_hash(Path(__file__).with_name(
            "my_online_update_v28_model.py")),
        "action_guard_sha256": file_hash(Path(__file__).with_name(
            "action_feedback_guard_v28.py")),
        "actor_sha256": file_hash(Path(__file__).with_name(
            "alfworld_actor_generic_v28.py")),
        "seed": args.seed, "max_rollouts": args.max_rollouts,
        "sigma": args.sigma, "lr": args.lr,
        "max_effective_norm": args.max_effective_norm,
        "matrix_penalty": args.matrix_penalty,
        "utility_weight": args.utility_weight,
        "max_steps": 50, "max_new_tokens": 64,
        "history_turns": 2, "repeat_limit": 2}
    bindings_per_pair = []
    for index, (source_id, target_id) in enumerate(selected):
        row = {"index": index, "source_id": source_id,
            "target_id": target_id,
            "source_game": sources[source_id]["game"],
            "source_records_sha256": sources[source_id]["records_sha256"],
            "target_game": targets[target_id]["game"],
            "target_game_sha256": file_hash(args.data_root /
                                            targets[target_id]["game"]),
            "fixed_sha256": digest(fixed)}
        bindings_per_pair.append({**row,
            "input_content_sha256": digest(row),
            "reviewed_target": True})
    return sources, targets, source, queries, rewards, base, pretrained, \
        fixed, bindings_per_pair


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    *_, fixed, pairs = reviewed(args)
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps({"protocol":
        "Fresh input-content-bound train-domain reward rollouts for joint event/utility/LoRA residual learning",
        **fixed, "pairs": pairs, "pairs_sha256": digest(pairs)},
        ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"reviewed_pairs": len(pairs)}), flush=True)


def effective_residual_norm(agent, basis, mean):
    shift = mean @ basis.T
    total = mean.new_tensor(0.)
    for layer, head in zip(agent.adapters, agent.b_heads, strict=True):
        delta_b = torch.nn.functional.linear(shift, head.weight).reshape(
            layer.base.out_features, layer.rank)
        delta_w = delta_b @ layer.a.float() / layer.rank
        total = total + delta_w.square().sum()
    return (total + 1e-12).sqrt()


def train(args):
    if args.output.exists() or args.save_model.exists():
        raise FileExistsError("Use fresh reward-run files")
    if (args.max_rollouts < 1 or args.max_rollouts > 144 or
            args.sigma <= 0 or args.lr <= 0 or
            args.max_effective_norm <= 0 or
            args.matrix_penalty < 0 or args.utility_weight < 0):
        raise ValueError("Invalid reward training budget")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    (sources, targets, source, queries, rewards, base, pretrained,
     fixed, pairs) = reviewed(args)
    review = json.loads(args.review.read_text())
    if (review["pairs"] != pairs or review["pairs_sha256"] != digest(pairs)
            or any(review[key] != value for key, value in fixed.items())):
        raise ValueError("Changed reviewed joint-training targets")
    matrix = json.loads(args.matrices[0].read_text())
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != "contextual" or
            agent.task_context_scope != "current" or
            agent.task_pair_pooling != "mean"):
        raise ValueError("Expected task-conditioned contextual hypernetwork")
    agent.eval()
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    basis_args = SimpleNamespace(**vars(args))
    basis_args.code_dim = pretrained["code_dim"]
    fields, basis = latent_basis(agent, tokenizer, sources, targets,
                                  matrix, basis_args)
    model = OnlineUpdateModel(pretrained["width"], pretrained["hidden"],
                              pretrained["code_dim"]).to(args.device)
    model.load_state_dict(pretrained["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=.01)
    vectors = {key: value.to(args.device) for key, value in source.items()}
    q = queries.to(args.device)
    label = (rewards - base[:, None]).to(args.device)
    weight = 1 + 2 * label.abs()
    old_reward = target_reward_matrix(matrix, 8, 18)
    result = {"protocol": "我的在线更新 v28: jointly train event encoder, shared future-utility scorer, and low-dimensional LoRA residual using real official reward; domain-neutral action-feedback guard",
        "review_sha256": file_hash(args.review),
        "pretrained_sha256": file_hash(args.pretrained),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "actor_sha256": fixed["actor_sha256"],
        "action_guard_sha256": fixed["action_guard_sha256"],
        "sigma": args.sigma, "seed": args.seed,
        "max_rollouts": args.max_rollouts,
        "max_steps": 50, "max_new_tokens": 64,
        "history_turns": 2, "repeat_limit": 2,
        "train": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for entry in pairs:
        si, ti = entry["source_id"], entry["target_id"]
        utilities, codes = model(vectors["global"], vectors["events"],
            vectors["deltas"], vectors["event_counts"], q)
        mean = codes[ti, si]
        if not bool(torch.isfinite(mean).all()):
            raise ValueError("Nonfinite generated LoRA code before sampling")
        sampled = mean.detach() + args.sigma * torch.randn_like(mean)
        raw_effective_norm = effective_residual_norm(
            agent, basis, sampled).detach()
        if not bool(torch.isfinite(raw_effective_norm)):
            raise ValueError("Nonfinite effective LoRA update")
        scale = min(1.0, args.max_effective_norm /
                    max(float(raw_effective_norm), 1e-12))
        bounded_sample = sampled * scale
        shift = bounded_sample @ basis.T
        with ResidualInjection(agent, shift):
            episode = run_episode(agent, tokenizer,
                args.data_root / targets[ti]["game"], fields[si],
                adapter=True, device=args.device, max_steps=50,
                max_new_tokens=64, constrain_actions=True,
                actor_history_turns=2, loop_guard_max=2)
        if (episode["status"] != "complete" or
                episode["initial_observation"] !=
                matrix["base"][ti]["episode"]["initial_observation"]):
            result["failures"].append({"index": entry["index"],
                "raw_sampled_code": sampled.detach().cpu().tolist(),
                "raw_effective_norm": float(raw_effective_norm),
                "clip_scale": scale, "episode": episode})
            save(args.output, result)
            raise RuntimeError("Changed official reward-training episode")
        reward = float(episode["reward"])
        baseline = float(old_reward[ti, si])
        policy_loss = latent_adapter_reinforce(mean, sampled, args.sigma,
                                               reward, baseline)
        utility_loss = (weight * (utilities - label).square()).mean()
        effective_norm = effective_residual_norm(agent, basis, mean)
        loss = (policy_loss + args.utility_weight * utility_loss +
                args.matrix_penalty * effective_norm.square())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if any(not bool(torch.isfinite(value).all())
               for value in model.parameters()):
            raise ValueError("Joint-training parameters became nonfinite")
        result["train"].append({"index": entry["index"],
            "source_id": si, "target_id": ti,
            "input_content_sha256": entry["input_content_sha256"],
            "sampled_code": sampled.detach().cpu().tolist(),
            "raw_effective_norm": float(raw_effective_norm),
            "clip_scale": scale,
            "baseline_old_source_reward": baseline,
            "base_reward": float(base[ti]),
            "sampled_reward": reward,
            "effective_residual_frobenius": float(effective_norm.detach()),
            "utility_loss": float(utility_loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "episode": episode})
        save(args.output, result)
        print(json.dumps({"rollouts": len(result["train"]),
            "reward": reward, "old_source": baseline,
            "base": float(base[ti]), "failures": len(result["failures"])}),
            flush=True)
    args.save_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {key: value.detach().cpu()
                          for key, value in model.state_dict().items()},
        "width": model.width, "hidden": model.hidden,
        "code_dim": model.code_dim, "basis": basis.detach().cpu(),
        "pretrained_sha256": file_hash(args.pretrained),
        "review_sha256": file_hash(args.review),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "stage": "reward_trained"}, args.save_model)
    result["trained_model_sha256"] = file_hash(args.save_model)
    result["summary"] = {"rollouts": len(result["train"]),
        "sampled_wins": sum(row["sampled_reward"] for row in result["train"]),
        "old_source_wins": sum(row["baseline_old_source_reward"]
                               for row in result["train"]),
        "base_wins": sum(row["base_reward"] for row in result["train"]),
        "failures": len(result["failures"])}
    save(args.output, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "train"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--pretrained", type=Path, default=Path(
        "results/trajectory_hyperlora/my_online_update_v28_utility_train_20261009.pt"))
    parser.add_argument("--source-report-seq0", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq0_6_20261007.json"))
    parser.add_argument("--source-report-seq6", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq6_6_20261007.json"))
    parser.add_argument("--plan", type=Path, default=Path(
        "ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--target-policy", choices=("train18",),
                        default="train18")
    parser.add_argument("--checkpoint-training-candidates", type=Path,
        default=Path("results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json"))
    parser.add_argument("--warmstart-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_candidates_20261005.json"))
    parser.add_argument("--prior-train-review", type=Path, default=Path(
        "data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/my_online_update_v28_reward_train_20261009.json"))
    parser.add_argument("--features", type=Path, nargs=2, default=[Path(
        "results/trajectory_hyperlora/alf_future_reward_train8x18_features_20261007.pt"),
        Path("results/trajectory_hyperlora/alf_future_reward_additional8x18_features_v3_20261007.pt")])
    parser.add_argument("--matrices", type=Path, nargs=2, default=[Path(
        "results/trajectory_hyperlora/alf_own_success_future_reward_8x18_20261007.json"),
        Path("results/trajectory_hyperlora/alf_own_success_future_reward_additional8x18_v3_merged_20261007.json")])
    parser.add_argument("--audits", type=Path, nargs=2, default=[Path(
        "results/trajectory_hyperlora/alf_own_success_future_reward_8x18_audited_20261007.json"),
        Path("results/trajectory_hyperlora/alf_own_success_future_reward_additional8x18_v3_merged_audited_20261007.json")])
    parser.add_argument("--reviews", type=Path, nargs=2, default=[Path(
        "data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json"),
        Path("data/annotations/alf_own_success_future_reward_additional8x18_v3_reviewed_20261007.json")])
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/my_online_update_v28_reward_train_20261009.json"))
    parser.add_argument("--save-model", type=Path, default=Path(
        "results/trajectory_hyperlora/my_online_update_v28_reward_train_20261009.pt"))
    parser.add_argument("--seed", type=int, default=311)
    parser.add_argument("--max-rollouts", type=int, default=24)
    parser.add_argument("--sigma", type=float, default=.5)
    parser.add_argument("--max-effective-norm", type=float, default=1.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--utility-weight", type=float, default=.1)
    parser.add_argument("--matrix-penalty", type=float, default=.0001)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.65)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else train)(args)
