"""Independent static rank-4 B-LoRA GRPO on the old 33 train games.

Four on-policy constrained-command episodes form each task group. Official
terminal ALFWorld reward is normalized within the group. The clipped policy
objective updates only the same two static B tensors used by v41 SFT.
This v41b revision reattaches B after rollout cleanup before backpropagation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch

from ttcl.experience_evolution.environment import ACTOR_SYSTEM, clean_command, make_env
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import guarded_commands
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.train_alf_next_task import compact_messages
from ttcl.trajectory_hyperlora.train_alfworld_static_lora_v41 import (
    checked_labels, static_agent,
)


def groups(args):
    dataset = checked_labels(args)
    tasks = list(dataset["tasks"])
    random.Random(args.seed).shuffle(tasks)
    return tasks


def bindings(args, tasks):
    fixed = {"method": "v41 independent static B-LoRA on-policy GRPO",
        "labels_sha256": file_hash(args.labels),
        "source_review_sha256": file_hash(args.source_review),
        "source_training_report_sha256": file_hash(args.source_training),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "trainer_sha256": file_hash(Path(__file__)),
        "sft_helper_sha256": file_hash(Path(__file__).with_name("train_alfworld_static_lora_v41.py")),
        "rank": 4, "layers": 2, "trainable_factors": "B_only",
        "objective": "group_normalized_official_terminal_reward_clipped_token_policy",
        "epochs": 1, "group_size": args.group_size,
        "max_steps": 50, "max_new_tokens": 64,
        "actor_history_turns": 2, "loop_guard_max": 2,
        "sampling_temperature": 1., "top_p": 1., "top_k": 0,
        "cliprange": .2, "kl_coefficient": 0.,
        "lr": args.lr, "weight_decay": args.weight_decay,
        "seed": args.seed}
    targets = []
    for i, task in enumerate(tasks):
        seeds = [args.seed * 100000 + i * args.group_size + j
                 for j in range(args.group_size)]
        content = {"group_index": i, "game": task["target_game"],
            "game_sha256": task["target_game_sha256"],
            "source_training_action_sha256": [x["input_content_sha256"]
                                               for x in task["labels"]],
            "sampling_seeds": seeds,
            "trainer_sha256": fixed["trainer_sha256"],
            "checkpoint_sha256": fixed["checkpoint_sha256"]}
        targets.append({**content, "input_content_sha256": digest(content),
                        "reviewed_target": True})
    return fixed, targets


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    tasks = groups(args)
    fixed, targets = bindings(args, tasks)
    save(args.review, {"protocol": "Exact old 33 official train games; independent zero-B static rank-4 last-two-down-projection GRPO; four on-policy constrained-command rollouts per group; only official terminal reward; reviewed targets and seeds frozen before sampling",
        **fixed, "targets": targets, "targets_sha256": digest(targets)})
    print(json.dumps({"groups": len(targets),
        "declared_rollouts": len(targets) * args.group_size}), flush=True)


def command_paths(tokenizer, content, prompt_ids, allowed_commands,
                  max_new_tokens):
    paths = []
    for command in sorted(set(allowed_commands)):
        full = tokenizer(content + command, add_special_tokens=False).input_ids
        if full[:len(prompt_ids)] != prompt_ids:
            raise ValueError("Command tokenization changed actor prefix")
        suffix = tuple(full[len(prompt_ids):])
        if not suffix or len(suffix) + 1 > max_new_tokens:
            raise ValueError("Admissible command exceeds token budget")
        paths.append(suffix)
    if not paths:
        raise ValueError("No admissible commands")
    return paths


def allowed_next(paths, suffix, eos):
    allowed = {path[len(suffix)] for path in paths
               if len(path) > len(suffix) and path[:len(suffix)] == suffix}
    if any(path == suffix for path in paths):
        allowed.add(eos)
    if not allowed:
        raise ValueError("Generated command escaped allowed trie")
    return sorted(allowed)


def sample_action(agent, tokenizer, actor_messages, allowed_commands,
                  device, max_new_tokens):
    content = tokenizer.apply_chat_template(actor_messages, tokenize=False,
                                            add_generation_prompt=True)
    encoded = tokenizer(content, return_tensors="pt").to(device)
    prompt_ids = encoded.input_ids[0].tolist()
    paths = command_paths(tokenizer, content, prompt_ids, allowed_commands,
                          max_new_tokens)
    eos = tokenizer.eos_token_id

    def constraint(_batch_index, generated_ids):
        return allowed_next(paths, tuple(generated_ids[len(prompt_ids):].tolist()), eos)

    with torch.inference_mode():
        output = agent.model.generate(**encoded, do_sample=True,
            temperature=1., top_p=1., top_k=0,
            max_new_tokens=max_new_tokens, pad_token_id=eos,
            prefix_allowed_tokens_fn=constraint,
            return_dict_in_generate=True, output_scores=True)
    response_ids = output.sequences[0, len(prompt_ids):].tolist()
    if len(response_ids) != len(output.scores) or not response_ids:
        raise ValueError("Missing sampled command token scores")
    allowed_sets = []
    old_log_probs = []
    for j, token in enumerate(response_ids):
        allowed = allowed_next(paths, tuple(response_ids[:j]), eos)
        if token not in allowed:
            raise ValueError("Sampled token outside command trie")
        allowed_sets.append(allowed)
        score = output.scores[j][0].float()
        old_log_probs.append(float(torch.log_softmax(score[allowed], dim=0)[
            allowed.index(token)]))
    response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    event = {"prompt_ids": prompt_ids, "response_ids": response_ids,
        "allowed_token_ids": allowed_sets, "old_log_probs": old_log_probs,
        "raw_response": response,
        "token_binding": digest([prompt_ids, response_ids, allowed_sets])}
    return response, event


def rollout(agent, tokenizer, game, factors, device, sample_seed):
    torch.manual_seed(sample_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(sample_seed)
    agent.model.eval()
    agent.set_source(None)
    for layer, factor in zip(agent.adapters, factors, strict=True):
        layer.b = factor
    env = make_env(game)
    trajectory, events = [], []
    seen = {}
    try:
        state = env.reset()
        initial = str(state["feedback"])
        initial_commands = list(state["admissible_commands"])
        messages = [{"role": "system", "content": ACTOR_SYSTEM}]
        for turn in range(50):
            available = list(state["admissible_commands"])
            feedback = str(state["feedback"])
            allowed = guarded_commands(feedback, available, seen, 2)
            messages.append({"role": "user", "content": feedback +
                "\nAvailable commands:\n" + "\n".join(available)})
            actor_messages = compact_messages(messages, 2)
            response, event = sample_action(agent, tokenizer, actor_messages,
                                            allowed, device, 64)
            command = clean_command(response, available)
            valid = command in available
            key = (feedback, command)
            seen[key] = seen.get(key, 0) + 1
            state, _, done = env.step(command)
            messages.append({"role": "assistant", "content": response})
            trajectory.append({"turn": turn, "response": response,
                "command": command, "valid": valid,
                "observation": str(state["feedback"]), "won": bool(state["won"])})
            events.append(event)
            if done or state["won"]:
                break
        episode = {"status": "complete", "reward": float(bool(state["won"])),
            "steps": len(trajectory), "initial_observation": initial,
            "memory_sha256": hashlib.sha256(b"").hexdigest(),
            "initial_commands_sha256": hashlib.sha256(json.dumps(
                initial_commands).encode()).hexdigest(),
            "invalid_commands": sum(not x["valid"] for x in trajectory),
            "trajectory": trajectory,
            "termination": "success" if state["won"] else "budget_or_environment_done"}
        return episode, events
    finally:
        env.close()
        agent.set_source(None)


def current_log_probs(agent, event, device):
    prompt = event["prompt_ids"]
    response = event["response_ids"]
    allowed_sets = event["allowed_token_ids"]
    if (event["token_binding"] != digest([prompt, response, allowed_sets])
            or len(response) != len(allowed_sets) or
            len(response) != len(event["old_log_probs"])):
        raise ValueError("Changed sampled policy tokens")
    ids = torch.tensor([prompt + response[:-1]], device=device)
    hidden = agent.model.model(input_ids=ids,
        attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
    hidden = hidden[0, len(prompt)-1:len(prompt)-1+len(response)]
    logits = agent.model.lm_head(hidden).float()
    terms = []
    for j, (token, allowed) in enumerate(zip(response, allowed_sets, strict=True)):
        if token not in allowed:
            raise ValueError("Changed constrained sampled response")
        terms.append(logits[j, token] - torch.logsumexp(logits[j, allowed], dim=0))
    return torch.stack(terms)


def update_group(agent, factors, optimizer, episodes, events, device):
    rewards = torch.tensor([x["reward"] for x in episodes], dtype=torch.float32)
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    if float(std) == 0:
        return {"updated": False, "reward_mean": float(mean),
                "reward_std": 0., "loss": None, "policy_tokens": 0}
    advantages = (rewards - mean) / std
    # Rollout closes each environment by clearing generated adapters. Restore
    # the same on-policy static B tensors for differentiable log probabilities.
    for layer, factor in zip(agent.adapters, factors, strict=True):
        layer.b = factor
    total_tokens = sum(len(event["response_ids"]) for group in events for event in group)
    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.
    max_old_diff = 0.
    for group, advantage in zip(events, advantages.tolist(), strict=True):
        for event in group:
            logp = current_log_probs(agent, event, device)
            old = torch.tensor(event["old_log_probs"], device=device)
            difference = float((logp.detach() - old).abs().mean())
            max_old_diff = max(max_old_diff, difference)
            if difference > .15 or not bool(torch.isfinite(logp).all()):
                raise ValueError(f"On-policy constrained logprob mismatch: {difference}")
            ratio = torch.exp(logp - old)
            unclipped = ratio * advantage
            clipped = ratio.clamp(.8, 1.2) * advantage
            loss = -torch.minimum(unclipped, clipped).sum() / total_tokens
            loss.backward()
            loss_total += float(loss.detach())
    norm = torch.nn.utils.clip_grad_norm_(factors, 1., error_if_nonfinite=True)
    optimizer.step()
    return {"updated": True, "reward_mean": float(mean),
        "reward_std": float(std), "loss": loss_total,
        "gradient_norm": float(norm), "policy_tokens": total_tokens,
        "max_old_logprob_difference": max_old_diff}


def train(args):
    if args.output.exists() or args.model_output.exists():
        raise FileExistsError("Use new GRPO output paths")
    tasks = groups(args)
    fixed, targets = bindings(args, tasks)
    review = json.loads(args.review.read_text())
    if (review.get("targets") != targets or review.get("targets_sha256") != digest(targets)
            or any(review.get(k) != v for k, v in fixed.items())):
        raise ValueError("Changed reviewed GRPO groups")
    agent, tokenizer, factors = static_agent(args)
    optimizer = torch.optim.AdamW(factors, lr=args.lr,
                                  weight_decay=args.weight_decay)
    report = {"protocol": review["protocol"], "review_sha256": file_hash(args.review),
              **fixed, "groups": [], "failures": []}
    for i, (task, binding) in enumerate(zip(tasks, targets, strict=True)):
        episodes, events = [], []
        before = factors_hash([x.detach().cpu() for x in factors])
        for seed in binding["sampling_seeds"]:
            try:
                episode, sampled = rollout(agent, tokenizer,
                    args.data_root / binding["game"], factors, args.device, seed)
            except Exception as error:
                report["failures"].append({"group": i, "seed": seed,
                    "error": repr(error)})
                save(args.output, report)
                raise
            episodes.append(episode)
            events.append(sampled)
        result = update_group(agent, factors, optimizer, episodes, events,
                              args.device)
        row = {"group_index": i, "game": binding["game"],
            "input_content_sha256": binding["input_content_sha256"],
            "sampling_seeds": binding["sampling_seeds"],
            "factor_sha256_before": before,
            "factor_sha256_after": factors_hash([x.detach().cpu() for x in factors]),
            "episodes": episodes, "events": events, "update": result}
        report["groups"].append(row)
        report["summary"] = {"groups": len(report["groups"]),
            "rollouts": sum(len(x["episodes"]) for x in report["groups"]),
            "wins": sum(ep["reward"] for x in report["groups"] for ep in x["episodes"]),
            "updated_groups": sum(x["update"]["updated"] for x in report["groups"]),
            "failures": len(report["failures"])}
        save(args.output, report)
        print(json.dumps(report["summary"]), flush=True)
    final = [x.detach().cpu().float().clone() for x in factors]
    torch.save({"factors": final, "factor_shapes": [list(x.shape) for x in final],
        "review_sha256": file_hash(args.review),
        "checkpoint_sha256": fixed["checkpoint_sha256"],
        "trainer_sha256": fixed["trainer_sha256"],
        "rank": 4, "layers": 2,
        "groups": len(tasks), "group_size": args.group_size,
        "seed": args.seed}, args.model_output)
    report["model_sha256"] = file_hash(args.model_output)
    report["factor_sha256"] = factors_hash(final)
    save(args.output, report)
    print(json.dumps({"final": report["summary"],
        "factor_sha256": report["factor_sha256"]}), flush=True)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "train"))
    defaults = {
        "labels": "results/trajectory_hyperlora/alf_sibling_simple_train40_onpolicy_best_labels_20261006.json",
        "source_review": "data/annotations/alf_sibling_simple_train40_onpolicy_best_reviewed_20261006.json",
        "source_training": "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.json",
        "checkpoint": "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt",
        "model": "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507",
        "data_root": "ttcl/data/alfworld_delta",
        "review": "data/annotations/alfworld_static_lora_grpo_v41b_train33_20261010.json",
        "output": "results/trajectory_hyperlora/alfworld_static_lora_grpo_v41b_train33_20261010.json",
        "model_output": "results/trajectory_hyperlora/alfworld_static_lora_grpo_v41b_train33_20261010.pt"}
    for name, value in defaults.items():
        p.add_argument("--" + name.replace("_", "-"), type=Path, default=Path(value))
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--seed", type=int, default=20261010)
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--gpu-fraction", type=float, default=.65)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.group_size < 2 or args.lr <= 0:
        raise ValueError("Invalid GRPO budget")
    (prepare if args.command == "prepare" else train)(args)
