"""One-shot inference-stream bootstrap from the base actor's own successes.

The first half of an interleaved official TRAIN stream is a chronological
bootstrap prefix. A fresh contextual hypernetwork and a fresh static B-LoRA
share its successful action targets. The second half is evaluated without
updating either model. This is a train-stream mechanism test, not unseen score.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.experience_evolution.environment import ACTOR_SYSTEM, make_env
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import run_episode
from ttcl.trajectory_hyperlora.direct_composition_pilot import DirectRelationHyperLoRA
from ttcl.trajectory_hyperlora.contextual_alf_source import (
    contextual_source_fields, contextual_text_fields, task_context_text,
)
from ttcl.trajectory_hyperlora.train_alf_next_task import query_ids, target_loss
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay


def ordered(args):
    raw = json.loads(args.source_report.read_text())
    rows = raw["games"]
    if (len(rows) != 240 or raw["summary"] != {"n": 240, "failures": 0,
            "base": 57.0, "own": 73.0, "wrong": 77.0} or raw["failures"]):
        raise ValueError("Changed training stream source")
    groups = defaultdict(list)
    for row in rows:
        if "/train/" not in "/" + row["game"]:
            raise ValueError("Nontrain task in inference stream")
        groups[row["family"]].append(row)
    if len(groups) != 6 or any(len(v) != 40 for v in groups.values()):
        raise ValueError("Expected six 40-game families")
    order = [groups[family][i] for i in range(40)
             for family in sorted(groups)]
    if len({x["game"] for x in order}) != 240:
        raise ValueError("Repeated task")
    return order


def action_rows(game, episode):
    """Recreate actor prompts and verify own successful actions in the env."""
    if episode["status"] != "complete" or episode["reward"] != 1:
        raise ValueError("Only completed own wins may become targets")
    env = make_env(game)
    result = []
    try:
        state = env.reset()
        if str(state["feedback"]) != episode["initial_observation"]:
            raise ValueError("Changed initial observation")
        messages = [{"role": "system", "content": ACTOR_SYSTEM}]
        for step in episode["trajectory"]:
            command = step["command"]
            if command not in state["admissible_commands"] or not step["valid"]:
                raise ValueError("Changed successful action")
            current = str(state["feedback"])
            messages.append({"role": "user", "content": current +
                "\nAvailable commands:\n" +
                "\n".join(state["admissible_commands"])})
            result.append({"target_messages": [dict(x) for x in messages],
                "target_action": command, "current_observation": current})
            state, _, done = env.step(command)
            if (str(state["feedback"]) != step["observation"] or
                    bool(state["won"]) != step["won"]):
                raise ValueError("Changed success feedback")
            messages.append({"role": "assistant", "content": step["response"]})
        if not state["won"] or episode["steps"] != len(result):
            raise ValueError("Changed own success")
    finally:
        env.close()
    return result


def inputs(args):
    rows = ordered(args)
    prefix, suffix = rows[:120], rows[120:]
    successes = []
    for stream_index, row in enumerate(prefix):
        ep = row["arms"]["base"]
        if not ep["reward"]:
            continue
        replay(args.data_root / row["game"], ep)
        actions = action_rows(args.data_root / row["game"], ep)
        successes.append({"stream_index": stream_index, "game": row["game"],
            "family": row["family"], "episode_sha256": digest(ep),
            "game_sha256": file_hash(args.data_root / row["game"]),
            "source_records": records_from_episode(ep),
            "actions": actions})
    if len(successes) != 31 or sum(len(x["actions"]) for x in successes) != 384:
        raise ValueError("Changed base-own-success bootstrap")
    return rows, successes, suffix


def bindings(args, successes, suffix):
    fixed = {"method": "v43 fresh hypernetwork vs fresh static B from own wins",
        "source_report_sha256": file_hash(args.source_report),
        "runner_sha256": file_hash(Path(__file__)),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "actor_sha256": file_hash(Path(__file__).with_name("alfworld_zero_shot_probe.py")),
        "stream_order": "six-family-interleave, first120 bootstrap, last120 evaluation",
        "seed": args.seed, "steps": args.steps,
        "hyper_lr": args.hyper_lr, "static_lr": args.static_lr,
        "rank": 4, "layers": 2, "max_steps": 30,
        "max_new_tokens": 64, "constrain_actions": True,
        "actor_history_turns": None, "loop_guard_max": None}
    sources = []
    for i, item in enumerate(successes):
        content = {"success_index": i, "stream_index": item["stream_index"],
            "game": item["game"], "family": item["family"],
            "game_sha256": item["game_sha256"],
            "episode_sha256": item["episode_sha256"],
            "source_records_sha256": digest(item["source_records"]),
            "actions_sha256": digest(item["actions"]),
            "runner_sha256": fixed["runner_sha256"]}
        sources.append({**content, "input_content_sha256": digest(content),
                        "reviewed_target": True})
    targets = []
    for i, row in enumerate(suffix):
        content = {"index": i, "game": row["game"], "family": row["family"],
            "game_sha256": file_hash(args.data_root / row["game"]),
            "old_base_episode_sha256": digest(row["arms"]["base"]),
            "runner_sha256": fixed["runner_sha256"]}
        targets.append({**content, "input_content_sha256": digest(content),
                        "reviewed_target": True})
    return fixed, sources, targets


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    _, successes, suffix = inputs(args)
    fixed, sources, targets = bindings(args, successes, suffix)
    save(args.review, {"protocol": "Previously recorded base-only train-stream own successes rechecked in original environment; first120 tasks supply 31 own wins and 384 action labels, after which fresh models train once and freeze for last120 train tasks; no unseen tasks or old hypernetwork weights used",
        **fixed, "sources": sources, "sources_sha256": digest(sources),
        "targets": targets, "targets_sha256": digest(targets)})
    print(json.dumps({"own_successes": len(successes),
                      "success_actions": sum(len(x["actions"]) for x in successes),
                      "future_targets": len(targets)}), flush=True)


def checked(args):
    rows, successes, suffix = inputs(args)
    fixed, sources, targets = bindings(args, successes, suffix)
    review = json.loads(args.review.read_text())
    if (review.get("sources") != sources or review.get("targets") != targets or
            review.get("sources_sha256") != digest(sources) or
            review.get("targets_sha256") != digest(targets) or
            any(review.get(k) != v for k, v in fixed.items())):
        raise ValueError("Changed own-success annotations")
    return rows, successes, suffix, fixed, sources, targets


def new_agent(args):
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(str(args.model),
        torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    base.config.use_cache = False
    base.generation_config.temperature = 1.0
    base.generation_config.top_p = 1.0
    base.generation_config.top_k = 50
    agent = DirectRelationHyperLoRA(base, rank=4, layers=2,
        encoder_kind="contextual", task_conditioned=True).to(args.device)
    agent.contextual_source_max_tokens = 2048
    agent.task_pair_pooling = "mean"
    agent.task_context_scope = "current"
    return agent, tokenizer


def examples(successes):
    pool = defaultdict(list)
    for i, row in enumerate(successes):
        pool[row["family"]].append(i)
    result = []
    for i, target in enumerate(successes):
        eligible = [j for j in pool[target["family"]] if j != i]
        if not eligible:
            continue
        for action in target["actions"]:
            j = min(eligible, key=lambda k: digest([
                "v43_source", target["game"], successes[k]["game"]]))
            result.append((j, i, action))
    if not result:
        raise ValueError("No cross-task success pairs")
    return result


def train(args):
    if args.hyper_model.exists() or args.static_model.exists() or args.train_report.exists():
        raise FileExistsError("Use fresh v43 model and report paths")
    _, successes, _, fixed, _, _ = checked(args)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction, device=args.device)
    agent, tokenizer = new_agent(args)
    items = examples(successes)
    source_fields = [contextual_source_fields(agent, tokenizer,
        row["source_records"], args.device, 2048, pooling="both")
        for row in successes]
    target_fields = {}
    prefixes = {}
    for i, row in enumerate(successes):
      for action in row["actions"]:
        key = digest([i, action["target_messages"]])
        if key not in prefixes:
            initial = successes[i]["actions"][0]["current_observation"]
            target_fields[key] = contextual_text_fields(agent, tokenizer,
                task_context_text(initial, action["current_observation"]),
                args.device, 2048, pooling="both")
            prefixes[key] = query_ids(tokenizer, action, history_turns=2,
                                      max_prompt_tokens=3000)
    initial_a = [layer.a.detach().cpu().float().clone() for layer in agent.adapters]
    optimizer = torch.optim.AdamW((p for p in agent.parameters() if p.requires_grad),
        lr=args.hyper_lr, weight_decay=.01)
    rng = random.Random(args.seed)
    order = list(range(len(items)))
    losses = []
    for step in range(args.steps):
        if step % len(order) == 0:
            rng.shuffle(order)
        j, i, action = items[order[step % len(order)]]
        key = digest([i, action["target_messages"]])
        agent.set_source(source_fields[j], target_fields=target_fields[key])
        loss = target_loss(agent, tokenizer, action, prefixes[key], args.device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in agent.parameters()
                                         if p.requires_grad), 1.)
        optimizer.step()
        agent.set_source(None)
        losses.append(float(loss.detach()))
        if (step + 1) % 50 == 0:
            print(json.dumps({"arm": "hyper", "step": step + 1,
                "last50_loss": sum(losses[-50:]) / 50}), flush=True)
    checkpoint = {"rank": 4, "layers": 2, "source_encoder": "contextual_lm",
        "encoder_kind": "contextual", "relation_bottleneck": "none",
        "context_mode": "none", "context_strength": 0.,
        "task_conditioned": True, "task_pair_pooling": "mean",
        "task_context_scope": "current", "contextual_source_max_tokens": 2048,
        "trainable_state": {name: p.detach().cpu().clone()
            for name, p in agent.named_parameters() if p.requires_grad},
        "review_sha256": file_hash(args.review),
        "trainer_sha256": fixed["runner_sha256"],
        "bootstrap_own_successes": len(successes)}
    args.hyper_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.hyper_model)
    # Reinitialize the actor and A factors from the exact same seed. Static B
    # sees the same target actions, while all source information is omitted.
    del agent
    torch.cuda.empty_cache()
    static_agent, _ = new_agent(args)
    if any(not torch.equal(layer.a.detach().cpu().float(), a)
           for layer, a in zip(static_agent.adapters, initial_a, strict=True)):
        raise ValueError("Static A initialization differs")
    factors = torch.nn.ParameterList([
        torch.nn.Parameter(torch.zeros(1, layer.base.out_features, layer.rank,
                                       device=args.device))
        for layer in static_agent.adapters])
    static_opt = torch.optim.AdamW(factors, lr=args.static_lr, weight_decay=.01)
    represented_targets = {i for _, i, _ in items}
    static_items = [(i, action) for i, row in enumerate(successes)
                    if i in represented_targets for action in row["actions"]]
    if len(static_items) != len(items):
        raise ValueError("Static and hypernetwork action targets differ")
    order = list(range(len(static_items)))
    rng = random.Random(args.seed)
    static_losses = []
    for step in range(args.steps):
        if step % len(order) == 0:
            rng.shuffle(order)
        i, action = static_items[order[step % len(order)]]
        key = digest([i, action["target_messages"]])
        for layer, factor in zip(static_agent.adapters, factors, strict=True):
            layer.b = factor
        loss = target_loss(static_agent, tokenizer, action,
                           prefixes[key], args.device)
        static_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(factors, 1.)
        static_opt.step()
        static_agent.set_source(None)
        static_losses.append(float(loss.detach()))
        if (step + 1) % 50 == 0:
            print(json.dumps({"arm": "static", "step": step + 1,
                "last50_loss": sum(static_losses[-50:]) / 50}), flush=True)
    torch.save({"rank": 4, "layers": 2, "A": initial_a,
        "factors": [x.detach().cpu().float().clone() for x in factors],
        "review_sha256": file_hash(args.review),
        "trainer_sha256": fixed["runner_sha256"]}, args.static_model)
    save(args.train_report, {"protocol": "Fresh equal-step models trained only from base actor's own successful prefix episodes, no pretrained hypernetwork or target-suffix actions",
        **fixed, "review_sha256": file_hash(args.review),
        "hyper_model_sha256": file_hash(args.hyper_model),
        "static_model_sha256": file_hash(args.static_model),
        "own_successes": len(successes), "hyper_cross_task_actions": len(items),
        "static_actions": len(static_items),
        "hyper_last50_loss": sum(losses[-50:]) / min(50, len(losses)),
        "static_last50_loss": sum(static_losses[-50:]) / min(50, len(static_losses))})


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "train"))
    defaults = {"source_report": "results/trajectory_hyperlora/alf_sibling_train240_taskpair_current1000_20261006.json",
        "model": "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507",
        "data_root": "ttcl/data/alfworld_delta",
        "review": "data/annotations/alfworld_bootstrap_own_success_v43_train240_20261010.json",
        "hyper_model": "results/trajectory_hyperlora/alfworld_bootstrap_own_success_v43_hyper_20261010.pt",
        "static_model": "results/trajectory_hyperlora/alfworld_bootstrap_own_success_v43_static_20261010.pt",
        "train_report": "results/trajectory_hyperlora/alfworld_bootstrap_own_success_v43_train_20261010.json"}
    for name, value in defaults.items():
        p.add_argument("--" + name.replace("_", "-"), type=Path, default=Path(value))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--hyper-lr", type=float, default=1e-4)
    p.add_argument("--static-lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda:6")
    p.add_argument("--gpu-fraction", type=float, default=.65)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    (prepare if args.command == "prepare" else train)(args)
