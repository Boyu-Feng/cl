"""Sequential inference-time hypernetwork updates from the actor's own wins.

The first eight environment-confirmed wins bootstrap a fresh hypernetwork.
Each later task is scored before its own outcome can update the model. A frozen
copy reads the same growing source pool to isolate parameter learning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_bootstrap_own_success_v43 import (
    action_rows, examples, new_agent, ordered,
)
from ttcl.trajectory_hyperlora.evaluate_alfworld_bootstrap_own_success_v43 import source_for
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import (
    contextual_source_fields, contextual_text_fields, task_context_text,
)
from ttcl.trajectory_hyperlora.train_alf_next_task import query_ids, target_loss
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay


def bound(args):
    rows = ordered(args)
    if args.bootstrap_wins != 8 or args.bootstrap_steps != 300 or args.online_steps != 20:
        raise ValueError("v52 frozen update budget changed")
    fixed = {"method": "v52 fresh H from eight own wins then success-triggered online updates",
             "source_report_sha256": file_hash(args.source_report),
             "model_config_sha256": file_hash(args.model / "config.json"),
             "runner_sha256": file_hash(Path(__file__)),
             "actor_sha256": file_hash(Path(__file__).with_name("alfworld_zero_shot_probe.py")),
             "seed": args.seed, "rank": 4, "layers": 2,
             "bootstrap_wins": args.bootstrap_wins,
             "bootstrap_steps": args.bootstrap_steps,
             "online_steps": args.online_steps,
             "bootstrap_lr": args.bootstrap_lr,
             "online_lr": args.online_lr,
             "max_steps": 30, "max_new_tokens": 64,
             "actor_history_turns": 2, "loop_guard_max": 2,
             "constrain_actions": True,
             "family_source_min_count": 2}
    targets = []
    for i, row in enumerate(rows):
        content = {"index": i, "game": row["game"],
                   "family": row["family"],
                   "game_sha256": file_hash(args.data_root / row["game"]),
                   "historical_initial_observation_sha256": hashlib.sha256(
                       row["arms"]["base"]["initial_observation"].encode()).hexdigest(),
                   "runner_sha256": fixed["runner_sha256"]}
        targets.append({**content, "input_content_sha256": digest(content),
                        "reviewed_target": True})
    return rows, fixed, targets


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    _, fixed, targets = bound(args)
    save(args.review, {"protocol": "Previously exposed official train240 stream, sequential; each task scored before its own trajectory is used; eight live base wins start fresh H; frozen H reads same growing success pool",
                       **fixed, "targets": targets, "targets_sha256": digest(targets)})
    print(json.dumps({"stream_tasks": len(targets),
                      "bootstrap_wins": args.bootstrap_wins}), flush=True)


def checked(args):
    rows, fixed, targets = bound(args)
    review = json.loads(args.review.read_text())
    if (review.get("targets") != targets or
            review.get("targets_sha256") != digest(targets) or
            any(review.get(k) != v for k, v in fixed.items())):
        raise ValueError("Changed v52 task review")
    return rows, fixed, targets


def trainable_digest(agent):
    h = hashlib.sha256()
    for name, value in agent.named_parameters():
        if value.requires_grad:
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def approved_success(args, target, episode, stream_index, annotation_path):
    game = args.data_root / target["game"]
    replay(game, episode)
    actions = action_rows(game, episode)
    records = records_from_episode(episode)
    content = {"stream_index": stream_index,
               "game": target["game"], "family": target["family"],
               "game_sha256": target["game_sha256"],
               "episode_sha256": digest(episode),
               "source_records_sha256": digest(records),
               "actions_sha256": digest(actions),
               "runner_sha256": file_hash(Path(__file__))}
    annotation = {**content, "input_content_sha256": digest(content),
        "reviewed_target": True,
        "review_note": "Official reward 1; reset, admissible commands, feedback and win replay verified"}
    history = json.loads(annotation_path.read_text()) if annotation_path.exists() else {
        "protocol": "v52 new live own-win targets; reviewed only after exact environment replay",
        "eval_review_sha256": file_hash(args.review), "annotations": []}
    if history.get("eval_review_sha256") != file_hash(args.review) or any(
            x["stream_index"] == stream_index for x in history["annotations"]):
        raise ValueError("Changed or duplicate v52 annotation")
    history["annotations"].append(annotation)
    save(annotation_path, history)
    return {"stream_index": stream_index, "game": target["game"],
            "family": target["family"], "game_sha256": target["game_sha256"],
            "episode_sha256": digest(episode), "source_records": records,
            "actions": actions, "annotation_input_content_sha256": annotation["input_content_sha256"]}


def action_fields(agent, tokenizer, action, initial, device):
    prefix = query_ids(tokenizer, action, history_turns=2, max_prompt_tokens=3000)
    fields = contextual_text_fields(agent, tokenizer,
        task_context_text(initial, action["current_observation"]),
        device, 2048, pooling="both")
    return prefix, fields


def fit_examples(agent, tokenizer, optimizer, successes, items, steps, rng, device):
    if not items:
        return {"steps": 0, "loss_mean": None}
    agent.train()
    sources = {}
    fields = {}
    for j, i, action in items:
        if j not in sources:
            sources[j] = contextual_source_fields(agent, tokenizer,
                successes[j]["source_records"], device, 2048, pooling="both")
        key = digest([i, action["target_messages"]])
        if key not in fields:
            initial = successes[i]["actions"][0]["current_observation"]
            fields[key] = action_fields(agent, tokenizer, action, initial, device)
    order = list(range(len(items)))
    losses = []
    for step in range(steps):
        if step % len(order) == 0:
            rng.shuffle(order)
        j, i, action = items[order[step % len(order)]]
        prefix, target_fields = fields[digest([i, action["target_messages"]])]
        agent.set_source(sources[j], target_fields=target_fields)
        loss = target_loss(agent, tokenizer, action, prefix, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in agent.parameters() if p.requires_grad),
                                       1., error_if_nonfinite=True)
        optimizer.step()
        agent.set_source(None)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite online H loss")
        losses.append(float(loss.detach()))
    agent.eval()
    return {"steps": steps, "loss_mean": sum(losses) / len(losses)}


def evaluate(args):
    if any(path.exists() for path in (args.output, args.annotations, args.model_output)):
        raise FileExistsError("Use fresh v52 output paths")
    rows, fixed, targets = checked(args)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction, device=args.device)
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                                  device=args.frozen_device)
    agent, tokenizer = new_agent(args)
    agent.eval()
    rng = random.Random(args.seed)
    successes = []
    frozen = None
    frozen_tokenizer = None
    optimizer = None
    report = {"protocol": "v52 true sequential own-trajectory hypernetwork updates; previously exposed official train stream",
              "review_sha256": file_hash(args.review), **fixed,
              "rows": [], "failures": [], "bootstrap": None}
    for i, (source, target) in enumerate(zip(rows, targets, strict=True)):
        game = args.data_root / target["game"]
        count = sum(s["family"] == target["family"] for s in successes)
        active = frozen is not None and count >= 2
        j = source_for(target, successes) if count else None
        memory_before = len(successes)
        params_before = trainable_digest(agent) if frozen is not None else None
        if active:
            source_fields = contextual_source_fields(agent, tokenizer,
                successes[j]["source_records"], args.device, 2048, pooling="both")
            online = run_episode(agent, tokenizer, game, source_fields,
                adapter=True, device=args.device, max_steps=30,
                max_new_tokens=64, constrain_actions=True,
                actor_history_turns=2, loop_guard_max=2)
            frozen_fields = contextual_source_fields(frozen, frozen_tokenizer,
                successes[j]["source_records"], args.frozen_device, 2048,
                pooling="both")
            frozen_ep = run_episode(frozen, frozen_tokenizer, game, frozen_fields,
                adapter=True, device=args.frozen_device, max_steps=30,
                max_new_tokens=64, constrain_actions=True,
                actor_history_turns=2, loop_guard_max=2)
        else:
            online = run_episode(agent, tokenizer, game, {}, adapter=False,
                device=args.device, max_steps=30, max_new_tokens=64,
                constrain_actions=True, actor_history_turns=2,
                loop_guard_max=2)
            frozen_ep = online
        if (online["status"] != "complete" or frozen_ep["status"] != "complete" or
                online["initial_observation"] != source["arms"]["base"]["initial_observation"] or
                frozen_ep["initial_observation"] != online["initial_observation"]):
            report["failures"].append({"index": i, "online_status": online["status"],
                                       "frozen_status": frozen_ep["status"]})
            save(args.output, report)
            raise RuntimeError("Incomplete v52 episode")
        update = {"steps": 0, "loss_mean": None}
        annotation_hash = None
        if online["reward"] == 1:
            item = approved_success(args, target, online, i, args.annotations)
            annotation_hash = item["annotation_input_content_sha256"]
            if frozen is None:
                successes.append(item)
                if len(successes) == args.bootstrap_wins:
                    items = examples(successes)
                    optimizer = torch.optim.AdamW(
                        (p for p in agent.parameters() if p.requires_grad),
                        lr=args.bootstrap_lr, weight_decay=.01)
                    report["bootstrap"] = {"last_stream_index": i,
                        "own_wins": len(successes), "cross_task_action_targets": len(items),
                        "parameters_before_sha256": trainable_digest(agent)}
                    update = fit_examples(agent, tokenizer, optimizer, successes,
                                          items, args.bootstrap_steps, rng, args.device)
                    report["bootstrap"].update(update)
                    report["bootstrap"]["parameters_after_sha256"] = trainable_digest(agent)
                    frozen_args = SimpleNamespace(**vars(args))
                    frozen_args.device = args.frozen_device
                    frozen, frozen_tokenizer = new_agent(frozen_args)
                    frozen_state = {name: p.detach().cpu() for name, p in agent.named_parameters()
                                    if p.requires_grad}
                    with torch.no_grad():
                        for name, p in frozen.named_parameters():
                            if p.requires_grad:
                                p.copy_(frozen_state[name].to(p.device, p.dtype))
                    frozen.eval()
                    optimizer = torch.optim.AdamW(
                        (p for p in agent.parameters() if p.requires_grad),
                        lr=args.online_lr, weight_decay=.01)
            else:
                if j is not None:
                    next_item = len(successes)
                    fit_pool = successes + [item]
                    update = fit_examples(agent, tokenizer, optimizer, fit_pool,
                        [(j, next_item, action) for action in item["actions"]],
                        args.online_steps, rng, args.device)
                successes.append(item)
        entry = {"index": i, "game": target["game"],
            "target_input_content_sha256": target["input_content_sha256"],
            "memory_before": memory_before, "source_index": j,
            "source_episode_sha256": successes[j]["episode_sha256"] if j is not None else None,
            "adapter_active": active, "parameters_before_sha256": params_before,
            "parameters_after_sha256": trainable_digest(agent) if frozen is not None else None,
            "annotation_input_content_sha256": annotation_hash,
            "update": update, "online": online, "frozen_same_sources": frozen_ep}
        report["rows"].append(entry)
        report["summary"] = {"n": len(report["rows"]),
            "online": sum(r["online"]["reward"] for r in report["rows"]),
            "frozen_same_sources": sum(r["frozen_same_sources"]["reward"] for r in report["rows"]),
            "own_successes": len(successes),
            "online_updates": sum(r["update"]["steps"] for r in report["rows"]),
            "failures": len(report["failures"])}
        save(args.output, report)
        print(json.dumps(report["summary"]), flush=True)
    checkpoint = {"rank": 4, "layers": 2, "source_encoder": "contextual_lm",
        "encoder_kind": "contextual", "relation_bottleneck": "none",
        "context_mode": "none", "context_strength": 0.,
        "task_conditioned": True, "task_pair_pooling": "mean",
        "task_context_scope": "current", "contextual_source_max_tokens": 2048,
        "trainable_state": {name: p.detach().cpu().clone()
            for name, p in agent.named_parameters() if p.requires_grad},
        "review_sha256": file_hash(args.review),
        "report_sha256": file_hash(args.output),
        "runner_sha256": fixed["runner_sha256"]}
    torch.save(checkpoint, args.model_output)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "evaluate"))
    paths = {
        "source_report": "results/trajectory_hyperlora/alf_sibling_train240_taskpair_current1000_20261006.json",
        "model": "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507",
        "data_root": "ttcl/data/alfworld_delta",
        "review": "data/annotations/alfworld_online_own_trajectory_v52_stream240_20261010.json",
        "annotations": "data/annotations/alfworld_online_own_trajectory_v52_own_wins_20261010.json",
        "output": "results/trajectory_hyperlora/alfworld_online_own_trajectory_v52_stream240_20261010.json",
        "model_output": "results/trajectory_hyperlora/alfworld_online_own_trajectory_v52_20261010.pt",
    }
    for k, value in paths.items():
        p.add_argument("--" + k.replace("_", "-"), type=Path, default=Path(value))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--frozen-device", default="cuda:1")
    p.add_argument("--gpu-fraction", type=float, default=.80)
    p.add_argument("--bootstrap-wins", type=int, default=8)
    p.add_argument("--bootstrap-steps", type=int, default=300)
    p.add_argument("--online-steps", type=int, default=20)
    p.add_argument("--bootstrap-lr", type=float, default=1e-4)
    p.add_argument("--online-lr", type=float, default=1e-5)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    (prepare if args.command == "prepare" else evaluate)(args)
